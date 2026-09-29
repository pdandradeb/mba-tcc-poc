"""Loopback-only HTTP UI. Static assets are local; no provider calls are made."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from socketserver import TCPServer
from urllib.parse import urlparse
from .store import ResponseStore, Conflict

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / 'static'


class LoopbackHTTPServer(HTTPServer):
    def server_bind(self):
        # Avoid reverse DNS: this server is deliberately bound only to loopback.
        TCPServer.server_bind(self)
        self.server_name = 'localhost'
        self.server_port = self.server_address[1]


def make_server(store, port=8081, static=STATIC, export_filename="avaliacao-respostas.json"):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def respond(self, value, status=200, content_type='application/json; charset=utf-8', download=False):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if content_type.startswith('application/json') else value
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            if download: self.send_header('Content-Disposition', f'attachment; filename="{export_filename}"')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if not self.allowed_host(): return self.respond({'error':'Host não autorizado.'},403)
            path=urlparse(self.path).path
            if path in ['/', '/app.js', '/style.css']:
                name, mime = {'/':('index.html','text/html; charset=utf-8'), '/app.js':('app.js','text/javascript; charset=utf-8'), '/style.css':('style.css','text/css; charset=utf-8')}[path]
                return self.respond((static/name).read_bytes(),content_type=mime)
            if path=='/api/items': return self.respond({**store.listing(),'csrf_token':token})
            if path=='/api/sources': return self.respond([store.source(i) for i in store.corpus])
            if path=='/api/export': return self.respond(store.export(),download=True)
            if path.startswith('/api/items/'):
                try: return self.respond(store.detail(path.removeprefix('/api/items/')))
                except KeyError: return self.respond({'error':'Resposta não encontrada.'},404)
            return self.respond({'error':'Página não encontrada.'},404)

        def do_POST(self):
            expected_origins={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.allowed_host() or self.headers.get('Origin') not in expected_origins or self.headers.get('X-Review-Token')!=token:
                return self.respond({'error':'Requisição não autorizada. Reabra a interface local.'},403)
            path=urlparse(self.path).path
            if not path.startswith('/api/items/'):
                return self.respond({'error':'Rota não encontrada.'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1_000_000: raise ValueError('Tamanho da requisição inválido.')
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('Envie JSON.')
                payload=json.loads(self.rfile.read(size))
                return self.respond(store.save(path.removeprefix('/api/items/'),payload))
            except Conflict as exc: return self.respond({'error':str(exc)},409)
            except KeyError: return self.respond({'error':'Resposta não encontrada.'},404)
            except (ValueError,TypeError,UnicodeError) as exc: return self.respond({'error':str(exc)},400)
            except OSError: return self.respond({'error':'Não foi possível gravar. Sua avaliação não foi confirmada; verifique o disco.'},500)

    return LoopbackHTTPServer(('127.0.0.1',port), Handler)


def main(store_class=ResponseStore, static=STATIC, default_port=8081,
         default_store="local_reviews/responses.json", export_filename="avaliacao-respostas.json"):
    parser=argparse.ArgumentParser(description='Avaliar respostas registradas, sem executar motores ou consultar APIs.')
    parser.add_argument('--results',type=Path,required=True,help='Arquivo results.json da execução a avaliar.')
    parser.add_argument('--cases',type=Path,default=Path('data/cases_reviewed.jsonl'))
    parser.add_argument('--corpus',type=Path,default=Path('data/public_corpus.json'))
    parser.add_argument('--store',type=Path,default=Path(default_store),help='Use um arquivo diferente para cada avaliador.')
    parser.add_argument('--port',type=int,default=default_port)
    args=parser.parse_args()
    def local(p):return p if p.is_absolute() else ROOT/p
    try:
        store=store_class(local(args.results),local(args.cases),local(args.corpus),local(args.store))
        server=make_server(store,args.port,static,export_filename)
    except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
    print(f'Avaliação de respostas: http://127.0.0.1:{server.server_port}',flush=True)
    print(f'{len(store.queue)} respostas. Avaliações salvas em {store.path}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
