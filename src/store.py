from dataclasses import dataclass, field
import json
import re
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from .corpus import load_corpus
from .config import EMBEDDING_DIMENSION

@dataclass
class Chunk:
    chunk_id: str
    title: str
    content: str
    similarity_score: float
    metadata: dict = field(default_factory=dict)

class PostgresKnowledgeStore:
    def __init__(self, database_url, embedding=None, data_dir=None):
        self.database_url = database_url
        self.embedding = embedding
        self.corpus_id, self.chunks = load_corpus() if data_dir is None else load_corpus(data_dir)
        self.events = []

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=10)

    def initialize(self, *, rebuild_vectors=False):
        with self.connect() as db:
            db.execute((Path(__file__).resolve().parents[1]/'sql/001_knowledge.sql').read_text())
            vector_type = db.execute("""SELECT format_type(atttypid, atttypmod) AS type
                FROM pg_attribute WHERE attrelid='kb_embeddings'::regclass
                AND attname='embedding' AND NOT attisdropped""").fetchone()['type']
            if rebuild_vectors:
                # Only derived vectors are rebuilt; public documents and chunks remain intact.
                db.execute('LOCK TABLE kb_embeddings IN ACCESS EXCLUSIVE MODE')
                db.execute('TRUNCATE kb_embeddings')
                db.execute('ALTER TABLE kb_embeddings ALTER COLUMN embedding TYPE vector(1536)')
            elif vector_type != f'vector({EMBEDDING_DIMENSION})':
                raise ValueError('Vector dimension mismatch; run index --rebuild-vectors')
            for c in self.chunks:
                db.execute('INSERT INTO kb_documents VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                           (self.corpus_id,c['source_path'],c['title'],'public'))
                db.execute('INSERT INTO kb_chunks(corpus_id,id,source_path,title,content,content_hash) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                           (self.corpus_id,c['chunk_id'],c['source_path'],c['title'],c['content'],c['content_hash']))

    def index_embeddings(self):
        if self.embedding is None:
            raise ValueError('An explicit neural embedding provider is required')
        with self.connect() as db:
            existing={r['chunk_id'] for r in db.execute('SELECT chunk_id FROM kb_embeddings WHERE corpus_id=%s AND model=%s AND execution_kind=%s',
                (self.corpus_id,self.embedding.model,self.embedding.execution_kind))}
        missing=[c for c in self.chunks if c['chunk_id'] not in existing]
        for offset in range(0,len(missing),16):
            batch=missing[offset:offset+16]
            vectors=self.embedding.embed([c['title']+'\n'+c['content'] for c in batch])
            with self.connect() as db:
                for c,v in zip(batch,vectors,strict=True):
                    db.execute('INSERT INTO kb_embeddings VALUES (%s,%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING',
                        (self.corpus_id,c['chunk_id'],self.embedding.model,self.embedding.execution_kind,json.dumps(v)))
        return len(missing)

    def search(self, query, *, method, top_k=3):
        if top_k < 1: raise ValueError('top_k must be positive')
        if method not in {'fts','dense'}: raise ValueError('Unknown retrieval method')
        with self.connect() as db:
            if method=='fts':
                terms=re.findall(r'[^\W_]+',query.lower())
                tsquery=' | '.join(dict.fromkeys(t for t in terms if len(t)>2))
                rows=db.execute('''SELECT c.*, ts_rank_cd(c.search_vector, to_tsquery('portuguese', %s)) AS score
                    FROM kb_chunks c JOIN kb_documents d USING(corpus_id,source_path)
                    WHERE c.corpus_id=%s AND d.visibility='public'
                    AND c.search_vector @@ to_tsquery('portuguese', %s)
                    ORDER BY score DESC, c.id LIMIT %s''',(tsquery,self.corpus_id,tsquery,top_k)).fetchall()
            else:
                if self.embedding is None: raise ValueError('Missing neural embedding provider')
                count=db.execute('SELECT count(*) AS n FROM kb_embeddings WHERE corpus_id=%s AND model=%s AND execution_kind=%s',
                    (self.corpus_id,self.embedding.model,self.embedding.execution_kind)).fetchone()['n']
                if count!=len(self.chunks): raise ValueError('Vector index is incomplete; run index first')
                vector=json.dumps(self.embedding.embed([query])[0])
                rows=db.execute('''SELECT c.*, 1-(e.embedding <=> %s::vector) AS score
                    FROM kb_chunks c JOIN kb_documents d USING(corpus_id,source_path)
                    JOIN kb_embeddings e ON e.corpus_id=c.corpus_id AND e.chunk_id=c.id
                    WHERE c.corpus_id=%s AND d.visibility='public' AND e.model=%s AND e.execution_kind=%s
                    ORDER BY e.embedding <=> %s::vector, c.id LIMIT %s''',
                    (vector,self.corpus_id,self.embedding.model,self.embedding.execution_kind,vector,top_k)).fetchall()
        self.events.append({'component':'retrieval','backend':'postgres_fts' if method=='fts' else 'pgvector_cosine',
            'corpus_sha256':self.corpus_id,'query':query,'top_k':top_k,
            'chunks':[{'id':r['id'],'score':r['score'],'content_hash':r['content_hash']} for r in rows]})
        return [Chunk(r['id'],r['title'],r['content'],r['score'],{'source_path':r['source_path'],'content_hash':r['content_hash']}) for r in rows]
