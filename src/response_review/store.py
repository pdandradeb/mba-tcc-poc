"""Immutable experiment inputs, blinded queue, validated and revisioned judgments."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import secrets
import tempfile

FIELDS = {
    'next_best_action': 'Próximo passo',
    'strategic_rationale': 'Justificativa',
    'primary_risk': 'Risco prioritário',
    'risk_mitigation': 'Mitigação',
    'suggested_talk_track': 'Minuta para o consumidor',
    'high_construal_guidance': 'Orientação de alto nível',
    'low_construal_template': 'Orientação operacional',
    'anti_manipulation_flags': 'Observações de antimanipulação',
}
COVERAGE = {'full', 'partial', 'missing', 'not_applicable', 'uncertain'}
FORBIDDEN = {'absent', 'present', 'uncertain'}
OVERALL = {'no_issues', 'issues', 'uncertain'}
ISSUES = {'unsupported', 'contradiction', 'incorrect_value', 'unconfirmed_action', 'manipulation', 'other'}

class Conflict(ValueError):
    pass

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

def now():
    return datetime.now(timezone.utc).isoformat()

def normalized(text):
    return ' '.join(text.split())

def blank_review(case):
    return {
        'coverage': [{'verdict': '', 'quote': '', 'reason': ''} for _ in case['evaluation_rubric']['expected_facts']],
        'forbidden': [{'verdict': '', 'quote': '', 'reason': ''} for _ in case['evaluation_rubric']['forbidden_claims']],
        'overall': '', 'issues': [], 'notes': '',
    }

class ResponseStore:
    SCHEMA_VERSION = 1

    def __init__(self, results, cases, corpus, store):
        self.paths = [Path(p).resolve() for p in (results, cases, corpus)]
        self.path = Path(store).resolve()
        if self.path in self.paths:
            raise ValueError('O arquivo de avaliações deve ser diferente dos arquivos de entrada.')
        raw = [p.read_bytes() for p in self.paths]
        self.fingerprints = dict(zip(['results_sha256', 'cases_sha256', 'corpus_sha256'], map(digest, raw)))
        run = json.loads(raw[0]); case_rows = [json.loads(s) for s in raw[1].decode().splitlines() if s.strip()]
        self.cases = {c['id']: c for c in case_rows}
        if len(self.cases) != len(case_rows):
            raise ValueError('Identificadores de casos duplicados.')
        for case in case_rows:
            rubric = case.get('evaluation_rubric', {})
            for key in ['expected_facts', 'forbidden_claims']:
                if not isinstance(rubric.get(key), list) or any(not isinstance(x, str) or not x.strip() for x in rubric[key]):
                    raise ValueError('Rubrica ausente ou inválida.')
        chunks = json.loads(raw[2]); self.corpus = {c['chunk_id']: c for c in chunks}
        if len(chunks) != len(self.corpus):
            raise ValueError('Identificadores de fontes duplicados.')
        manifest = run['manifest']
        for key, value in [('corpus_sha256', self.fingerprints['corpus_sha256'])]:
            if manifest.get(key) != value:
                raise ValueError('O corpus não corresponde ao manifesto da execução.')
        known_cases_hash = manifest.get('source_sha256', {}).get('data/cases_reviewed.jsonl')
        if known_cases_hash and known_cases_hash != self.fingerprints['cases_sha256']:
            raise ValueError('Os casos foram alterados depois da execução.')
        if manifest.get('status') not in {'completed', 'completed_with_failures'}:
            raise ValueError('Selecione uma execução finalizada.')
        self.items = {}; self.excluded = []; seen = set()
        for row in run['records']:
            key = (row['case_id'], row['method'])
            if key in seen:
                raise ValueError('Há pares caso/método duplicados na execução.')
            seen.add(key)
            if key[0] not in self.cases or key[1] not in manifest['methods']:
                raise ValueError('Caso ou método sem correspondência nas entradas.')
            case = self.cases[key[0]]
            if row['rubric'] != case['evaluation_rubric']:
                raise ValueError('A rubrica não corresponde à registrada na execução.')
            support = row.get('result', {}).get('support') if row['status'] == 'completed' else None
            kind = 'delivered'
            if row['status'] == 'blocked':
                attempts = row.get('review_audit', {}).get('attempts', [])
                support = attempts[-1].get('draft') if attempts else None
                kind = 'blocked_final_draft'
            if not isinstance(support, dict) or not any(support.get(f) for f in FIELDS):
                self.excluded.append({'case_id': key[0], 'method': key[1], 'execution_status': row['status'], 'reason': 'no_reviewable_response'})
                continue
            fields = [{'key': k, 'label': label, 'text': support[k] if isinstance(support.get(k), str) else json.dumps(support.get(k, []), ensure_ascii=False)} for k, label in FIELDS.items()]
            if any(not isinstance(f['text'], str) for f in fields):
                raise ValueError('Resposta não textual.')
            for c in row.get('result', {}).get('retrieved_chunks', []):
                original = self.corpus.get(c['chunk_id'])
                if original is None or c['content'] != original['content']:
                    raise ValueError('Trecho recuperado diferente do corpus informado.')
            self.items[key] = {'row': row, 'case': case, 'fields': fields, 'kind': kind,
                               'response_sha256': digest(canonical(fields))}
        if not self.items:
            raise ValueError('Nenhuma resposta disponível para avaliação.')
        expected = {(cid, method) for cid in manifest['case_ids'] for method in manifest['methods']}
        if seen != expected:
            raise ValueError('Execução incompleta: faltam pares caso/método.')
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('schema_version') != self.SCHEMA_VERSION or self.state.get('inputs') != self.fingerprints:
                raise ValueError('Este arquivo de avaliações pertence a outras entradas. Use outro --store.')
        else:
            self.state = {'schema_version': self.SCHEMA_VERSION, 'inputs': self.fingerprints, 'created_at': now(),
                          'seed': secrets.token_hex(16), 'reviewer': '', 'revision': 0, 'reviews': {}, 'history': []}
        keys = sorted(self.items); random.Random(self.state['seed']).shuffle(keys)
        self.queue = {f'R-{i:04d}': key for i, key in enumerate(keys, 1)}
        if not set(self.state['reviews']).issubset(self.queue):
            raise ValueError('Arquivo de avaliações contém identificadores desconhecidos.')
        if not self.path.exists():
            self._persist(self.state)

    def _persist(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.path.name+'.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def listing(self):
        rows = []
        for item_id, key in self.queue.items():
            item = self.items[key]; review = self.state['reviews'].get(item_id, {})
            question = '\n'.join(m['content'] for m in item['row']['input']['chat_transcript'])
            rows.append({'id': item_id, 'question': question, 'category': item['case'].get('category', ''),
                         'status': review.get('status', 'pending')})
        counts = {status: sum(r['status'] == status for r in rows) for status in ['pending', 'draft', 'completed']}
        return {'items': rows, 'counts': counts, 'total': len(rows), 'excluded_count': len(self.excluded),
                'reviewer': self.state['reviewer'], 'revision': self.state['revision']}

    def source(self, chunk_id):
        c = self.corpus[chunk_id]
        return {'id': chunk_id, 'title': c['title'], 'content': c['content'], 'path': c['source_path']}

    def detail(self, item_id):
        item = self.items[self.queue[item_id]]; row = item['row']; case = item['case']
        review = self.state['reviews'].get(item_id, {})
        reference_ids = [i for i in case.get('grounding_chunk_ids', []) if i in self.corpus]
        retrieved_ids = [c['chunk_id'] for c in row.get('result', {}).get('retrieved_chunks', [])]
        return {'id': item_id, 'turns': row['input']['chat_transcript'], 'response': item['fields'],
                'expected_facts': case['evaluation_rubric']['expected_facts'], 'forbidden_claims': case['evaluation_rubric']['forbidden_claims'],
                'case_evidence': {k: row['input'].get(k) for k in ['bill_facts','offer_projection','catalog_terms'] if row['input'].get(k) is not None},
                'reference_sources': [self.source(i) for i in reference_ids],
                'retrieved_sources': [self.source(i) for i in retrieved_ids],
                'review': deepcopy(review.get('judgment', blank_review(case))),
                'status': review.get('status', 'pending'), 'revision': review.get('revision', 0),
                'reviewer': self.state['reviewer'], 'saved_at': review.get('saved_at')}

    def validate(self, item_id, judgment, complete):
        item = self.items[self.queue[item_id]]
        if not isinstance(judgment, dict) or set(judgment) != {'coverage','forbidden','overall','issues','notes'}:
            raise ValueError('Formulário incompleto ou com campos desconhecidos.')
        response = normalized('\n'.join(f['text'] for f in item['fields']))
        def text(value, label, required=False):
            if not isinstance(value, str) or len(value)>20000 or (required and not value.strip()):
                raise ValueError(f'{label}: preencha um texto válido.')
        def quote(value, label, required):
            text(value, label, required)
            if value.strip() and normalized(value) not in response:
                raise ValueError(f'{label}: copie um trecho literal da resposta.')
        for group, rubric_key, choices in [('coverage','expected_facts',COVERAGE),('forbidden','forbidden_claims',FORBIDDEN)]:
            values=judgment[group]; expected=item['case']['evaluation_rubric'][rubric_key]
            if not isinstance(values,list) or len(values)!=len(expected):
                raise ValueError('Responda a todos os itens da rubrica.')
            for index,value in enumerate(values,1):
                label=f'{"Fato" if group=="coverage" else "Afirmação proibida"} {index}'
                if not isinstance(value,dict) or set(value)!={'verdict','quote','reason'}:
                    raise ValueError(f'{label}: estrutura inválida.')
                verdict=value['verdict']
                if verdict not in choices and not (not complete and verdict==''):
                    raise ValueError(f'{label}: selecione uma classificação.')
                requires_quote=verdict in {'full','partial','present'}
                quote(value['quote'],label,complete and requires_quote)
                text(value['reason'],label,complete and verdict in {'partial','not_applicable','uncertain','present'})
        if judgment['overall'] not in OVERALL and not (not complete and judgment['overall']==''):
            raise ValueError('Selecione o parecer de conteúdo.')
        text(judgment['notes'],'Observações',complete and judgment['overall']=='uncertain')
        if not isinstance(judgment['issues'],list) or len(judgment['issues'])>100:
            raise ValueError('Lista de problemas inválida.')
        for issue in judgment['issues']:
            if not isinstance(issue,dict) or set(issue)!={'type','quote','reason','source_ids'} or issue['type'] not in ISSUES:
                raise ValueError('Classifique o tipo de cada problema.')
            quote(issue['quote'],'Trecho do problema',complete)
            text(issue['reason'],'Justificativa do problema',complete)
            if not isinstance(issue['source_ids'],list) or any(i not in self.corpus for i in issue['source_ids']):
                raise ValueError('Fonte desconhecida no problema.')
        present=any(v['verdict']=='present' for v in judgment['forbidden'])
        if complete:
            if judgment['overall']=='no_issues' and (present or judgment['issues'] or any(v['verdict']=='uncertain' for v in judgment['forbidden'])):
                raise ValueError('O parecer sem problemas contradiz as marcações de conteúdo.')
            if judgment['overall']=='issues' and not (present or judgment['issues']):
                raise ValueError('Descreva um problema ou uma afirmação proibida presente.')
        return deepcopy(judgment)

    def save(self,item_id,payload):
        if item_id not in self.queue: raise KeyError(item_id)
        if not isinstance(payload,dict) or set(payload)!={'reviewer','revision','status','judgment'}:
            raise ValueError('Requisição inválida.')
        reviewer=payload['reviewer']
        if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>120:
            raise ValueError('Informe o nome do avaliador.')
        reviewer=reviewer.strip()
        if self.state['reviewer'] and reviewer!=self.state['reviewer']:
            raise Conflict('Este arquivo pertence a outro avaliador. Inicie com outro --store.')
        old=self.state['reviews'].get(item_id,{})
        if type(payload['revision']) is not int or payload['revision']!=old.get('revision',0):
            raise Conflict('Esta resposta foi alterada em outra aba. Recarregue antes de salvar.')
        if payload['status'] not in {'draft','completed'}:
            raise ValueError('Estado de revisão inválido.')
        judgment=self.validate(item_id,payload['judgment'],payload['status']=='completed')
        state=deepcopy(self.state); state['reviewer']=reviewer; state['revision']+=1
        record={'status':payload['status'],'revision':old.get('revision',0)+1,'saved_at':now(),
                'reviewed_by':reviewer,'judgment':judgment}
        state['reviews'][item_id]=record
        state['history'].append({'item_id':item_id,**deepcopy(record)})
        # Refuse concurrent processes using the same store instead of silently losing work.
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            disk=json.loads(self.path.read_text())
            if disk['revision']!=self.state['revision'] or disk['seed']!=self.state['seed']:
                raise Conflict('O arquivo foi alterado por outra instância. Reinicie a aplicação.')
            self._persist(state)
        self.state=state
        return self.detail(item_id)

    def export(self):
        records=[]
        for item_id,key in self.queue.items():
            item=self.items[key]
            record={'item_id':item_id,'case_id':key[0],'method':key[1], 'response_kind':item['kind'],
                    'execution_status':item['row']['status'],'response_sha256':item['response_sha256'],
                    'review':deepcopy(self.state['reviews'].get(item_id))}
            if record['review'] and record['review']['status']=='completed':
                cov=record['review']['judgment']['coverage']
                counts={v:sum(x['verdict']==v for x in cov) for v in sorted(COVERAGE)}
                eligible=counts['full']+counts['partial']+counts['missing']
                record['coverage_summary']={'counts':counts,'eligible':eligible,
                    'score':(counts['full']+.5*counts['partial'])/eligible if eligible else None}
            records.append(record)
        return {'schema_version':1,'exported_at':now(),'inputs':self.fingerprints,'reviewer':self.state['reviewer'],
                'progress':self.listing()['counts'],'protocol':{'partial_weight':.5,'coverage_excludes':['not_applicable','uncertain'],
                'blocked_drafts_are_delivered':False,'method_identity_hidden_during_review':True},
                'records':records,'excluded':self.excluded,'history':deepcopy(self.state['history'])}
