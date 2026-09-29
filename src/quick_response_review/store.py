"""Global human judgments of delivered responses or final blocking decisions."""
from copy import deepcopy
from src.response_review.store import ResponseStore, now

REASONS = {
    'unsupported_claim': 'O revisor apontou uma afirmação sem apoio nas fontes.',
    'unsupported_or_contradicted_claim': 'A auditoria apontou uma afirmação sem apoio ou em contradição com as fontes.',
    'citation_not_found': 'Uma citação usada pelo revisor não foi encontrada na evidência fornecida.',
    'claim_not_in_draft': 'O revisor avaliou uma afirmação que não foi localizada na minuta.',
    'unit_coverage_mismatch': 'A revisão não correspondeu a todos os trechos da minuta na ordem esperada.',
}


def blocking_info(row):
    if row['status'] != 'blocked':
        return None
    attempts = row.get('review_audit', {}).get('attempts', [])
    last = attempts[-1] if attempts else {}
    reasons = []
    excerpts = []
    units = {u['id']: u['text'] for u in last.get('draft_units', [])}
    for name in ('generation', 'grounding', 'fact_audit'):
        stage = last.get(name) or {}
        reason = stage.get('reason')
        if reason:
            code = reason.get('code', '') if isinstance(reason, dict) else ''
            message = reason.get('validator_message', '') if isinstance(reason, dict) else str(reason)
            reasons.append({'stage': name, 'code': code, 'message': REASONS.get(code, message or code), 'recorded_message': message})
        report = stage.get('report') or {}
        review = report.get('review', report)
        for segment in review.get('segments', []):
            if segment.get('kind') == 'unsupported' and segment.get('unit_id') in units:
                excerpts.append(units[segment['unit_id']])
        for unit in report.get('fact_audit', {}).get('units', []):
            for claim in unit.get('claims', []):
                if claim.get('status') in {'unsupported', 'contradicted'}:
                    excerpts.append(claim.get('text', ''))
    return {'reasons': reasons or [{'stage': '', 'code': 'not_recorded', 'message': 'O registro não informa um motivo estruturado para o bloqueio.', 'recorded_message': ''}],
            'flagged_excerpts': list(dict.fromkeys(x for x in excerpts if x)), 'attempts': len(attempts)}


class QuickResponseStore(ResponseStore):
    SCHEMA_VERSION = 2

    def listing(self):
        data = super().listing()
        for row in data['items']:
            item = self.items[self.queue[row['id']]]
            row['blocked'] = item['row']['status'] == 'blocked'
            row['verdict'] = self.state['reviews'].get(row['id'], {}).get('judgment', {}).get('verdict', '')
        data['uncertain_count'] = sum(x['verdict'] == 'uncertain' for x in data['items'])
        return data

    def detail(self, item_id):
        data = super().detail(item_id)
        item = self.items[self.queue[item_id]]
        review = self.state['reviews'].get(item_id, {})
        blocked = item['row']['status'] == 'blocked'
        data.update(blocked=blocked, target='block' if blocked else 'response',
                    blocking=blocking_info(item['row']),
                    review=deepcopy(review.get('judgment', {'verdict': '', 'notes': ''})))
        return data

    def validate(self, item_id, judgment, complete):
        if not isinstance(judgment, dict) or set(judgment) != {'verdict', 'notes'}:
            raise ValueError('Avaliação simplificada inválida.')
        if judgment['verdict'] not in ['correct', 'incorrect', 'uncertain']:
            raise ValueError('Escolha correto, incorreto ou em dúvida.')
        if not isinstance(judgment['notes'], str) or len(judgment['notes']) > 20000:
            raise ValueError('Comentário inválido ou muito longo.')
        if complete != (judgment['verdict'] != 'uncertain'):
            raise ValueError('Uma avaliação em dúvida deve permanecer como rascunho.')
        return deepcopy(judgment)

    def export(self):
        records = []
        summary = {target: {verdict: 0 for verdict in ['correct', 'incorrect', 'uncertain', 'pending']} for target in ['response', 'block']}
        for item_id, key in self.queue.items():
            item = self.items[key]
            target = 'block' if item['row']['status'] == 'blocked' else 'response'
            review = deepcopy(self.state['reviews'].get(item_id))
            verdict = review['judgment']['verdict'] if review else 'pending'
            summary[target][verdict] += 1
            records.append({'item_id': item_id, 'case_id': key[0], 'method': key[1], 'target': target,
                            'response_kind': item['kind'], 'execution_status': item['row']['status'],
                            'response_sha256': item['response_sha256'], 'blocking': blocking_info(item['row']), 'review': review})
        return {'schema_version': self.SCHEMA_VERSION, 'exported_at': now(), 'inputs': self.fingerprints,
                'reviewer': self.state['reviewer'], 'progress': self.listing()['counts'], 'summary': summary,
                'protocol': {'name': 'simplified_global_judgment', 'blocking_reason_visible': True,
                             'blinded': False, 'comment_required': False, 'blocked_drafts_are_delivered': False,
                             'response_and_block_judgments_are_separate': True},
                'records': records, 'excluded': self.excluded, 'history': deepcopy(self.state['history'])}
