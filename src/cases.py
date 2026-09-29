import json
from .corpus import DATA, check_public_text
from .models import ConsumerProfile, BillFacts


def load_cases(path=DATA/'cases.jsonl'):
    cases=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if row.get('synthetic') is not True: raise ValueError('Only explicitly synthetic public cases are accepted')
        check_public_text(json.dumps(row,ensure_ascii=False))
        if not row.get('turns') or row['turns'][-1].get('role') != 'user':
            raise ValueError('The input must end with a user request, never a reference answer')
        if any(t.get('role') not in {'user', 'assistant'} or not isinstance(t.get('content'), str) or not t['content'].strip() for t in row['turns']):
            raise ValueError('Invalid conversation turns')
        c=ConsumerProfile(consumer_id=row['id'],name=row['id'],segment='unknown',region='unknown',
            consumption_profile='unknown',interests=[],maturity='unknown',urgency='unknown',
            restrictions=[],objections=[],current_stage='descoberta',distributor=None,monthly_consumption_kwh=None,
            icp_category='unknown',tenant_status='unknown',
            chat_transcript=[{'sender':'customer' if t['role']=='user' else 'assistant','content':t['content']} for t in row['turns']],notes='')
        if row.get('bill_facts'):
            facts=row['bill_facts']
            if facts['source_kind']!='synthetic_bill_fixture': raise ValueError('Real billing evidence is not public test data')
            c.bill_facts=BillFacts(**facts)
        cases.append((c,row.get('evaluation_rubric',{})))
    if not cases or len({c.consumer_id for c,_ in cases})!=len(cases): raise ValueError('Empty or duplicate cases')
    return cases
