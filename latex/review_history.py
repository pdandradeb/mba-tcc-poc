"""Audit saved judgments without inferring reading time or UI interactions."""
from collections import Counter, defaultdict
from datetime import datetime

from analysis_metrics import distribution_summary


def summarize_history(records, history):
    by_id = {r['item_id']: r for r in records}
    assert len(by_id) == len(records)
    indexed = sorted(history, key=lambda e: datetime.fromisoformat(e['saved_at']))
    by_item = defaultdict(list)
    times = []
    gaps = []
    for event in indexed:
        assert event['item_id'] in by_id
        time = datetime.fromisoformat(event['saved_at'])
        assert time.tzinfo is not None
        if times:
            gaps.append(dict(previous_item_id=indexed[len(times)-1]['item_id'],
                             item_id=event['item_id'], target=by_id[event['item_id']]['target'],
                             seconds=(time-times[-1]).total_seconds()))
        times.append(time)
        by_item[event['item_id']].append(event)
    assert set(by_item) == set(by_id)
    revisits = []
    for item_id, events in by_item.items():
        assert [e['revision'] for e in events] == list(range(1, len(events)+1))
        final = by_id[item_id]['review']
        assert all(events[-1][k] == final[k] for k in final)
        if len(events) > 1:
            revisits.append(dict(item_id=item_id, case_id=by_id[item_id]['case_id'],
                                 method=by_id[item_id]['method'], target=by_id[item_id]['target'],
                                 verdicts=[e['judgment']['verdict'] for e in events],
                                 revisions=len(events)))
    grouped = {}
    for target in ('response', 'block'):
        items = [r for r in records if r['target'] == target]
        events = [e for e in indexed if by_id[e['item_id']]['target'] == target]
        first = Counter(by_item[r['item_id']][0]['judgment']['verdict'] for r in items)
        final = Counter(r['review']['judgment']['verdict'] for r in items)
        changes = Counter()
        for r in items:
            seq = by_item[r['item_id']]
            for a, b in zip(seq, seq[1:]):
                changes[a['judgment']['verdict']+'__'+b['judgment']['verdict']] += 1
        grouped[target] = dict(items=len(items), saves=len(events),
            resaved_items=sum(len(by_item[r['item_id']])>1 for r in items),
            first_verdicts=dict(first), final_verdicts=dict(final),
            verdict_changes=sum(v for k,v in changes.items() if len(set(k.split('__')))>1),
            unchanged_verdict_resaves=sum(v for k,v in changes.items() if len(set(k.split('__')))==1),
            transitions=dict(changes),
            gaps_ending_at_target=distribution_summary([g['seconds'] for g in gaps if g['target']==target]))
    return dict(events=len(indexed), items=len(by_id), first_saved_at=indexed[0]['saved_at'],
                last_saved_at=indexed[-1]['saved_at'], elapsed_span_s=(times[-1]-times[0]).total_seconds(),
                groups=grouped, resaved_items=sorted(revisits, key=lambda r:r['item_id']),
                consecutive_gaps=distribution_summary([g['seconds'] for g in gaps]),
                gaps_at_most_5s=sum(g['seconds']<=5 for g in gaps),
                gap_definition='Elapsed wall time between successive saved events, grouped by the target of the later save. Not reading time.',
                gaps=gaps)
