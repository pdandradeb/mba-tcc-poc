#!/usr/bin/env python3
"""Recompute thesis tables from a saved run. Uses no model, database or network."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from analysis_metrics import boxplot_summary, distribution_summary, retrieval_at_k, wilson95
from experiment_figures import latency_figure
from review_reason_labels import REASON_LABELS

METHODS = ['heuristic', 'fts', 'neuro_symbolic', 'vector']
LABELS = {m: f'Método {i}' for i, m in enumerate(METHODS, 1)}
# Editorial descriptions; document membership and counts come from the verified corpus.
CORPUS_TOPICS = {
    'okf/market/acl.md': 'Mercado Livre de Energia, Grupo A, varejista e CCEE',
    'okf/market/distribuidoras-e-tarifas.md': 'Distribuidoras, TUSD, TE e reajustes tarifários',
    'okf/market/energia-solar-por-assinatura.md': 'Geração distribuída remota, regras de adesão e inquilinos',
    'okf/market/glossario.md': 'Nomenclatura regulatória e termos do setor elétrico',
    'okf/market/marcos-regulatorios.md': 'Legislação, Lei 14.300/2022 e normas setoriais',
    'okf/market/regulamentacao-e-atores.md': 'Papéis institucionais de ANEEL, ONS, CCEE e distribuidoras',
    'okf/sales/contratos-e-pos-venda.md': 'Prazos de ativação, regras de fidelidade, rescisão e queixas',
    'okf/sales/leitura-de-fatura.md': 'Diagnóstico técnico de faturas, encargos e contas inválidas',
    'okf/sales/objecoes.md': 'Argumentação consultiva para objeções recorrentes',
    'okf/sales/perguntas-frequentes-de-clientes.md': 'Dúvidas de clientes sobre duas contas, garantias e cancelamento',
}

def corpus_summary(poc, manifest):
    raw = (poc/'data/public_corpus.json').read_bytes()
    publication = json.loads((poc/'data/publication_manifest.json').read_text())
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == manifest['corpus_sha256'] == publication['corpus_sha256']
    chunks = json.loads(raw)
    counts = Counter(chunk['source_path'] for chunk in chunks)
    documents = publication['documents']
    assert len(documents) == len(set(documents))
    assert set(counts) == set(documents) == set(CORPUS_TOPICS)
    assert len(chunks) == publication['chunks']
    assert len({chunk['chunk_id'] for chunk in chunks}) == len(chunks)
    def sizes(selected):
        return dict(characters=distribution_summary([len(c['content']) for c in selected]),
                    whitespace_units=distribution_summary([len(c['content'].split()) for c in selected]))
    return dict(corpus_sha256=digest, documents=len(documents), chunks=len(chunks),
                chunks_by_document={name: counts[name] for name in documents},
                size_definition=dict(field='content', characters='Unicode code points, including whitespace and Markdown',
                                     whitespace_units='Nonempty units produced by Python str.split(); not model tokens'),
                sizes=sizes(chunks), sizes_by_document={name:sizes([c for c in chunks if c['source_path']==name]) for name in documents})

def matrix(rows, cases, *, initial=False):
    result = dict(VP=0, VN=0, FP=0, FN=0)
    for row in rows:
        if initial:
            c = row['result']['interpretation']
            predicted = c['explicit_human_requested'] or c['handoff_recommended']
        else:
            predicted = row['result']['handoff']['action'] == 'escalate_human'
        expected = cases[row['case_id']]['handoff_expected']
        result['VP' if expected and predicted else 'FN' if expected else 'FP' if predicted else 'VN'] += 1
    return result

def metrics(m):
    tp, tn, fp, fn = (m[k] for k in ['VP', 'VN', 'FP', 'FN'])
    n = tp + tn + fp + fn
    p = (tp + tn) / n
    return dict(accuracy=p, precision=tp/(tp+fp), recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn),
                accuracy_wilson95=wilson95(tp+tn, n))

def retrieval_summary(grouped, cases, corpus_ids):
    references = {cid: case['grounding_chunk_ids'] for cid, case in cases.items()}
    assert all(set(ids) <= corpus_ids and len(ids) == len(set(ids)) for ids in references.values())
    eligible = sorted(cid for cid, ids in references.items() if ids)
    excluded = sorted(set(cases)-set(eligible))
    summary = dict(k=3, eligible_cases=len(eligible), excluded_no_reference=excluded,
                   reference_size_counts=dict(Counter(len(references[cid]) for cid in eligible)), methods={})
    for method in METHODS[1:]:
        details = []
        for row in sorted(grouped[method], key=lambda r: r['case_id']):
            cid = row['case_id']
            retrieved = [c['chunk_id'] for c in row['result']['retrieved_chunks']]
            assert len(retrieved) <= 3 and set(retrieved) <= corpus_ids
            score = retrieval_at_k(retrieved, references[cid])
            if score is not None:
                details.append(dict(case_id=cid, status=row['status'], reference_ids=references[cid],
                                    retrieved_ids=retrieved, **score))
        assert len(details) == len(eligible) > 0
        summary['methods'][method] = dict(
            n=len(details), recall_at_3=statistics.mean(d['recall'] for d in details),
            mrr_at_3=statistics.mean(d['reciprocal_rank'] for d in details),
            cases_with_match=sum(d['first_reference_rank'] is not None for d in details),
            per_case=details)
    return summary

def latency(rows):
    x = sorted(r['elapsed_s'] for r in rows)
    return dict(mean=statistics.mean(x), median=statistics.median(x), p95=x[math.ceil(.95*len(x))-1])

def signed_rank_exact(differences):
    # Exact two-sided Wilcoxon signed-rank distribution for nonzero untied differences.
    assert all(x != 0 for x in differences)
    assert len(set(map(abs, differences))) == len(differences)
    ranked = sorted(differences, key=abs)
    wplus = sum(i for i, x in enumerate(ranked, 1) if x > 0)
    total = len(ranked)*(len(ranked)+1)//2
    w = min(wplus, total-wplus)
    counts = [0]*(total+1)
    counts[0] = 1
    reached = 0
    for rank in range(1, len(ranked)+1):
        for value in range(reached, -1, -1):
            counts[value+rank] += counts[value]
        reached += rank
    return dict(n=len(ranked), w=w, p_two_sided=min(1, 2*sum(counts[:w+1])/2**len(ranked)),
                median_difference_s=statistics.median(differences), positive_differences=sum(x>0 for x in differences))

def decimal(x, digits=2):
    return f'{x:.{digits}f}'.replace('.', ',')

def table(caption, label, columns, rows, spec, note='Cálculos próprios a partir dos registros da execução.'):
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+spec+'}',r'\hline', ' & '.join(columns)+r' \\',r'\hline']
    lines.extend(' & '.join(map(str,r))+r' \\' for r in rows)
    lines += [r'\hline',r'\end{tabular}',r'\fonte{'+note+'}',r'\end{table}', '']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poc', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='Path relative to --poc')
    args=parser.parse_args()
    poc=args.poc.resolve(); path=poc/args.results
    data=json.loads(path.read_text()); manifest=data['manifest']; rows=data['records']
    cases={r['id']:r for r in map(json.loads,(poc/'data/cases_reviewed.jsonl').read_text().splitlines())}
    assert len(cases)==100 and sum(c['handoff_expected'] for c in cases.values())==17
    assert len(rows)==400 and {(r['case_id'],r['method']) for r in rows}=={(i,m) for i in cases for m in METHODS}
    assert all(r['status'] in {'completed','blocked'} for r in rows)
    for name, digest in manifest['source_sha256'].items():
        source=poc/name
        # The runner hashed two unused legacy case sets too. The public
        # package contains the 100 reviewed cases passed to this run.
        if name in {'data/cases.jsonl', 'data/cases_legacy_134.jsonl'} and not source.exists():
            continue
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest, name
    corpus = corpus_summary(poc, manifest)
    grouped={m:[r for r in rows if r['method']==m] for m in METHODS}
    retrieval = retrieval_summary(grouped, cases, {c['chunk_id'] for c in json.loads((poc/'data/public_corpus.json').read_text())})
    paired={r['case_id'] for r in grouped['neuro_symbolic'] if r['status']=='completed'}
    common={m:[r for r in grouped[m] if r['case_id'] in paired] for m in ['neuro_symbolic','vector']}
    reasons=Counter(); categories=Counter()
    for r in grouped['neuro_symbolic']:
        if r['status']=='blocked':
            attempt=r['review_audit']['attempts'][-1]
            reason=(attempt['fact_audit'] or attempt['grounding'])['reason']
            reasons[reason['code']]+=1; categories[reason['category']]+=1
    indexed={(r['case_id'],r['method']):r for r in rows}
    diffs=[indexed[(i,'neuro_symbolic')]['elapsed_s']-indexed[(i,'vector')]['elapsed_s'] for i in sorted(cases)]
    stats=dict(source_results=str(args.results),results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               configuration={k:manifest[k] for k in ['started_at','finished_at','model','embedding_model','embedding_dimension','top_k','timeouts_s','workers','python','seed']},
               n_cases=len(cases),n_records=len(rows),corpus=corpus,retrieval=retrieval,methods={},paired={},rejection_codes=dict(reasons),rejection_categories=dict(categories),latency_wilcoxon=signed_rank_exact(diffs))
    for m, rs in grouped.items():
        mat=matrix(rs,cases)
        stats['methods'][m]=dict(matrix=mat,metrics=metrics(mat),initial_matrix=matrix(rs,cases,initial=True),latency=latency(rs),states=dict(Counter(r['status'] for r in rs)))
        summary=data['summary'][m]
        assert math.isclose(summary['mean_latency_s'],stats['methods'][m]['latency']['mean'])
    for m,rs in common.items():
        mat=matrix(rs,cases);stats['paired'][m]=dict(n=len(rs),matrix=mat,metrics=metrics(mat),latency=latency(rs))
    dest=Path(__file__).resolve().parent/'generated';dest.mkdir(exist_ok=True)
    def save(name, text): (dest/name).write_text('% Gerado por generate_experiment_tables.py; não editar manualmente.\n'+text)
    stats['latency_figure']={method:boxplot_summary([r['elapsed_s'] for r in grouped[method]]) for method in METHODS}
    save('fig_latencias.tex',latency_figure(stats['latency_figure']))
    save('tamanhos_corpus.tex', table(
        'Distribuição do tamanho dos 86 trechos do corpus', 'tab:tamanhos_corpus',
        ['Unidade', 'Mínimo', 'Mediana', 'Média', 'P95', 'Máximo'],
        [[label, v['minimum'], decimal(v['median'],1), decimal(v['mean'],1), v['p95'], v['maximum']]
         for label, v in [('Caracteres',corpus['sizes']['characters']),('Unidades por espaços',corpus['sizes']['whitespace_units'])]],
        'lrrrrr', note=r'Contagem no campo \texttt{content}, sem acrescentar título ou metadados. Caracteres incluem espaços e marcação Markdown; unidades por espaços resultam da separação por espaços em branco, sem tokenizador do modelo.'))
    save('recuperacao.tex', table(
        'Correspondência da recuperação com os trechos anotados nos casos', 'tab:recuperacao',
        ['Método', 'Casos', 'Recall@3', 'MRR@3', 'Com correspondência'],
        [[LABELS[m], v['n'], decimal(v['recall_at_3'], 3), decimal(v['mrr_at_3'], 3), v['cases_with_match']]
         for m, v in retrieval['methods'].items()], 'lrrrr',
        note='Cálculos próprios nos 91 casos com referência. Médias por pergunta; nove casos sem trechos anotados são excluídos. Incluem-se casos bloqueados.'))
    save('documentos_corpus.tex', table(
        'Documentos autorizados que compõem o corpus público da base de conhecimento',
        'tab:documentos_corpus', ['Documento canônico', 'Domínio e conteúdo principal', 'Trechos'],
        [[r'\path{'+name+'}', CORPUS_TOPICS[name], count]
         for name, count in corpus['chunks_by_document'].items()]
        + [['Total consolidado', 'Recorte público de mercado e atendimento', corpus['chunks']]],
        'p{6.8cm}p{6.5cm}r',
        note=r'Contagem em \texttt{public\_corpus.json}, conferida com o manifesto da execução e a lista de documentos de \texttt{publication\_manifest.json}.'))
    save('estados.tex',table('Estados observados nos 100 casos de cada método','tab:estados_atuais',
        ['Método','Concluídos','Bloqueados','Falhos','Encaminhamentos'],
        [[LABELS[m],stats['methods'][m]['states'].get('completed',0),stats['methods'][m]['states'].get('blocked',0),0,stats['methods'][m]['matrix']['VP']+stats['methods'][m]['matrix']['FP']] for m in METHODS],'lrrrr'))
    save('latencias.tex',table('Latências por caso, com unidade indicada por método','tab:latencias_atuais',
        ['Método','Unidade','Média','Mediana','Percentil 95'],
        [[LABELS[m], 'ms' if m=='heuristic' else 's',*[decimal(stats['methods'][m]['latency'][k]*(1000 if m=='heuristic' else 1),3 if m=='heuristic' else 2) for k in ['mean','median','p95']]] for m in METHODS],'llrrr'))
    reason_labels=REASON_LABELS
    assert set(reasons)==set(reason_labels)
    save('bloqueios.tex',table('Causas registradas para a rejeição final no Método 3','tab:causas_bloqueios',
        ['Causa','Casos'],[[v,reasons[k]] for k,v in reason_labels.items()]+[['Total',sum(reasons.values())]],'lr'))
    save('handoff_matriz.tex',table('Decisão final de encaminhamento perante o gabarito humano','tab:handoff_matriz',
        ['Método','VP','VN','FP','FN'],[[LABELS[m],*[stats['methods'][m]['matrix'][k] for k in ['VP','VN','FP','FN']]] for m in METHODS],'lrrrr'))
    save('handoff_metricas.tex',table('Métricas da decisão final de encaminhamento','tab:handoff_metricas',
        ['Método','Acurácia',r'IC 95\% da acurácia','Precisão','Revocação','$F_1$'],
        [[LABELS[m],decimal(stats['methods'][m]['metrics']['accuracy'],3), '$['+'; '.join(decimal(v,3) for v in stats['methods'][m]['metrics']['accuracy_wilson95'])+']$',*[decimal(stats['methods'][m]['metrics'][k],3) for k in ['precision','recall','f1']]] for m in METHODS],'lrrrrr',note='Cálculos próprios da matriz de confusão e dos intervalos de Wilson.'))
    save('handoff_inicial.tex',table('Encaminhamento antes da aplicação da política de bloqueio','tab:handoff_inicial',
        ['Método','VP','VN','FP','FN'],[[LABELS[m],*[stats['methods'][m]['initial_matrix'][k] for k in ['VP','VN','FP','FN']]] for m in METHODS],'lrrrr'))
    save('pareada.tex',table('Encaminhamento e latência nas 60 perguntas com entrega comum','tab:comparacao_pareada',
        ['Método','VP','VN','FP','FN','Precisão','Média (s)'],
        [[LABELS[m],*[stats['paired'][m]['matrix'][k] for k in ['VP','VN','FP','FN']],decimal(stats['paired'][m]['metrics']['precision'],3),decimal(stats['paired'][m]['latency']['mean'])] for m in common],'lrrrrrr'))
    evidence=Path(__file__).resolve().parent.parent/'output/revisao/metricas-experimento-atual-2026-09-27.json'
    evidence.write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),paired=len(paired),workers=manifest['workers'],wilcoxon=stats['latency_wilcoxon']),indent=2))

if __name__=='__main__':main()
