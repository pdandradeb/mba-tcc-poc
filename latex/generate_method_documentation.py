"""Extract active prompts and an illustrative case from hash-verified local sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def expression(node, names):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name): return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return expression(node.left,names)+expression(node.right,names)
    if isinstance(node, ast.Subscript) and isinstance(node.slice,ast.Slice):
        s=node.slice
        return expression(node.value,names)[slice(*(expression(x,names) if x else None for x in [s.lower,s.upper,s.step]))]
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='index' and not node.keywords:
        return expression(node.func.value,names).index(*(expression(x,names) for x in node.args))
    raise ValueError('Unsupported source expression: '+ast.dump(node))


def assignment(tree, name):
    return next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def latex_escape(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in text)


def decode_response(raw):
    value=raw.strip()
    if value.startswith('```'):
        value=value.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poc',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();poc=args.poc.resolve();run_path=poc/args.results
    run=json.loads(run_path.read_text());manifest=run['manifest']
    paths=['src/nlu_extractor.py','src/closing_llm.py','src/grounding.py','src/coaching_units.py',
           'src/coaching_fact_audit.py','src/heuristic_engine.py','src/pipeline.py','src/methods.py',
           'src/review_recording.py','src/store.py','src/models.py']
    sources={};trees={};hashes={}
    for path in paths:
        raw=(poc/path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==manifest['source_sha256'][path],path
        sources[path]=raw.decode();trees[path]=ast.parse(sources[path]);hashes[path]=digest
    def value(path,name,names=None):return expression(assignment(trees[path],name),names or {})
    closing=value('src/closing_llm.py','CLOSING_PROMPT')
    guidance=value('src/grounding.py','REVIEW_GUIDANCE')
    base=value('src/grounding.py','GROUNDING_PROMPT')
    units=value('src/coaching_units.py','GROUNDING_PROMPT',{'BASE_GROUNDING_PROMPT':base})
    rewrite_guidance=value('src/coaching_units.py','REWRITE_GUIDANCE')
    rewrite=value('src/closing_llm.py','surgical_prompt',{'CLOSING_PROMPT':closing,'REWRITE_GUIDANCE':rewrite_guidance})
    prompts={'interpretacao':value('src/nlu_extractor.py','_NLU_PROMPT'), 'geracao':closing,
             'revisao_unidades':units+guidance,'auditoria_factual':value('src/coaching_units.py','FACT_AUDIT_PROMPT'),
             'reformulacao':rewrite}
    out=ROOT/'latex/generated/prompts';out.mkdir(parents=True,exist_ok=True)
    for name,text in prompts.items():(out/(name+'.txt')).write_text(text)
    assert rewrite.startswith(closing)
    (out/'reformulacao_complemento.txt').write_text(rewrite[len(closing):])
    heuristic=trees['src/heuristic_engine.py'];text=sources['src/heuristic_engine.py']
    for class_name,filename in [('HeuristicNLUExtractor','heuristica_interpretacao.py.txt'),('HeuristicClosingGenerator','heuristica_geracao.py.txt')]:
        cls=next(n for n in heuristic.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        # Preserve operational definitions, excluding constructors and metadata-only declarations.
        selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'extract_from_consumer','generate_support'}
                  or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and (t.id.endswith('_KEYWORDS') or t.id=='TEMPLATES') for t in n.targets)]
        (out/filename).write_text('\n\n'.join(ast.get_source_segment(text,n) for n in selected)+'\n')
    human_path=ROOT/'output/revisao/avaliacao-humana-simplificada-2026-09-27.json'
    human=json.loads(human_path.read_text())
    assert human['inputs']['results_sha256']==hashlib.sha256(run_path.read_bytes()).hexdigest()
    by_case={r['case_id']:r for r in human['records'] if r['method']=='neuro_symbolic'}
    candidates=[r for r in run['records'] if r['method']=='neuro_symbolic' and r['status']=='blocked'
                and by_case[r['case_id']]['review']['judgment']['verdict']=='incorrect']
    case=min(candidates,key=lambda r:(len(r['review_audit']['attempts'][-1]['draft']['suggested_talk_track']),r['case_id']))
    assert case['case_id']=='CASE-015'
    attempts=case['review_audit']['attempts'];last=attempts[-1];assert len(attempts)==2
    audit=decode_response(last['fact_audit']['response_text'])
    lookup={u['id']:u['text'] for u in last['draft_units']}
    unanchored=[dict(unit_id=u['unit_id'],unit_text=lookup[u['unit_id']],claim_text=c['text'])
                for u in audit['units'] for c in u['claims'] if c['text'] not in lookup[u['unit_id']]]
    assert unanchored
    example=dict(selection='Shortest final consumer draft among the 34 blocks judged incorrect; illustrative, not representative',
                 case=case,human_review=by_case[case['case_id']],unanchored_claims=unanchored)
    evidence=ROOT/'output/revisao/documentacao-metodo-2026-09-29';evidence.mkdir(exist_ok=True)
    (evidence/'caso-ilustrativo.json').write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n')
    record=dict(results_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(), source_sha256=hashes,
                prompts={k:dict(sha256=hashlib.sha256(v.encode()).hexdigest(),characters=len(v)) for k,v in prompts.items()},
                extraction='AST literals and string concatenations; no module imported or inference executed',
                case_id=case['case_id'],human_export_sha256=hashlib.sha256(human_path.read_bytes()).hexdigest())
    (evidence/'manifesto.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    q=case['input']['chat_transcript'][0]['content'];draft=last['draft']['suggested_talk_track']
    lines=[r'% Generated by generate_method_documentation.py.',r'\subsection{Caso ilustrativo: explicação sobre TUSD}',r'\label{sec:caso_ilustrativo}',
           r'O caso \texttt{CASE-015} foi selecionado por ter a menor minuta final ao consumidor entre os 34 bloqueios julgados inadequados. O critério favorece uma apresentação compacta; o exemplo não representa a frequência dos motivos no conjunto.',
           r'\paragraph*{Pergunta.} '+latex_escape(q),
           r'\paragraph*{Recuperação.} O Método 3 recuperou os seguintes trechos, na ordem registrada:',r'\begin{enumerate}']
    for c in case['result']['retrieved_chunks']:
        lines.append(r'\item '+latex_escape(c['title'])+r', identificador \path{'+c['chunk_id']+r'}, similaridade '+f"{c['similarity_score']:.3f}".replace('.',',')+'.')
    lines.extend([r'\end{enumerate}',r'O primeiro trecho contém a evidência:'])
    quote='Custo do transporte e manutenção da rede física.'
    assert quote in case['result']['retrieved_chunks'][0]['content']
    lines.extend([r'\begin{quote}'+latex_escape(quote)+r'\end{quote}',
                  r'\paragraph*{Tentativa inicial.} A classificação por unidades foi aprovada. A auditoria factual foi rejeitada pelo validador com o código \texttt{citation\_not\_found}. O fluxo efetuou uma reformulação.',
                  r'\paragraph*{Minuta final ao consumidor.}',r'\begin{quote}'+latex_escape(draft)+r'\end{quote}',
                  r'\paragraph*{Segunda revisão.} A classificação por unidades foi aprovada novamente. A auditoria factual foi rejeitada com \texttt{claim\_not\_in\_draft}, porque uma alegação devolvida pelo auditor não era um trecho literal da unidade correspondente. O contraste preservado no registro é:',
                  r'\begin{description}',r'\item[Unidade fornecida:] '+latex_escape(unanchored[0]['unit_text'].strip()),
                  r'\item[Alegação extraída:] '+latex_escape(unanchored[0]['claim_text']),r'\end{description}',
                  r'O validador exige que a alegação extraída seja uma substring exata da unidade. Essa verificação é distinta da conferência de citações nas fontes, que admite normalização. A rejeição estrutural do parecer não equivale a demonstrar erro factual na minuta.',
                  r'\paragraph*{Decisão e avaliação humana.} O sistema bloqueou a entrega e registrou recomendação de atendimento humano. A latência total foi de '+f"{case['elapsed_s']:.2f}".replace('.',',')+r' s. O avaliador marcou o bloqueio como incorreto, indicando que a minuta era globalmente adequada e poderia ser entregue. O exemplo distingue a validação do parecer automático da avaliação global do texto; não permite estender o mesmo motivo aos outros bloqueios.',
                  r'Os três trechos completos, as duas orientações estruturadas, os pareceres e o julgamento humano estão preservados no arquivo \path{caso-ilustrativo.json}, em \path{output/revisao/documentacao-metodo-2026-09-29/}.'])
    (ROOT/'latex/generated/caso_ilustrativo.tex').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps({'prompts':record['prompts'],'case':case['case_id'],'unanchored':unanchored},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
