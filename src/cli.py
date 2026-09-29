"""Index the public corpus and run the four methods against PostgreSQL."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
from time import perf_counter
from uuid import uuid4
from .cases import load_cases
from .config import Settings, EMBEDDING_DIMENSION
from .corpus import DATA, sha256, load_corpus
from .methods import METHODS, build_pipeline
from .pipeline import Pipeline
from .providers import NeuralEmbedding
from .provenance import current_method_hashes
from .store import PostgresKnowledgeStore

ROOT=Path(__file__).resolve().parents[1]


def summarize(records):
    summary={}
    for method in METHODS:
        rows=[r for r in records if r['method']==method]
        if not rows: continue
        times=[r['elapsed_s'] for r in rows]
        returned=[r for r in rows if r['status']=='completed']
        summary[method]={'attempted':len(rows),'completed':len(returned),
            'blocked':sum(r['status']=='blocked' for r in rows),'failed':sum(r['status']=='failed' for r in rows),
            'handoff_recommended':sum(r.get('result',{}).get('handoff',{}).get('action')=='escalate_human' for r in rows),
            'retrieved_chunks_mean':statistics.mean(len(r['result']['retrieved_chunks']) for r in returned) if returned else None,
            'mean_latency_s':statistics.mean(times),'median_latency_s':statistics.median(times),
            'p95_latency_s':sorted(times)[max(0,__import__('math').ceil(.95*len(times))-1)]}
    return summary


def run(settings, methods, *, cases_path=None, limit=None, case_id=None, output=None, completion=None, embedding_provider=None, workers=1):
    if workers<1 or workers>16: raise ValueError("workers must be between 1 and 16")
    cases=load_cases(path=cases_path or (DATA/'cases.jsonl'))
    if case_id: cases=[c for c in cases if c[0].consumer_id==case_id]
    if limit is not None:
        if limit<1: raise ValueError('Limit must be positive')
        cases=cases[:limit]
    if not cases: raise ValueError('No cases selected')
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods): raise ValueError('Choose distinct methods')
    output=Path(output) if output else ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    corpus_id,_=load_corpus()
    sources={}
    for folder in ['src','sql','data']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.py','.json','.jsonl','.sql'}:
                rel=path.relative_to(ROOT);dest=output/'snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);sources[str(rel)]=sha256(path.read_bytes())
    for name in ['pyproject.toml','uv.lock','compose.yaml','Dockerfile']:
        shutil.copy2(ROOT/name,output/'snapshot'/name);sources[name]=sha256((ROOT/name).read_bytes())
    doc={'manifest':{'started_at':datetime.now(timezone.utc).isoformat(),'methods':{m:METHODS[m] for m in methods},
        'model':settings.model,'embedding_model':settings.embedding_model,'embedding_dimension':EMBEDDING_DIMENSION, 'timeouts_s':settings.timeouts,'top_k':settings.top_k,
        'execution_kind':'fixture' if completion or embedding_provider else 'live','corpus_sha256':corpus_id,
        'method_source_sha256':{m:current_method_hashes(m) for m in methods},
        'source_sha256':sources,'python':platform.python_version(),'seed':2026,'workers':workers,
        'case_ids':[c.consumer_id for c,_ in cases], 'status':'running',
        'latency_scope':'per-case processing, including NLU, query embedding, retrieval, generation and review; excludes corpus indexing'},'records':[]}
    result_path=output/'results.json'
    def save():
        p=output/'results.tmp';p.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n');p.replace(result_path)
    save()
    try:
        embed=NeuralEmbedding(settings.embedding_model,provider=embedding_provider, timeout_s=settings.embedding_timeout_s)
        store=PostgresKnowledgeStore(settings.database_url,embed)
        if any(m!='heuristic' for m in methods):
            store.initialize()
            # Indexing is a separate command, never hidden in case latency.
        def execute_case(item):
            index,(case,rubric)=item
            local_embed=NeuralEmbedding(settings.embedding_model,provider=embedding_provider, timeout_s=settings.embedding_timeout_s)
            local_store=PostgresKnowledgeStore(settings.database_url,local_embed)
            pipelines={m:build_pipeline(m,local_store if m!='heuristic' else None,settings.model,completion=completion,top_k=settings.top_k, llm_timeout_s=settings.llm_timeout_s, review_timeout_s=settings.review_timeout_s) for m in methods}
            order=list(methods);random.Random(2026+index).shuffle(order)
            rows=[]
            for method in order:
                pipeline=pipelines[method];start=perf_counter()
                row={'case_id':case.consumer_id,'method':method,'input':asdict(case),'rubric':rubric}
                try:
                    result=pipeline.process(deepcopy(case));row.update(status=result['status'],result=result)
                except Exception as exc:
                    row.update(status='failed',error_type=type(exc).__name__)
                finally:
                    row.update(elapsed_s=perf_counter()-start,trace=pipeline.last_trace)
                    if method=='neuro_symbolic':row['review_audit']=pipeline.review_audit
                rows.append(row)
                print(f"{case.consumer_id} {method}: {row['status']}",flush=True)
            return rows
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(execute_case,item) for item in enumerate(cases)]
            for future in as_completed(futures):
                doc['records'].extend(future.result());save()
        doc['manifest']['status']='completed_with_failures' if any(r['status']=='failed' for r in doc['records']) else 'completed'
    except BaseException:
        doc['manifest']['status']='aborted'
        raise
    finally:
        doc['summary']=summarize(doc['records']);doc['manifest']['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return result_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    aud=sub.add_parser('audit');aud.add_argument('--cases',type=Path,default=None,help='Path to cases JSONL file')
    idx=sub.add_parser('index');idx.add_argument('--rebuild-vectors',action='store_true',help='Rebuild only the derived vector table for the current dimension')
    r=sub.add_parser('run');r.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS));r.add_argument('--limit',type=int);r.add_argument('--case');r.add_argument('--cases',type=Path,default=None,help='Path to cases JSONL file');r.add_argument('--output',type=Path);r.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv);settings=Settings.from_env()
    if args.command=='audit':
        _,chunks=load_corpus();cases=load_cases(path=args.cases or (DATA/'cases.jsonl'));print(json.dumps({'public_documents':len({c['source_path'] for c in chunks}),'public_chunks':len(chunks),'synthetic_cases':len(cases)}));return 0
    if args.command=='index':
        store=PostgresKnowledgeStore(settings.database_url,NeuralEmbedding(settings.embedding_model, timeout_s=settings.embedding_timeout_s));store.initialize(rebuild_vectors=args.rebuild_vectors);start=perf_counter();count=store.index_embeddings();print(json.dumps({'new_vectors':count,'dimension':EMBEDDING_DIMENSION,'elapsed_s':perf_counter()-start}));return 0
    path=run(settings,args.methods,cases_path=args.cases,limit=args.limit,case_id=args.case,output=args.output,workers=args.workers);print(path)
    return 1 if json.loads(path.read_text())['manifest']['status']!='completed' else 0

if __name__=='__main__':
    raise SystemExit(main())
