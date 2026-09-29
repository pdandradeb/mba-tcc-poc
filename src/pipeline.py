"""Three comparable methods with one input and output contract."""
from dataclasses import asdict, replace
from .closing_llm import GroundedClosingLLM, GenerationUnavailable, ReviewRejected
from .heuristic_engine import HeuristicNLUExtractor, HeuristicClosingGenerator
from .models import HandoffDecision
from .nlu_extractor import NLUExtractor

METHODS = {
    'heuristic': {'interpretation':'rules','retrieval':'none','generation':'templates','review':'none'},
    'fts': {'interpretation':'llm','retrieval':'postgres_fts','generation':'llm','review':'none'},
    'neuro_symbolic': {'interpretation':'llm','retrieval':'pgvector_cosine','generation':'llm','review':'units_citations_fact_audit'},
}

class Pipeline:
    def __init__(self, method, store, model, *, completion=None, top_k=3, llm_timeout_s=60, review_timeout_s=120):
        if method not in METHODS: raise ValueError('Unknown method')
        self.method, self.store, self.top_k = method, store, top_k
        self.nlu = HeuristicNLUExtractor() if method=='heuristic' else NLUExtractor(model_name=model, completion=completion, timeout_s=llm_timeout_s)
        self.generator = HeuristicClosingGenerator() if method=='heuristic' else GroundedClosingLLM(model_name=model, completion=completion, timeout_s=llm_timeout_s, review_timeout_s=review_timeout_s)
        self.last_trace=[]

    def process(self, consumer):
        components=[self.nlu,self.generator]
        if self.store:
            components.append(self.store)
            if self.store.embedding: components.append(self.store.embedding)
        for c in components: c.events.clear()
        try:
            context=self.nlu.extract_from_consumer(consumer)
            consumer=replace(consumer,objections=context.detected_objections,current_stage=context.current_stage)
            escalate=context.explicit_human_requested or context.handoff_recommended
            handoff=HandoffDecision('escalate_human' if escalate else 'keep_automated',
                'high' if escalate else 'none', context.handoff_reason or 'Decisão de apoio ao consultor, sem transferência executada.')
            query=context.raw_summary
            chunks=[] if self.method=='heuristic' else self.store.search(query,
                method='fts' if self.method=='fts' else 'dense',top_k=self.top_k)
            if self.method=='heuristic':
                support=self.generator.generate_support(consumer,context,handoff,chunks)
            else:
                try:
                    support=self.generator.generate_support(consumer,context,handoff,chunks,skip_review=self.method=='fts')
                except GenerationUnavailable as exc:
                    cause=exc
                    while cause is not None:
                        if isinstance(cause,ReviewRejected):
                            return {'status':'blocked','support':None,'handoff':asdict(replace(handoff,action='escalate_human',urgency='high',rationale='Revisão rejeitou a resposta; assistência humana recomendada.')),
                                    'interpretation':asdict(context),'retrieved_chunks':[asdict(c) for c in chunks],
                                    'review':cause.review}
                        cause=cause.__cause__
                    raise
            return {'status':'completed','support':asdict(support),'handoff':asdict(handoff),
                'interpretation':asdict(context),'retrieved_chunks':[asdict(c) for c in chunks]}
        finally:
            self.last_trace=[dict(e) for c in components for e in c.events]
