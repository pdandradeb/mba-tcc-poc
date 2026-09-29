"""Current four-method registry; unchanged baseline implementations remain shared."""
from copy import deepcopy
from dataclasses import asdict, replace

from .pipeline import Pipeline, METHODS as BASE_METHODS
from .models import HandoffDecision
from .review_recording import RecordedReviewLLM

METHODS = {**BASE_METHODS,
    'vector': {'interpretation':'llm','retrieval':'pgvector_cosine','generation':'llm','review':'none'}}


class ReviewedPipeline(Pipeline):
    def __init__(self, method, store, model, *, completion=None, top_k=3, llm_timeout_s=60, review_timeout_s=120):
        super().__init__(method,store,model,completion=completion,top_k=top_k,
                         llm_timeout_s=llm_timeout_s,review_timeout_s=review_timeout_s)
        self.generator=RecordedReviewLLM(model_name=model,completion=completion,
                                        timeout_s=llm_timeout_s,review_timeout_s=review_timeout_s)
        self.review_audit=None

    def process(self, consumer):
        self.generator.review_audit=None
        try:
            return super().process(consumer)
        finally:
            self.review_audit=deepcopy(self.generator.review_audit)


class VectorPipeline(Pipeline):
    def __init__(self, method, store, model, **kwargs):
        super().__init__('fts',store,model,**kwargs)
        self.method=method

    def process(self, consumer):
        components=[self.nlu,self.generator,self.store]
        if self.store.embedding:components.append(self.store.embedding)
        for component in components:component.events.clear()
        try:
            context=self.nlu.extract_from_consumer(consumer)
            consumer=replace(consumer,objections=context.detected_objections,current_stage=context.current_stage)
            escalate=context.explicit_human_requested or context.handoff_recommended
            handoff=HandoffDecision('escalate_human' if escalate else 'keep_automated',
                'high' if escalate else 'none',context.handoff_reason or 'Decisão de apoio ao consultor, sem transferência executada.')
            chunks=self.store.search(context.raw_summary,method='dense',top_k=self.top_k)
            support=self.generator.generate_support(consumer,context,handoff,chunks,skip_review=True)
            return {'status':'completed','support':asdict(support),'handoff':asdict(handoff),
                    'interpretation':asdict(context),'retrieved_chunks':[asdict(c) for c in chunks]}
        finally:
            self.last_trace=[dict(event) for component in components for event in component.events]


def build_pipeline(method, *args, **kwargs):
    cls=ReviewedPipeline if method=='neuro_symbolic' else VectorPipeline if method=='vector' else Pipeline
    return cls(method,*args,**kwargs)
