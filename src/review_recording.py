"""Preserve review evidence without changing prompts or generation parameters."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

from .closing_llm import GroundedClosingLLM, ReviewRejected, coaching_sources, support_review_text
from .coaching_units import draft_units

# Messages originate in local validators, never in provider exception bodies.
REASONS = {
    'Unsupported claim': ('unsupported_claim', 'content'),
    'Factual audit found unsupported or contradicted claim': ('unsupported_or_contradicted_claim', 'content'),
    'Completed action lacks operational confirmation': ('unconfirmed_action', 'content'),
    'Missing, duplicate, reordered or foreign draft units': ('unit_coverage_mismatch', 'coverage'),
    'Missing coverage': ('missing_coverage', 'coverage'),
    'Review omitted or changed draft text': ('draft_coverage_mismatch', 'coverage'),
    'Factual audit omitted or changed draft text': ('fact_coverage_mismatch', 'coverage'),
    'Factual claim not anchored in draft': ('claim_not_in_draft', 'coverage'),
    'Completed action not anchored in draft': ('action_not_in_draft', 'coverage'),
    'Mixed unit audit must cover both case and general claims': ('mixed_claim_coverage', 'coverage'),
    'Facts require sources; nonfacts cannot launder citations': ('citation_classification_mismatch', 'citation'),
    'Citation not found in supplied evidence': ('citation_not_found', 'citation'),
    'Evidence has the wrong scope': ('evidence_scope_mismatch', 'citation'),
    'Mixed facts require general and case evidence, not operation receipts': ('mixed_evidence_scope', 'citation'),
    'Mixed facts require both general and case citations': ('mixed_citations_missing', 'citation'),
    'Completed action requires an operational citation': ('operation_citation_missing', 'citation'),
    'Empty citation': ('empty_citation', 'citation'),
}
SCHEMA_MESSAGES = {
    'Invalid unit review schema', 'Invalid unit fields', 'Invalid unit kind',
    'Invalid unit citation schema', 'Invalid completed actions schema',
    'Invalid completed action schema', 'Invalid action citation schema',
    'Invalid factual audit schema', 'Invalid factual audit unit', 'Invalid factual claim schema',
    'Invalid factual citation schema', 'Invalid review schema', 'Invalid segment',
    'Empty segment', 'Invalid citation', 'Generation output must contain exactly the required fields',
    'Generation text fields must be nonempty strings', 'Invalid generation flags',
}


def rejection_reason(exc):
    if isinstance(exc, json.JSONDecodeError):
        return {'code': 'invalid_json', 'category': 'schema', 'validator_message': None}
    message = str(exc)
    if message in REASONS:
        code, category = REASONS[message]
        return {'code': code, 'category': category, 'validator_message': message}
    if message in SCHEMA_MESSAGES:
        return {'code': 'invalid_response_schema', 'category': 'schema', 'validator_message': message}
    return {'code': 'unclassified_rejection' if isinstance(exc, ReviewRejected) else 'unexpected_error',
            'category': 'validation' if isinstance(exc, ReviewRejected) else 'execution',
            'validator_message': None}


class RecordedReviewLLM(GroundedClosingLLM):
    def __init__(self, *args, completion=None, **kwargs):
        super().__init__(*args, completion=completion, **kwargs)
        self._provider = completion
        self._completion = self._record_completion
        self.review_audit = None
        self._stage = 'generation'

    def _record_completion(self, **kwargs):
        stage = self._attempt[self._stage]
        try:
            if self._provider is None:
                from litellm import completion
                response = completion(**kwargs)
            else:
                response = self._provider(**kwargs)
        except Exception as exc:
            stage.update(status='failed', error_type=type(exc).__name__,
                         reason={'code':'provider_failure', 'category':'provider', 'validator_message':None})
            raise
        stage['response_text'] = response.choices[0].message.content
        return response

    def _start_attempt(self, kind):
        self._attempt = {'index':len(self.review_audit['attempts']), 'kind':kind,
                         'draft':None, 'draft_text':None, 'draft_sha256':None, 'draft_units':None,
                         'generation':{'status':'running'}, 'grounding':None, 'fact_audit':None,
                         'decision':'pending'}
        self.review_audit['attempts'].append(self._attempt)
        self._stage = 'generation'

    def _stage_error(self, stage, exc):
        if stage.get('reason', {}).get('category') != 'provider':
            stage.update(status='rejected' if isinstance(exc, ReviewRejected) else 'failed',
                         error_type=type(exc).__name__, reason=rejection_reason(exc))
        if isinstance(exc, ReviewRejected):
            stage['report'] = deepcopy(exc.review)

    def generate_support(self, consumer, extracted_context, handoff, retrieved_knowledge, *, skip_review=False):
        if skip_review:
            raise ValueError('RecordedReviewLLM requires review; use the vector baseline for unreviewed output')
        self.review_audit = {'schema_version':1, 'sources':coaching_sources(consumer,retrieved_knowledge),
                             'conversation_context':deepcopy(consumer.chat_transcript),
                             'attempts':[], 'outcome':'running', 'delivered_attempt':None}
        self._start_attempt('initial')
        try:
            support = super().generate_support(consumer,extracted_context,handoff,retrieved_knowledge)
            self.review_audit.update(outcome='completed', delivered_attempt=self._attempt['index'])
            return support
        except Exception as exc:
            # Follow explicit causes, matching the pipeline's existing blocked/failed distinction.
            chain=[];cause=exc
            while cause is not None:
                chain.append(cause);cause=cause.__cause__
            blocked=any(isinstance(c,ReviewRejected) for c in chain)
            self.review_audit['outcome']='blocked' if blocked else 'failed'
            if self._attempt['decision']=='pending':
                self._stage_error(self._attempt['generation'],chain[-1])
                self._attempt['decision']='failed'
            raise

    def _check_support(self, support, sources, completion, conversation_context):
        text=support_review_text(support)
        self._attempt.update(draft=asdict(support),draft_text=text,
                             draft_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             draft_units=draft_units(text))
        self._attempt['generation']['status']='success'
        try:
            report=super()._check_support(support,sources,completion,conversation_context)
            self._attempt['decision']='approved'
            return report
        except Exception as exc:
            self._attempt['decision']='rejected' if isinstance(exc,ReviewRejected) else 'failed'
            raise

    def _capture_stage(self, name, callback):
        self._stage=name
        record=self._attempt[name]={'status':'running'}
        try:
            report=callback()
            record.update(status='approved',report=deepcopy(report))
            return report
        except Exception as exc:
            self._stage_error(record,exc)
            raise

    def _review_support(self, support, sources, completion, conversation_context):
        return self._capture_stage('grounding',lambda:super(RecordedReviewLLM,self)._review_support(
            support,sources,completion,conversation_context))

    def _audit_facts(self, support, sources, completion, conversation_context, *, review=None):
        return self._capture_stage('fact_audit',lambda:super(RecordedReviewLLM,self)._audit_facts(
            support,sources,completion,conversation_context,review=review))

    def _rewrite_support(self, support, sources, completion, conversation_context, rejection):
        self._start_attempt('rewrite')
        try:
            return super()._rewrite_support(support,sources,completion,conversation_context,rejection)
        except Exception as exc:
            self._stage_error(self._attempt['generation'],exc)
            self._attempt['decision']='failed'
            raise
