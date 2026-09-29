from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BillFacts:
    """Explicit document evidence; synthetic fixtures must never masquerade as real bills."""
    source_id: str
    source_kind: str  # document_extraction | synthetic_bill_fixture
    monthly_consumption_kwh: float | None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Bill evidence requires a source ID")
        if self.source_kind not in {"document_extraction", "synthetic_bill_fixture"}:
            raise ValueError("Unsupported bill evidence source")
        value = self.monthly_consumption_kwh
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Bill consumption must be a finite nonnegative number or unknown")


@dataclass(frozen=True)
class OfferProjection:
    source_id: str
    source_kind: str  # offer_projection | synthetic_offer_fixture
    proposal_id: str
    product_id: str
    selected_product_id: str | None = None
    expected_yearly_savings: float | None = None
    expected_bill_cost: float | None = None

    def __post_init__(self):
        if self.source_kind not in {"offer_projection", "synthetic_offer_fixture"}:
            raise ValueError("Invalid projection origin")
        if any(not isinstance(v, str) or not v.strip() for v in (self.source_id, self.proposal_id, self.product_id)):
            raise ValueError("Projection requires source, proposal and product IDs")
        for value in (self.expected_yearly_savings, self.expected_bill_cost):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Projection must be finite or unknown")


@dataclass(frozen=True)
class CatalogTerms:
    source_id: str
    source_kind: str  # current_product_catalog | synthetic_catalog_fixture
    product_id: str
    loyalty_months: int | None = None

    def __post_init__(self):
        if self.source_kind not in {"current_product_catalog", "synthetic_catalog_fixture"}:
            raise ValueError("Invalid catalog origin")
        if any(not isinstance(v, str) or not v.strip() for v in (self.source_id, self.product_id)):
            raise ValueError("Catalog requires source and product IDs")
        if self.loyalty_months is not None and (type(self.loyalty_months) is not int or self.loyalty_months < 0):
            raise ValueError("Loyalty must be nonnegative months or unknown")


@dataclass
class ConsumerProfile:
    consumer_id: str
    name: str
    segment: str
    region: str
    consumption_profile: str
    interests: list[str]
    maturity: str
    urgency: str
    restrictions: list[str]
    objections: list[str]
    current_stage: str
    distributor: str | None = None
    monthly_consumption_kwh: float | None = None
    icp_category: str = "unknown"  # icp_1 (PME BT), icp_2 (Redes/Multi-UC), icp_3 (Condominios), icp_4 (Grupo A/ACL), non_icp
    installation_code: str = ""  # Codigo de instalacao da UC
    tenant_status: str = "unknown"  # titular | inquilino_sem_titularidade | imovel_proprio
    chat_transcript: list[dict[str, str]] = field(default_factory=list)
    notes: str = ""
    bill_facts: BillFacts | None = None
    offer_projection: OfferProjection | None = None
    catalog_terms: CatalogTerms | None = None


@dataclass
class HandoffDecision:
    action: str  # "keep_automated" | "escalate_human"
    urgency: str  # "none" | "low" | "high" | "immediate"
    rationale: str
    triggered_policies: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class ClosingSupport:
    next_best_action: str
    strategic_rationale: str
    primary_risk: str
    risk_mitigation: str
    suggested_talk_track: str
    # Dual-Construal Enablement (Casenave & Schmitt, 2025)
    high_construal_guidance: str = ""  # Diretriz estratégica, intenção psicológica e objetivo conceitual ("o porquê")
    low_construal_template: str = ""  # Template operacional flexível com campos adaptáveis ("o como")
    # Guardrail de Transparência e Anti-Manipulação (Liu, Wang & Zhu, 2025)
    transparency_score: float | None = None  # No independent assessment has been collected.
    generation_source: str = "unverified"
    grounding_review: dict[str, Any] = field(default_factory=lambda: {"status": "not_reviewed"})
    anti_manipulation_flags: list[str] = field(default_factory=list)
