"""Motor Simbólico Heurístico Puro (Linha de Base / Baseline Simbólico).

Executa regras determinísticas e correspondência léxica de palavras-chave sem chamadas
a modelos de linguagem (LLMs) ou embeddings.
"""
from __future__ import annotations

import re
from typing import Any

from .models import (
    ClosingSupport,
    ConsumerProfile,
    HandoffDecision,
)
from .nlu_extractor import ExtractedDealContext


class HeuristicNLUExtractor:
    """Extração de intenção e contexto baseada em regras léxicas e expressões regulares."""
    execution_kind = "heuristic_rules"
    model_name = "symbolic-regex-v1"

    HUMAN_KEYWORDS = [
        r"\bhumano\b", r"\batendente\b", r"\bfalar com (alguem|pessoa|humano|corretor)\b",
        r"\bpessoa de verdade\b", r"\bespecialista\b", r"\bcorretor\b"
    ]
    CANCEL_KEYWORDS = [
        r"\bcancela(r|mento)?\b", r"\bmulta\b", r"\brescis(ao|ão)\b", r"\bfidelidade\b"
    ]
    INTERRUPT_KEYWORDS = [
        r"\bfalt(ar|ou)? luz\b", r"\bqueda de energia\b", r"\binterrup(cao|ção|coes|ções)\b",
        r"\bsem luz\b", r"\bquebr(ar|ou)\b"
    ]
    BILL_KEYWORDS = [
        r"\bduas (contas|faturas)\b", r"\bdupla cobran(ca|ça)\b", r"\bcobranca dobrada\b",
        r"\bduas vezes\b", r"\bconta da distribuidora\b"
    ]
    TENANT_KEYWORDS = [
        r"\balug(uel|ado|ar)?\b", r"\binquilino\b", r"\bdono do imovel\b", r"\bproprietario\b",
        r"\blocacao\b"
    ]
    URGENT_KEYWORDS = [
        r"\burgente\b", r"\bagora\b", r"\bimediato\b", r"\bhoje\b", r"\brapido\b"
    ]

    def __init__(self):
        self.events = []

    def extract_from_consumer(self, consumer: ConsumerProfile) -> ExtractedDealContext:
        full_text = " ".join([m.get("content", "") for m in consumer.chat_transcript]).lower()

        # Detecção de distribuidora
        dist = None
        for d in ["cemig", "cpfl", "enel", "copel", "light", "edp"]:
            if d in full_text:
                dist = d.upper()
                break

        # Consumo aproximado
        kwh = consumer.monthly_consumption_kwh or None
        kwh_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", full_text)
        if kwh_match:
            try:
                val = float(kwh_match.group(1))
                if val >= 0:
                    kwh = val
            except ValueError:
                pass

        # Objeções detectadas
        objections = []
        if any(re.search(pat, full_text) for pat in self.CANCEL_KEYWORDS):
            objections.append("cancelamento_e_fidelidade")
        if any(re.search(pat, full_text) for pat in self.INTERRUPT_KEYWORDS):
            objections.append("garantia_de_fornecimento")
        if any(re.search(pat, full_text) for pat in self.BILL_KEYWORDS):
            objections.append("duas_faturas")
        if any(re.search(pat, full_text) for pat in self.TENANT_KEYWORDS):
            objections.append("titularidade_inquilino")

        # Intenção de escalada humana
        explicit_human = any(re.search(pat, full_text) for pat in self.HUMAN_KEYWORDS)
        is_urgent = any(re.search(pat, full_text) for pat in self.URGENT_KEYWORDS)

        # Sentimento simplificado
        sentiment = "neutro"
        if any(w in full_text for w in ["ótimo", "bom", "legal", "interessante", "quero"]):
            sentiment = "positivo"
        elif any(w in full_text for w in ["medo", "receio", "preocupado", "estranho", "dúvida"]):
            sentiment = "cauteloso"
        elif any(w in full_text for w in ["absurdo", "ruim", "péssimo", "propaganda enganosa", "processo"]):
            sentiment = "hostil"

        # Estágio e intenção
        handoff_rec = bool(explicit_human or ("cancelamento_e_fidelidade" in objections))
        handoff_reason = "Palavra-chave indicativa de atendimento humano." if explicit_human else (
            "Dúvida crítica identificada por regra simbólica." if handoff_rec else ""
        )

        return ExtractedDealContext(
            distributor=dist or consumer.distributor,
            monthly_consumption_kwh=kwh,
            detected_objections=objections,
            sentiment=sentiment,
            friction_score=0.8 if sentiment == "hostil" else (0.5 if sentiment == "cauteloso" else 0.1),
            explicit_human_requested=explicit_human,
            is_immediate_urgency=is_urgent,
            current_stage="descoberta" if not consumer.bill_facts else "proposta",
            key_intent="atendimento_humano" if explicit_human else (objections[0] if objections else "duvida_geral"),
            raw_summary=f"Extração heurística simbólica baseada em {len(objections)} palavras-chave.",
            handoff_recommended=handoff_rec,
            handoff_reason=handoff_reason,
            interpretation_source="heuristic_rules",
        )


class HeuristicClosingGenerator:
    """Gera respostas a partir de templates estáticos pré-fabricados (sem chamada a LLM)."""
    execution_kind = "heuristic_templates"
    model_name = "static-template-v1"

    TEMPLATES = {
        "cancelamento_e_fidelidade": "Podemos conferir as condições de cancelamento e eventual fidelidade no contrato da oferta escolhida?",
        "garantia_de_fornecimento": "A distribuidora continua responsável pela rede física. A contratação de energia não elimina o risco de interrupções.",
        "duas_faturas": "Vamos conferir a fatura da distribuidora e a cobrança do fornecedor para identificar as parcelas e comparar o custo total?",
        "titularidade_inquilino": "Podemos verificar a titularidade da conta e as condições da oferta para um imóvel alugado?",
        "default": "Qual condição da proposta você gostaria de esclarecer? Para estimar economia, precisamos verificar os dados da fatura e da oferta."
    }

    def __init__(self):
        self.events = []

    def generate_support(
        self,
        consumer: ConsumerProfile,
        extracted_context: ExtractedDealContext,
        handoff: HandoffDecision,
        retrieved_knowledge: list[Any],
    ) -> ClosingSupport:
        # Seleciona template pela objeção prioritária
        primary = extracted_context.detected_objections[0] if extracted_context.detected_objections else "default"
        text = self.TEMPLATES.get(primary, self.TEMPLATES["default"])

        return ClosingSupport(
            next_best_action="enviar_template_predefinido",
            strategic_rationale="Resposta padronizada baseada em casamento de palavras-chave do dicionário operacional.",
            primary_risk="Rigidez comunicacional e potencial falta de contextualização para perguntas atípicas.",
            risk_mitigation="Recomendar escalada para atendimento humano se o cliente insistir na dúvida.",
            suggested_talk_track=text,
            high_construal_guidance=f"Esclarecer a dúvida sobre {primary} com base no roteiro padrão da empresa.",
            low_construal_template=text,
            generation_source="heuristic_template",
            grounding_review={"status": "not_reviewed", "protocol": "symbolic-template-v1"},
            anti_manipulation_flags=[],
        )
