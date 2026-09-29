"""Annotate candidate questions with suggested rubrics, grounding chunks and handoff targets using the Knowledge Base."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ANNOTATION_SYSTEM_PROMPT = """Você é um especialista em curadoria de conhecimento do setor elétrico e auditor de conformidade regulatória.
Sua missão é analisar perguntas de clientes sobre energia por assinatura e mercado livre de energia, comparando-as com a Base de Conhecimento (KB) canônica da empresa.

Para a pergunta recebida, você deve produzir:
1. 'category': uma das categorias:
   - energia_assinatura_gd (geração distribuída compartilhada, créditos solares)
   - mercado_livre_acl (migração Grupo A, comercializador varejista)
   - leitura_fatura_tarifas (TE, TUSD, Fio B, iluminação, bandeiras)
   - objecoes_e_seguranca (dúvidas de golpe, corte de energia, estabilidade)
   - pos_venda_e_suporte (fatura do mês sem desconto, atraso, cancelamento)
   - solicitacao_humana (pedido explícito de humano, atendente ou gerente)
2. 'intent_type':
   - duvida_conceitual
   - objecao_desconfianca
   - analise_fatura
   - cancelamento_reclamacao
   - solicitacao_humana
3. 'expected_facts_suggested': lista de 2 a 4 fatos atômicos essenciais que a resposta DEVE conter para ser considerada completa e precisa, embasados na KB.
4. 'forbidden_claims_suggested': lista de 2 a 4 afirmações proibidas que a resposta NUNCA deve fazer (ex: prometer ausência total de taxa de distribuição/Fio B, dizer que a empresa é dona da rede física da concessionária, prometer 100% de desconto, afirmar cancelamento sem multa sem ter contrato, induzir urgência artificial com falsas últimas cotas).
5. 'handoff_expected_suggested': booleano (true ou false):
   - true se a mensagem EXIGE intervenção de atendente/corretor humano (pedido explícito de pessoa, pedido ativo de cancelamento contratual, contestação de fatura individual específica com valores/dados pessoais, negociação customizada).
   - false se for dúvida geral, explicação de regras, cotação orientativa ou FAQ que pode ser atendida pelo sistema de apoio.
6. 'handoff_reason_suggested': justificativa objetiva da recomendação de handoff (ou de sua dispensa).
7. 'grounding_chunk_ids_suggested': lista com até 3 IDs de trechos da KB (ex: 'pub-...') que fundamentam os fatos esperados.

Responda EXCLUSIVAMENTE em formato JSON com o seguinte formato:
{
  "category": "...",
  "intent_type": "...",
  "expected_facts_suggested": ["fato 1", "fato 2"],
  "forbidden_claims_suggested": ["proibição 1", "proibição 2"],
  "handoff_expected_suggested": false,
  "handoff_reason_suggested": "...",
  "grounding_chunk_ids_suggested": ["pub-..."]
}
"""


def load_corpus_summary(corpus_path: Path | str) -> list[dict[str, Any]]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"chunk_id": c["chunk_id"], "title": c["title"], "content": c["content"][:400] + ("..." if len(c["content"]) > 400 else "")}
        for c in data
    ]


def _clean_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def annotate_single_case(
    case: dict[str, Any],
    corpus_summary: list[dict[str, Any]],
    model: str,
    completion_fn: Callable[..., Any] | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Annotate a single question with rubrics and handoff ground truth."""
    if completion_fn is None:
        from litellm import completion as completion_fn

    user_payload = {
        "question": case["question"],
        "metadata": {
            "source_model": case.get("source_model"),
            "category_hint": case.get("category") or case.get("suggested_category"),
        },
        "available_knowledge_base_chunks": corpus_summary,
    }

    response = completion_fn(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        timeout=timeout_s,
    )

    data = _clean_json(response.choices[0].message.content)

    return {
        "category": data.get("category", case.get("category", "energia_assinatura_gd")),
        "intent_type": data.get("intent_type", "duvida_conceitual"),
        "expected_facts_suggested": data.get("expected_facts_suggested", []),
        "forbidden_claims_suggested": data.get("forbidden_claims_suggested", []),
        "handoff_expected_suggested": bool(data.get("handoff_expected_suggested", False)),
        "handoff_reason_suggested": data.get("handoff_reason_suggested", ""),
        "grounding_chunk_ids_suggested": data.get("grounding_chunk_ids_suggested", []),
    }


from concurrent.futures import ThreadPoolExecutor, as_completed

def annotate_dataset(
    curated_cases_path: Path | str,
    corpus_path: Path | str = "data/public_corpus.json",
    annotator_model: str = "openai/gpt-4o",
    completion_fn: Callable[..., Any] | None = None,
    output_path: Path | str = "data/synthesis/candidates_with_suggested_rubrics.jsonl",
    workers: int = 5,
) -> list[dict[str, Any]]:
    """Annotate all curated cases with suggested rubrics in parallel, preparing them for human review."""
    curated_file = Path(curated_cases_path)
    if not curated_file.exists():
        raise FileNotFoundError(f"Curated cases file not found: {curated_file}")

    corpus_summary = load_corpus_summary(corpus_path)
    print(f"[annotate] Loaded {len(corpus_summary)} corpus chunks for grounding.", file=sys.stderr)

    with open(curated_file, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing annotated records to support resume
    existing_by_curated_id = {}
    if out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if "curated_id" in r:
                            existing_by_curated_id[r["curated_id"]] = r
            if existing_by_curated_id:
                print(f"[annotate] Resuming: found {len(existing_by_curated_id)} already annotated cases.", file=sys.stderr)
        except Exception:
            existing_by_curated_id = {}

    print(f"[annotate] Annotating {len(candidates)} cases with model '{annotator_model}' ({workers} concurrent workers)...", file=sys.stderr)

    def process_item(item):
        idx, case = item
        cid = case.get("curated_id", f"CASE-{idx:03d}")
        if cid in existing_by_curated_id:
            return idx, existing_by_curated_id[cid]

        q = case["question"]
        try:
            ann = annotate_single_case(case, corpus_summary, model=annotator_model, completion_fn=completion_fn)
        except Exception as exc:
            print(f"[annotate] ERROR annotating {cid}: {exc}. Using fallback baseline.", file=sys.stderr)
            ann = {
                "category": case.get("category", "energia_assinatura_gd"),
                "intent_type": "duvida_geral",
                "expected_facts_suggested": ["Informar regras do modelo de energia", "Apresentar condições comerciais de atendimento"],
                "forbidden_claims_suggested": ["Prometer 100% de desconto", "Dizer que a empresa substitui a distribuidora"],
                "handoff_expected_suggested": False,
                "handoff_reason_suggested": "Atendimento automatizado padrão",
                "grounding_chunk_ids_suggested": [],
            }

        record = {
            "id": f"REV-{idx:03d}",
            "curated_id": cid,
            "original_id": case.get("id"),
            "question": q,
            "source_model": case.get("source_model", "unknown"),
            "curator_model": case.get("curator_model", "unknown"),
            "annotator_model": annotator_model,
            "category": ann["category"],
            "intent_type": ann["intent_type"],
            "expected_facts_suggested": ann["expected_facts_suggested"],
            "forbidden_claims_suggested": ann["forbidden_claims_suggested"],
            "handoff_expected_suggested": ann["handoff_expected_suggested"],
            "handoff_reason_suggested": ann["handoff_reason_suggested"],
            "grounding_chunk_ids_suggested": ann["grounding_chunk_ids_suggested"],
            "expected_facts_reviewed": [],
            "forbidden_claims_reviewed": [],
            "handoff_expected_reviewed": None,
            "handoff_reason_reviewed": "",
            "grounding_chunk_ids_reviewed": [],
            "review_status": "pending",
            "rejection_reason": "",
            "reviewed_by": "",
            "reviewed_at": None,
        }
        print(f"[annotate] [{idx}/{len(candidates)}] Annotated {cid} successfully.", file=sys.stderr)
        return idx, record

    indexed_cases = list(enumerate(candidates, start=1))
    results_map = {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_item, item): item for item in indexed_cases}
        for future in as_completed(futures):
            idx, rec = future.result()
            results_map[idx] = rec

    ordered_records = [results_map[i] for i in range(1, len(candidates) + 1)]

    with open(out, "w", encoding="utf-8") as f:
        for item in ordered_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[annotate] Saved {len(ordered_records)} annotated records ready for review to {out}", file=sys.stderr)
    return ordered_records


def main():
    parser = argparse.ArgumentParser(description="Annotate curated cases with suggested rubrics")
    parser.add_argument("--input", type=str, default="data/synthesis/curated_candidates.jsonl")
    parser.add_argument("--corpus", type=str, default="data/public_corpus.json")
    parser.add_argument("--annotator-model", type=str, default="openai/gpt-4o")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=str, default="data/synthesis/candidates_with_suggested_rubrics.jsonl")
    args = parser.parse_args()

    annotate_dataset(
        curated_cases_path=args.input,
        corpus_path=args.corpus,
        annotator_model=args.annotator_model,
        output_path=args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
