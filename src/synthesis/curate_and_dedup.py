"""Curate, deduplicate and select balanced candidate questions using an independent model."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

CURATION_SYSTEM_PROMPT = """Você é um curador e auditor independente de dados para avaliação de sistemas de atendimento por IA no setor de energia brasileiro.
Você recebeu uma lista de perguntas candidatas geradas por outros modelos. Sua tarefa é avaliar cada pergunta com rigor metodológico.

Critérios de julgamento:
1. Relevância Setorial: Pertence estritamente ao domínio de energia por assinatura (GD compartilhada), mercado livre (ACL), faturas de concessionárias brasileiras (TUSD, TE, Fio B) ou atendimento/suporte comercial.
2. Realismo Conversacional: Soa como uma mensagem real de um consumidor no WhatsApp brasileiro (não como pergunta artificial de livro didático).
3. Redundância: Se a pergunta for quase idêntica a outra já avaliada, marque como duplicada.
4. Clareza da Intenção: Deve ser possível identificar o que o consumidor quer saber, contestar ou solicitar.

Para cada item avaliado, retorne um veredito ('KEEP' ou 'DISCARD') com justificativa sucinta e categoria refinada.

Responda EXCLUSIVAMENTE em formato JSON com uma lista de objetos:
[
  {
    "id": "ID original da pergunta",
    "verdict": "KEEP | DISCARD",
    "discard_reason": "null se KEEP, ou motivo se DISCARD (ex: duplicada, irrelevante, irrealista, ambigua)",
    "category": "energia_assinatura_gd | mercado_livre_acl | leitura_fatura_tarifas | objecoes_e_seguranca | pos_venda_e_suporte | solicitacao_humana",
    "quality_score": 1 a 10
  }
]
"""


def _clean_json(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    data = json.loads(text)
    if isinstance(data, dict) and "evaluations" in data:
        data = data["evaluations"]
    if not isinstance(data, list):
        raise ValueError("Curation output must be a JSON list")
    return data


def string_similarity(a: str, b: str) -> float:
    """Compute lexical sequence similarity ratio between two questions."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def prefilter_lexical_duplicates(cases: list[dict[str, Any]], threshold: float = 0.85) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter out near-identical string duplicates prior to LLM curation."""
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []

    for case in cases:
        q = case["question"].strip()
        is_dup = False
        dup_target = None
        for prev in kept:
            if string_similarity(q, prev["question"]) >= threshold:
                is_dup = True
                dup_target = prev["id"]
                break

        if is_dup:
            c_copy = dict(case)
            c_copy["verdict"] = "DISCARD"
            c_copy["discard_reason"] = f"lexical_near_duplicate_of_{dup_target}"
            c_copy["quality_score"] = 0
            discarded.append(c_copy)
        else:
            kept.append(case)

    return kept, discarded


def evaluate_candidates_with_llm(
    candidates: list[dict[str, Any]],
    model: str,
    completion_fn: Callable[..., Any] | None = None,
    batch_size: int = 25,
    timeout_s: int = 120,
) -> list[dict[str, Any]]:
    """Submit batches of questions to an independent curator model."""
    if completion_fn is None:
        from litellm import completion as completion_fn

    evaluations: list[dict[str, Any]] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        items_payload = [{"id": c["id"], "question": c["question"], "source_model": c.get("source_model")} for c in batch]

        prompt = (
            "Avalie o seguinte lote de perguntas candidatas. Retorne um veredito estruturado para cada ID:\n"
            + json.dumps(items_payload, ensure_ascii=False, indent=2)
        )

        response = completion_fn(
            model=model,
            messages=[
                {"role": "system", "content": CURATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=timeout_s,
        )

        parsed = _clean_json(response.choices[0].message.content)
        evaluations.extend(parsed)

    return evaluations


def curate_dataset(
    raw_cases_path: Path | str,
    target_count: int = 100,
    curator_model: str = "openai/gpt-4o",
    completion_fn: Callable[..., Any] | None = None,
    output_curated_path: Path | str = "data/synthesis/curated_candidates.jsonl",
    output_discarded_path: Path | str = "data/synthesis/discarded_cases.jsonl",
    output_report_path: Path | str = "data/synthesis/curation_report.json",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Curate raw cases down to target_count balanced questions."""
    raw_cases_file = Path(raw_cases_path)
    if not raw_cases_file.exists():
        raise FileNotFoundError(f"Raw cases file not found: {raw_cases_file}")

    with open(raw_cases_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print(f"[curate] Loaded {len(cases)} raw cases.", file=sys.stderr)

    # Step 1: Lexical deduplication
    unique_candidates, lex_discarded = prefilter_lexical_duplicates(cases, threshold=0.85)
    print(f"[curate] Filtered {len(lex_discarded)} lexical duplicates. Remaining: {len(unique_candidates)}", file=sys.stderr)

    # Step 2: LLM Curation
    print(f"[curate] Evaluating with curator model '{curator_model}'...", file=sys.stderr)
    llm_evals = evaluate_candidates_with_llm(
        unique_candidates,
        model=curator_model,
        completion_fn=completion_fn,
    )

    eval_by_id = {item["id"]: item for item in llm_evals if "id" in item}

    approved_pool: list[dict[str, Any]] = []
    llm_discarded: list[dict[str, Any]] = []

    for case in unique_candidates:
        ev = eval_by_id.get(case["id"])
        merged = dict(case)
        if ev and ev.get("verdict") == "KEEP":
            merged["category"] = ev.get("category", case.get("suggested_category"))
            merged["quality_score"] = ev.get("quality_score", 8)
            merged["curator_model"] = curator_model
            approved_pool.append(merged)
        else:
            merged["verdict"] = "DISCARD"
            merged["discard_reason"] = ev.get("discard_reason", "rejected_by_curator") if ev else "no_curator_evaluation"
            merged["curator_model"] = curator_model
            llm_discarded.append(merged)

    all_discarded = lex_discarded + llm_discarded
    print(f"[curate] Approved pool: {len(approved_pool)}, Discarded: {len(all_discarded)}", file=sys.stderr)

    # Step 3: Stratified selection to exactly target_count (e.g. 100)
    # Sort approved by quality_score descending
    approved_pool.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    by_category = defaultdict(list)
    for item in approved_pool:
        by_category[item.get("category", "geral")].append(item)

    selected: list[dict[str, Any]] = []
    # Guarantee round-robin across categories until target_count or pool exhausted
    cats = list(by_category.keys())
    idx = 0
    while len(selected) < target_count and any(by_category[c] for c in cats):
        c = cats[idx % len(cats)]
        if by_category[c]:
            selected.append(by_category[c].pop(0))
        idx += 1

    # If still need more to hit target_count, take remainder of approved
    if len(selected) < target_count:
        for c in cats:
            while by_category[c] and len(selected) < target_count:
                selected.append(by_category[c].pop(0))

    # Renumber selected cases sequentially for clean identification
    for seq, item in enumerate(selected, start=1):
        item["curated_id"] = f"CASE-{seq:03d}"

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "curator_model": curator_model,
        "total_raw_input": len(cases),
        "total_lexical_duplicates": len(lex_discarded),
        "total_llm_discarded": len(llm_discarded),
        "total_discarded": len(all_discarded),
        "total_selected": len(selected),
        "selected_by_category": dict(Counter(s.get("category") for s in selected)),
        "selected_by_source_model": dict(Counter(s.get("source_model") for s in selected)),
        "discard_reasons": dict(Counter(d.get("discard_reason", "unknown") for d in all_discarded)),
    }

    # Save outputs
    out_curated = Path(output_curated_path)
    out_curated.parent.mkdir(parents=True, exist_ok=True)
    with open(out_curated, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_disc = Path(output_discarded_path)
    out_disc.parent.mkdir(parents=True, exist_ok=True)
    with open(out_disc, "w", encoding="utf-8") as f:
        for item in all_discarded:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_rep = Path(output_report_path)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[curate] Curation complete. Retained {len(selected)} cases. Report saved to {out_rep}", file=sys.stderr)
    return selected, all_discarded, report


def main():
    parser = argparse.ArgumentParser(description="Curate and deduplicate questions down to 100 cases")
    parser.add_argument("--input", type=str, default="data/synthesis/raw_generated_cases.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--curator-model", type=str, default="openai/gpt-4o")
    parser.add_argument("--curated-output", type=str, default="data/synthesis/curated_candidates.jsonl")
    parser.add_argument("--discarded-output", type=str, default="data/synthesis/discarded_cases.jsonl")
    parser.add_argument("--report-output", type=str, default="data/synthesis/curation_report.json")
    args = parser.parse_args()

    curate_dataset(
        raw_cases_path=args.input,
        target_count=args.count,
        curator_model=args.curator_model,
        output_curated_path=args.curated_output,
        output_discarded_path=args.discarded_output,
        output_report_path=args.report_output,
    )


if __name__ == "__main__":
    main()
