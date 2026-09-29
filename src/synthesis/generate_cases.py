"""Generate candidate commercial questions across diverse LLM providers and models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SYNTHESIS_CATEGORIES = [
    "energia_assinatura_gd",
    "mercado_livre_acl",
    "leitura_fatura_tarifas",
    "objecoes_e_seguranca",
    "pos_venda_e_suporte",
    "solicitacao_humana",
]

GENERATION_SYSTEM_PROMPT = """Você é um especialista em geração de dados sintéticos realistas para avaliação de sistemas de atendimento comercial por WhatsApp no Brasil.
Seu objetivo é gerar dúvidas e mensagens autênticas de consumidores interessados ou com dúvidas sobre energia por assinatura (geração distribuída compartilhada), mercado livre de energia (ACL) e faturas de energia elétrica.

Diretrizes essenciais:
1. Realismo do canal WhatsApp: mensagens curtas ou médias, linguagem natural de WhatsApp brasileiro (coloquialismos, pontuação informal, abreviações usuais como 'vc', 'pq', 'tb', sem formalismos artificiais de robô).
2. Variedade de estilos e tons:
   - Dúvidas ingênuas de leigos ("precisa colocar placa no meu telhado?");
   - Dúvidas céticas / receio de golpe ("vi esse anúncio mas parece bom demais pra ser verdade, é golpe?");
   - Dúvidas técnicas ou financeiras ("como fica o Fio B da Lei 14.300 na minha conta?");
   - Problemas e atritos reais ("recebi duas contas este mês, por que a Enel ainda me cobrou?");
   - Solicitações explícitas de contato humano ("me transfere pra um atendente de verdade", "quero falar com uma pessoa agora");
   - Clientes residenciais, pequenos comerciantes (padaria, açougue, loja), inquilinos em imóveis alugados.
3. Não invente termos irreais; baseie-se no ecossistema elétrico brasileiro (CEMIG, Enel, CPFL, Equatorial, Neoenergia, TUSD, TE, Fio B, geração solar/biomassa).
4. As perguntas devem ser independentes e representar um primeiro contato ou turno comercial crítico.

Responda EXCLUSIVAMENTE em formato JSON com uma lista de objetos contendo exatamente:
[
  {
    "question": "texto exato da mensagem do consumidor",
    "suggested_category": "uma das categorias: energia_assinatura_gd, mercado_livre_acl, leitura_fatura_tarifas, objecoes_e_seguranca, pos_venda_e_suporte, solicitacao_humana",
    "tone": "coloquial | cetico | tecnico | formal | urgente | confuso",
    "persona": "residencial_b2c | comercial_b2b_pme | inquilino | propenso_cancelamento"
  }
]
"""


def _clean_json_response(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    data = json.loads(text)
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError("Model output must be a JSON list of questions")
    return data


def _generate_sub_batch(
    model: str,
    count: int,
    categories: Sequence[str],
    completion_fn: Callable[..., Any],
    timeout_s: int,
) -> list[dict[str, Any]]:
    user_prompt = (
        f"Gere exatamente {count} mensagens de WhatsApp de consumidores brasileiros "
        f"cobrindo de forma equilibrada as seguintes categorias: {', '.join(categories)}. "
        f"Certifique-se de incluir pelo menos 2 mensagens com solicitação explícita de atendente humano "
        f"e variações informais realistas."
    )

    response = completion_fn(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        timeout=timeout_s,
    )

    content = response.choices[0].message.content
    items = _clean_json_response(content)

    results = []
    provider = model.split("/")[0] if "/" in model else "unknown"
    for item in items:
        if not isinstance(item, dict) or "question" not in item:
            continue
        q = str(item.get("question", "")).strip()
        if not q or len(q) < 8:
            continue
        results.append({
            "question": q,
            "source_model": model,
            "source_provider": provider,
            "suggested_category": item.get("suggested_category", "energia_assinatura_gd"),
            "tone": item.get("tone", "coloquial"),
            "persona": item.get("persona", "residencial_b2c"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return results


def generate_batch_for_model(
    model: str,
    target_count: int,
    categories: Sequence[str] = SYNTHESIS_CATEGORIES,
    completion_fn: Callable[..., Any] | None = None,
    timeout_s: int = 90,
    chunk_size: int = 20,
) -> list[dict[str, Any]]:
    """Generate target_count questions in manageable sub-batches to prevent truncation."""
    if completion_fn is None:
        from litellm import completion as completion_fn

    all_items = []
    remaining = target_count
    sub_count = min(remaining, chunk_size)

    while remaining > 0:
        take = min(remaining, sub_count)
        sub_items = _generate_sub_batch(model, take, categories, completion_fn, timeout_s)
        if not sub_items:
            break
        all_items.extend(sub_items)
        remaining -= len(sub_items)
        if len(sub_items) < take // 2:
            break

    return all_items[:target_count] if len(all_items) > target_count else all_items


def generate_cases_across_models(
    models: Sequence[str],
    count_per_model: int = 40,
    completion_fn: Callable[..., Any] | None = None,
    output_path: Path | str | None = None,
    resume_existing: bool = False,
) -> list[dict[str, Any]]:
    """Runs generation across multiple models, assigns unique IDs and saves raw candidates."""
    all_cases: list[dict[str, Any]] = []
    case_counter = 1

    out = Path(output_path) if output_path else None
    existing_by_model: dict[str, list[dict[str, Any]]] = {}

    if resume_existing and out and out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        m = item.get("source_model", "")
                        existing_by_model.setdefault(m, []).append(item)
                        all_cases.append(item)
            case_counter = len(all_cases) + 1
            print(f"[synthesis] Loaded {len(all_cases)} existing cases from {out} to resume.", file=sys.stderr)
        except Exception as exc:
            print(f"[synthesis] Warning: failed to load existing cases: {exc}", file=sys.stderr)
            all_cases = []
            existing_by_model = {}

    for model in models:
        existing_count = len(existing_by_model.get(model, []))
        if resume_existing and existing_count >= count_per_model:
            print(f"[synthesis] Skipping '{model}': already has {existing_count} cases.", file=sys.stderr)
            continue

        needed = count_per_model - existing_count if resume_existing else count_per_model
        try:
            print(f"[synthesis] Generating {needed} questions via model '{model}'...", file=sys.stderr)
            cases = generate_batch_for_model(
                model=model,
                target_count=needed,
                completion_fn=completion_fn,
            )
            for c in cases:
                c["id"] = f"GEN-{case_counter:04d}"
                case_counter += 1
                all_cases.append(c)
            print(f"[synthesis] Model '{model}' generated {len(cases)} valid items.", file=sys.stderr)
        except Exception as exc:
            print(f"[synthesis] ERROR generating with '{model}': {exc}", file=sys.stderr)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for item in all_cases:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[synthesis] Saved {len(all_cases)} raw cases to {out}", file=sys.stderr)

    return all_cases


def get_default_models() -> list[str]:
    """Detect available models based on existing API keys."""
    models = []
    if os.getenv("OPENAI_API_KEY"):
        models.append("openai/gpt-4o-mini")
    if os.getenv("GEMINI_API_KEY"):
        models.append("gemini/gemini-2.5-flash")
    if os.getenv("OPENROUTER_API_KEY"):
        models.append("openrouter/anthropic/claude-3-haiku")
    return models or ["gemini/gemini-2.5-flash"]


def main():
    parser = argparse.ArgumentParser(description="Multi-model synthetic case generation for energy sales benchmark")
    parser.add_argument("--models", type=str, default="", help="Comma-separated litellm model identifiers")
    parser.add_argument("--count-per-model", type=int, default=40, help="Number of questions to generate per model")
    parser.add_argument("--output", type=str, default="data/synthesis/raw_generated_cases.jsonl", help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file and only generate missing models")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else get_default_models()
    print(f"[synthesis] Target models: {models}", file=sys.stderr)
    generate_cases_across_models(
        models=models,
        count_per_model=args.count_per_model,
        output_path=args.output,
        resume_existing=args.resume,
    )


if __name__ == "__main__":
    main()
