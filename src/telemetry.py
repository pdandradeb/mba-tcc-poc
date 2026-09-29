"""Provider metadata safe to persist without credentials or raw exception messages."""
from __future__ import annotations


def response_metadata(response):
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    tokens = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name) if isinstance(usage, dict) else None
        tokens[name] = value if type(value) is int and value >= 0 else None
    model = getattr(response, "model", None)
    return {"returned_model": model if isinstance(model, str) else None, "tokens": tokens}
