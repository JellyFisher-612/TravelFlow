"""
Guard for LangChain structured output.

Purpose:
- Avoid repeated structured-output attempts for models that do not support
  response_format / structured output.
- Reduce latency caused by fail-then-fallback on every request.
"""

from __future__ import annotations

import os
from typing import Any, Dict


_STRUCTURED_CAPABILITY_CACHE: Dict[str, bool] = {}


def _model_cache_key(lc_model: Any) -> str:
    model_name = getattr(lc_model, "model_name", "")
    base_url = getattr(lc_model, "openai_api_base", "") or getattr(lc_model, "base_url", "")
    provider = getattr(lc_model, "__class__", type("_T", (), {})).__name__
    return f"{provider}|{model_name}|{base_url}"


def is_structured_output_disabled_by_env() -> bool:
    return os.getenv("DISABLE_STRUCTURED_OUTPUT", "false").lower() in {"1", "true", "yes", "on"}


def should_attempt_structured_output(lc_model: Any) -> bool:
    if lc_model is None:
        return False
    if is_structured_output_disabled_by_env():
        return False
    if not hasattr(lc_model, "with_structured_output"):
        return False

    key = _model_cache_key(lc_model)
    return _STRUCTURED_CAPABILITY_CACHE.get(key, True)


def mark_structured_output_unsupported(lc_model: Any):
    if lc_model is None:
        return
    key = _model_cache_key(lc_model)
    _STRUCTURED_CAPABILITY_CACHE[key] = False


def is_structured_output_unavailable_error(err: Exception) -> bool:
    text = str(err).lower()
    # Common provider responses when response_format is unsupported.
    return (
        "response_format" in text
        or "this response_format type is unavailable" in text
        or "invalid_request_error" in text
    )
