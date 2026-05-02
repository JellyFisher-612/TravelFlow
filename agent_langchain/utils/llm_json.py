"""Small JSON extraction helpers for LLM text fallbacks."""

from __future__ import annotations

import json
from typing import Any, Dict


def parse_json_text(text: str) -> Dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start_idx = clean.find("{")
        end_idx = clean.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return json.loads(clean[start_idx : end_idx + 1])
        raise
