"""LangSmith tracing setup helper."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from config import LANGSMITH_CONFIG

logger = logging.getLogger(__name__)


def _read_langsmith_key_from_api_file() -> str:
    """Read LangSmith API key from workspace-level API.txt.

    Supported formats:
    - lsv2_xxx
    - langsmith: lsv2_xxx
    """
    try:
        api_file = Path(__file__).resolve().parents[2] / "API.txt"
        if not api_file.exists():
            return ""

        raw = api_file.read_text(encoding="utf-8").strip()
        if not raw:
            return ""

        if ":" in raw:
            _, value = raw.split(":", 1)
            return value.strip()
        return raw.strip()
    except Exception:
        return ""


def setup_langsmith_tracing() -> bool:
    """Initialize LangSmith tracing by exporting required env vars.

    Returns:
        True if tracing is enabled and configured, else False.
    """
    if not LANGSMITH_CONFIG.get("enabled", False):
        return False

    api_key = (LANGSMITH_CONFIG.get("api_key") or "").strip() or _read_langsmith_key_from_api_file()
    if not api_key:
        logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY is empty; tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_ENDPOINT"] = str(LANGSMITH_CONFIG.get("endpoint") or "https://api.smith.langchain.com")
    os.environ["LANGSMITH_PROJECT"] = str(LANGSMITH_CONFIG.get("project") or "travelflow-travel-agent")

    logger.info(
        "LangSmith tracing enabled: project=%s endpoint=%s",
        os.environ.get("LANGSMITH_PROJECT"),
        os.environ.get("LANGSMITH_ENDPOINT"),
    )
    return True
