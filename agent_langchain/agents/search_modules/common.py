"""Shared models and source-quality helpers for search modules."""

from __future__ import annotations

import importlib
import re
from typing import Any, Dict

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")


class SummaryOutput(BaseModel):
    summary: str = Field(default="")


class SearchPlanOutput(BaseModel):
    search_strategy: str = Field(default="")
    demand_profile: dict = Field(default_factory=dict)
    tasks: list = Field(default_factory=list)
    must_verify: list = Field(default_factory=list)
    blocking_missing_fields: list = Field(default_factory=list)
    non_blocking_gaps: list = Field(default_factory=list)


_SUSPICIOUS_DOMAIN_PATTERN = re.compile(
    r"\.(cc|tk|ml|ga|cf|gq|xyz|top|work|click|link|pw|buzz)(/|$)",
    re.I,
)
_RANDOM_DOMAIN_PATTERN = re.compile(r"^[a-z0-9]{10,}$", re.I)

_OFFICIAL_SOURCE_RULES = [
    ("12306.cn", "official_transport"),
    ("www.12306.cn", "official_transport"),
    ("gugong.net", "official_attraction"),
    ("www.dpm.org.cn", "official_attraction"),
    ("dpm.org.cn", "official_attraction"),
    ("yuyue.tamgw.beijing.gov.cn", "official_attraction"),
    ("tamgw.beijing.gov.cn", "official_attraction"),
    ("beijing.gov.cn", "government"),
    ("gov.cn", "government"),
    ("amap.com", "official_map"),
    ("gaode.com", "official_map"),
]


def is_suspicious_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return True
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc or ""
        host = host.split(":")[0].lower()
        if not host:
            return True
        if _SUSPICIOUS_DOMAIN_PATTERN.search(host):
            return True
        parts = host.rsplit(".", 2)
        name = parts[0] if parts else ""
        if len(name) >= 10 and _RANDOM_DOMAIN_PATTERN.match(name):
            return True
        return False
    except Exception:
        return False


def classify_source(url: str) -> Dict[str, Any]:
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).netloc or "").split(":")[0].lower()
    except Exception:
        host = ""

    for domain, source_type in _OFFICIAL_SOURCE_RULES:
        if host == domain or host.endswith("." + domain):
            return {"source_type": source_type, "trust_level": "high", "official": True}
    return {"source_type": "web", "trust_level": "medium", "official": False}


def requires_official_source(query: str) -> bool:
    q = query or ""
    hard_words = (
        "高铁",
        "火车",
        "12306",
        "车次",
        "余票",
        "票价",
        "预约",
        "故宫",
        "天安门",
        "酒店",
        "门票",
    )
    return any(word in q for word in hard_words)
