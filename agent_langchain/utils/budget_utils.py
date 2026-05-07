"""Shared budget and lodging-budget detection helpers."""

from __future__ import annotations

import re
from typing import Optional


def detect_budget_level(text: str) -> str | None:
    """Detect the user-facing TravelFlow budget tier from Chinese text."""
    query = text or ""
    if any(word in query for word in ("经济", "省钱", "便宜", "300元以内", "预算低", "平价")):
        return "经济型"
    if any(word in query for word in ("舒适", "性价比", "300到600", "300-600")):
        return "舒适型"
    if any(word in query for word in ("品质", "高端", "豪华", "600元以上", "600以上", "不差钱", "预算充足")):
        return "品质型"

    budget_amount = re.search(r"预算[^0-9]*(\d+)\s*元?(?:以内|以下|内)?", query)
    if budget_amount:
        amount = int(budget_amount.group(1))
        if amount <= 300:
            return "经济型"
        if amount <= 600:
            return "舒适型"
        return "品质型"
    return None


def detect_lodging_budget(text: str) -> tuple[int | None, int | None]:
    """Extract a lodging budget range as ``(min, max)`` when present."""
    query = text or ""
    range_match = re.search(
        r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*(?:到|至|-|~|－|—)\s*(\d+)\s*元?",
        query,
    )
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return min(low, high), max(low, high)

    max_match = re.search(r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*元?(?:以内|以下|内)", query)
    if max_match:
        return None, int(max_match.group(1))

    min_match = re.search(r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*元?(?:以上|起)", query)
    if min_match:
        return int(min_match.group(1)), None

    return None, None


def infer_budget_profile(budget_level: str) -> str:
    """Map a budget level to the trip-search demand-profile tag."""
    if budget_level == "经济型":
        return "budget"
    if budget_level in {"舒适型", "品质型"}:
        return "premium"
    return ""
