"""TravelFlow 信息检索智能体。

对外仍是单一 ``search`` 智能体；内部按天气、铁路、网页兜底、
行程检索规划和执行拆分为 search_modules，便于测试和演进。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from utils.amap_service import AmapService
from utils.train_service import TrainService

from agents.search_modules.common import (
    SearchPlanOutput,
    SummaryOutput,
    classify_source as _classify_source,
    is_suspicious_url as _is_suspicious_url,
    requires_official_source as _requires_official_source,
)
from agents.search_modules.rail import RailSearchMixin
from agents.search_modules.trip_execution import TripSearchExecutionMixin
from agents.search_modules.trip_planner import TripSearchPlannerMixin
from agents.search_modules.weather import WeatherSearchMixin
from agents.search_modules.web import WebFallbackSearchMixin

logger = logging.getLogger(__name__)

try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    DDGS_AVAILABLE = False
    logger.warning("ddgs not installed. Install with: pip install ddgs")


class InformationQueryAgent(
    RailSearchMixin,
    WeatherSearchMixin,
    WebFallbackSearchMixin,
    TripSearchPlannerMixin,
    TripSearchExecutionMixin,
):
    """信息查询智能体（真实检索版）。"""

    def __init__(self, name: str = "InformationQueryAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader

        self.skill_loader = SkillLoader()

    def _new_amap_service(self):
        return AmapService()

    def _new_train_service(self):
        return TrainService()

    def _ddgs_available(self) -> bool:
        return bool(DDGS_AVAILABLE)

    def _new_ddgs(self):
        if DDGS is None:
            raise ImportError("ddgs not installed")
        return DDGS()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context: Dict[str, Any] = state.get("context", {})
        user_query = context.get("rewritten_query", "") or state.get("user_query", "")
        event_data = context.get("event_data") or None
        refinement_requests = context.get("plan_search_requests") or []

        if self._is_train_query(user_query, event_data) and not isinstance(event_data, dict):
            logger.info("Train query: %s", user_query)
            try:
                return await self._train_query(user_query, event_data)
            except Exception as e:
                logger.warning("12306 MCP train query failed: %s", e)
                return {
                    "query_type": "火车车次查询",
                    "query_success": False,
                    "results": {"message": f"12306 MCP 查询失败: {e}"},
                }

        if isinstance(event_data, dict):
            if event_data.get("destination"):
                try:
                    logger.info("Trip info query using Amap for destination: %s", event_data.get("destination"))
                    return await self._trip_info_query(event_data, user_query, refinement_requests)
                except Exception as e:
                    logger.warning("Trip info query via Amap failed: %s", e)
                    return {
                        "error": f"高德 MCP 行程信息查询失败: {e}",
                        "query_type": "行程相关信息查询",
                        "query_success": False,
                        "results": {"message": f"高德 MCP 行程信息查询失败: {e}"},
                    }
            return {
                "error": "行程信息不完整，缺少目的地，已停止外部检索。",
                "query_type": "行程相关信息查询",
                "query_success": False,
                "results": {
                    "message": "行程信息不完整，缺少目的地，已停止外部检索。",
                    "event_data": event_data,
                },
            }

        if self._is_weather_query(user_query):
            logger.info("Weather query: %s", user_query)
            try:
                return await self._weather_query(user_query)
            except Exception as e:
                logger.warning("Amap MCP weather query failed: %s", e)
                return {
                    "query_type": "天气查询",
                    "query_success": False,
                    "results": {"message": f"高德 MCP 天气查询失败: {e}"},
                }

        logger.info("Web search query: %s", user_query)
        try:
            return await self._web_search(user_query)
        except Exception as e:
            logger.error("Query failed: %s", e)
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {"error": str(e)},
            }


__all__ = [
    "InformationQueryAgent",
    "SearchPlanOutput",
    "SummaryOutput",
    "AmapService",
    "TrainService",
    "DDGS",
    "DDGS_AVAILABLE",
]
