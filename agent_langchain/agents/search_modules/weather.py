from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List

from utils.json_parser import robust_json_parse
from utils.langchain_runtime import ainvoke_text
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

from .common import (
    SearchPlanOutput,
    SummaryOutput,
    classify_source,
    is_suspicious_url,
    requires_official_source,
)

logger = logging.getLogger(__name__)

class WeatherSearchMixin:
    def _is_weather_query(self, query: str) -> bool:
        """简单判断是否为天气类问题。"""
        q = (query or "").strip()
        if not q:
            return False
        return "天气" in q or "气温" in q or "下雨" in q or "预报" in q

    def _extract_location_query(self, query: str) -> str:
        """提取用于位置搜索的关键词：优先返回城市命中，否则清洗掉“天气/明天”等词后返回原句。"""
        q = (query or "").strip()
        city = self._extract_city_from_query(q)
        if city:
            return city
        cleaned = re.sub(r"(天气|气温|预报|明天|后天|今天|现在|的|呢|吗)", "", q)
        cleaned = cleaned.strip()
        return cleaned or q

    def _normalize_location_for_api(self, location_query: str) -> str:
        """
        将位置查询词标准化，避免把整句传给 API。
        优先取最长的连续中文词段（2~12字），否则截断到前 16 字符。
        """
        q = (location_query or "").strip()
        segments = re.findall(r"[\u4e00-\u9fa5]{2,12}", q)
        if segments:
            return max(segments, key=len)[:12]
        return q[:16]

    async def _weather_query(self, query: str) -> Dict[str, Any]:
        """天气查询：只使用高德 MCP maps_weather。"""

        location_query = self._extract_location_query(query)
        if not location_query:
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": "未识别到地点，请说明具体城市/区县，如：成都郫都区明天的天气？"},
            }

        try:
            amap = self._new_amap_service()
            raw = await amap.maps_weather(city=location_query)
        except Exception as e:
            logger.warning("Amap MCP weather query failed for %s: %s", location_query, e)
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": f"高德 MCP 天气查询失败: {e}"},
            }

        summary = self._format_amap_weather(location_query, raw)
        return {
            "query_type": "天气查询",
            "query_success": True,
            "results": {
                "location_name": location_query,
                "summary": summary,
                "raw": raw,
                "sources": [{"title": "Amap MCP maps_weather"}],
            },
        }

    def _format_amap_weather(self, city: str, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict):
            return f"{city}天气查询成功。"

        lives = raw.get("lives") or raw.get("LiveWeather") or []
        forecasts = raw.get("forecasts") or raw.get("Forecasts") or []
        if lives and isinstance(lives, list):
            item = lives[0] if lives else {}
            weather = item.get("weather") or item.get("Weather") or ""
            temp = item.get("temperature") or item.get("Temperature") or ""
            humidity = item.get("humidity") or ""
            wind = item.get("winddirection") or item.get("windpower") or ""
            parts = [f"{city}当前天气"]
            if weather:
                parts.append(str(weather))
            if temp:
                parts.append(f"气温{temp}°C")
            if humidity:
                parts.append(f"湿度{humidity}%")
            if wind:
                parts.append(f"风况{wind}")
            return "，".join(parts) + "。"

        if forecasts and isinstance(forecasts, list):
            if forecasts and isinstance(forecasts[0], dict) and forecasts[0].get("date"):
                lines = []
                for cast in forecasts[:5]:
                    date = cast.get("date", "")
                    day_weather = cast.get("dayweather", "")
                    night_weather = cast.get("nightweather", "")
                    day_temp = cast.get("daytemp", "")
                    night_temp = cast.get("nighttemp", "")
                    lines.append(f"{date}: 白天{day_weather}，夜间{night_weather}，{night_temp}~{day_temp}°C")
                return f"{city}天气预报：" + "；".join(lines)

            forecast = forecasts[0] if forecasts else {}
            casts = forecast.get("casts") or []
            lines = []
            for cast in casts[:5]:
                date = cast.get("date", "")
                day_weather = cast.get("dayweather", "")
                night_weather = cast.get("nightweather", "")
                day_temp = cast.get("daytemp", "")
                night_temp = cast.get("nighttemp", "")
                lines.append(f"{date}: 白天{day_weather}，夜间{night_weather}，{night_temp}~{day_temp}°C")
            if lines:
                return f"{city}天气预报：" + "；".join(lines)

        return f"{city}天气查询成功：{json.dumps(raw, ensure_ascii=False)}"

    def _extract_city_from_query(self, query: str) -> str:
        """从问题中提取城市名（简单实现：常见城市列表匹配）。"""
        common_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "苏州",
            "天津", "重庆", "厦门", "青岛", "大连", "宁波", "无锡", "长沙", "郑州", "济南",
            "哈尔滨", "沈阳", "昆明", "合肥", "福州", "石家庄", "南昌", "贵阳", "太原", "南宁",
        ]
        q = (query or "").strip()
        for city in common_cities:
            if city in q:
                return city
        # 未匹配到常见城市则返回空，避免把整句话当城市导致乱码
        return ""

