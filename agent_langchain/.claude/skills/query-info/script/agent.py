"""TravelFlow 信息检索智能体。

优先使用高德开放平台 API 提供 POI、天气、地理编码和路径规划能力；
通用文本查询保留 DDGS 作为兜底搜索。
"""
from __future__ import annotations

from typing import Optional, Union, List, Dict, Any
import asyncio
import importlib
import json
import logging
import re
import sys
import os
from datetime import date, timedelta

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)
from utils.amap_service import AmapService
from utils.langchain_runtime import ainvoke_text
from utils.train_service import TrainService

logger = logging.getLogger(__name__)

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")


class SummaryOutput(BaseModel):
    summary: str = Field(default="")

# 尝试导入 duckduckgo_search (旧包名) 或 ddgs (新包名)
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("ddgs not installed. Install with: pip install ddgs")

# 疑似垃圾/低质域名：多为 SEO 或不良站，不展示给用户
_SUSPICIOUS_DOMAIN_PATTERN = re.compile(
    r"\.(cc|tk|ml|ga|cf|gq|xyz|top|work|click|link|pw|buzz)(/|$)",
    re.I
)
# 域名主体若为长随机字母（无明显词），则过滤
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


def _is_suspicious_url(url: str) -> bool:
    """过滤疑似垃圾/不良站点（如部分 .cc/.tk 等易被滥用的域名）。"""
    if not url or not url.startswith("http"):
        return True
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc or ""
        # 去掉端口
        host = host.split(":")[0].lower()
        if not host:
            return True
        # 可疑 TLD
        if _SUSPICIOUS_DOMAIN_PATTERN.search(host):
            return True
        # 主域名部分（最后一个 . 之前若还有多段则取倒数第二段之前）
        parts = host.rsplit(".", 2)
        name = parts[0] if parts else ""
        if len(name) >= 10 and _RANDOM_DOMAIN_PATTERN.match(name):
            return True
        return False
    except Exception:
        return False


def _classify_source(url: str) -> Dict[str, Any]:
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).netloc or "").split(":")[0].lower()
    except Exception:
        host = ""

    for domain, source_type in _OFFICIAL_SOURCE_RULES:
        if host == domain or host.endswith("." + domain):
            return {"source_type": source_type, "trust_level": "high", "official": True}
    return {"source_type": "web", "trust_level": "medium", "official": False}


def _requires_official_source(query: str) -> bool:
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


class InformationQueryAgent:
    """
    信息查询智能体（真实检索版）

    核心功能：
    - 行程相关检索：高德 POI、天气、地理编码、路径规划
    - 通用文本查询：DDGS 兜底搜索
    """

    def __init__(self, name: str = "InformationQueryAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context: Dict[str, Any] = state.get("context", {})
        user_query = context.get("rewritten_query", "") or state.get("user_query", "")
        event_data = context.get("event_data") or None
        refinement_requests = context.get("plan_search_requests") or []

        if self._is_train_query(user_query, event_data):
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

        # ---------- 行程规划场景：基于事件信息调用高德 API ----------
        if isinstance(event_data, dict):
            if event_data.get("destination"):
                try:
                    logger.info("Trip info query using Amap for destination: %s", event_data.get("destination"))
                    trip_result = await self._trip_info_query(event_data, user_query, refinement_requests)
                    return trip_result
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

        # 天气类问题优先走结构化天气接口，避免通用搜索返回低质结果
        if self._is_weather_query(user_query):
            logger.info(f"Weather query: {user_query}")
            try:
                result = await self._weather_query(user_query)
                return result
            except Exception as e:
                logger.warning(f"Amap MCP weather query failed: {e}")
                return {
                    "query_type": "天气查询",
                    "query_success": False,
                    "results": {"message": f"高德 MCP 天气查询失败: {e}"},
                }
        else:
            result = None

        if result is None:
            logger.info(f"Web search query: {user_query}")
            try:
                result = await self._web_search(user_query)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                result = {
                    "query_type": "网络搜索",
                    "query_success": False,
                    "results": {"error": str(e)},
                }

        return result

    def _is_weather_query(self, query: str) -> bool:
        """简单判断是否为天气类问题。"""
        q = (query or "").strip()
        if not q:
            return False
        return "天气" in q or "气温" in q or "下雨" in q or "预报" in q

    def _is_train_query(self, query: str, event_data: Any = None) -> bool:
        """判断是否为 12306 车次/票价/余票类问题。"""
        q = (query or "").strip()
        train_words = ("火车", "高铁", "动车", "城际", "车次", "12306", "余票", "票价", "火车票", "高铁票")
        if any(word in q for word in train_words):
            return True
        if isinstance(event_data, dict):
            transport = str(event_data.get("transportation") or event_data.get("transportation_preference") or "")
            return any(word in transport for word in train_words)
        return False

    async def _train_query(self, query: str, event_data: Any = None) -> Dict[str, Any]:
        params = self._extract_train_query_params(query, event_data)
        missing = [key for key in ("from_station", "to_station", "date") if not params.get(key)]
        if missing:
            return {
                "query_type": "火车车次查询",
                "query_success": False,
                "results": {
                    "message": "查询高铁/火车车次需要出发地、目的地和日期。你可以这样问：明天上海虹桥到杭州东的高铁有哪些？",
                    "missing_info": missing,
                    "parsed_params": params,
                },
            }

        train_filter_flags = self._train_filter_flags(query)
        sort_flag = self._train_sort_flag(query)
        service = TrainService()
        raw = await service.get_tickets(
            date=params["date"],
            from_station=params["from_station"],
            to_station=params["to_station"],
            train_filter_flags=train_filter_flags,
            sort_flag=sort_flag,
            sort_reverse=False,
            limited_num=10,
        )
        tickets = self._normalize_train_tickets(raw)
        summary = self._format_train_ticket_summary(params, tickets, raw)
        return {
            "query_type": "火车车次查询",
            "query_success": True,
            "verified": True,
            "requires_official_source": True,
            "trust_level": "high",
            "results": {
                "summary": summary,
                "tickets": tickets,
                "raw": raw,
                "parsed_params": params,
                "sources": [
                    {
                        "title": "12306 MCP get-tickets",
                        "url": "https://github.com/Joooook/12306-mcp",
                        "source_type": "official_transport",
                        "trust_level": "high",
                        "official": True,
                    }
                ],
            },
        }

    def _extract_train_query_params(self, query: str, event_data: Any = None) -> Dict[str, str]:
        q = (query or "").strip()
        station_query = self._strip_train_date_words(q)
        params = {
            "from_station": "",
            "to_station": "",
            "date": self._extract_train_date(q),
        }
        if isinstance(event_data, dict):
            params["from_station"] = str(event_data.get("origin") or "").strip()
            params["to_station"] = str(event_data.get("destination") or "").strip()
            params["date"] = str(event_data.get("start_date") or params["date"] or "").strip()

        station_match = re.search(
            r"(?:从)?(?P<from>[\u4e00-\u9fa5A-Za-z0-9]+?)(?:站)?(?:出发)?(?:到|去|至|前往)(?P<to>[\u4e00-\u9fa5A-Za-z0-9]+?)(?:站)?(?:的|高铁|动车|火车|车次|票|$)",
            station_query,
        )
        if station_match:
            params["from_station"] = self._clean_station_name(station_match.group("from"))
            params["to_station"] = self._clean_station_name(station_match.group("to"))

        return params

    def _strip_train_date_words(self, query: str) -> str:
        cleaned = re.sub(r"20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?", "", query or "")
        cleaned = re.sub(r"\d{1,2}月\d{1,2}日?", "", cleaned)
        cleaned = re.sub(r"(今天|明天|后天|上午|下午|晚上|早上|中午)", "", cleaned)
        return cleaned.strip()

    def _extract_train_date(self, query: str) -> str:
        q = query or ""
        today = date.today()
        if "后天" in q:
            return (today + timedelta(days=2)).isoformat()
        if "明天" in q:
            return (today + timedelta(days=1)).isoformat()
        if "今天" in q:
            return today.isoformat()

        match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", q)
        if match:
            y, m, d = [int(part) for part in match.groups()]
            return date(y, m, d).isoformat()

        match = re.search(r"(\d{1,2})月(\d{1,2})日?", q)
        if match:
            m, d = [int(part) for part in match.groups()]
            candidate = date(today.year, m, d)
            if candidate < today:
                candidate = date(today.year + 1, m, d)
            return candidate.isoformat()
        return ""

    def _clean_station_name(self, value: str) -> str:
        cleaned = re.sub(r"(查询|查|帮我|请问|有没有|哪些|什么|多少|票价|余票|时间|时刻表)", "", value or "")
        return cleaned.strip(" 的，,。？?：:")

    def _train_filter_flags(self, query: str) -> str:
        flags = []
        q = query or ""
        if "高铁" in q or "G字头" in q:
            flags.append("G")
        if "动车" in q or "D字头" in q:
            flags.append("D")
        if "城际" in q or "C字头" in q:
            flags.append("C")
        return "".join(flags)

    def _train_sort_flag(self, query: str) -> str:
        q = query or ""
        if any(word in q for word in ("最快", "耗时最短", "时间最短")):
            return "duration"
        if any(word in q for word in ("最早", "早上", "上午")):
            return "startTime"
        if any(word in q for word in ("最便宜", "便宜", "价格")):
            return "price"
        return ""

    def _normalize_train_tickets(self, raw: Any) -> List[Dict[str, Any]]:
        candidates = raw
        if isinstance(raw, dict):
            candidates = raw.get("tickets") or raw.get("data") or raw.get("results") or raw.get("list") or []
        if isinstance(candidates, dict):
            candidates = candidates.get("tickets") or candidates.get("list") or []
        if not isinstance(candidates, list):
            return []
        return [item for item in candidates if isinstance(item, dict)]

    def _format_train_ticket_summary(self, params: Dict[str, str], tickets: List[Dict[str, Any]], raw: Any) -> str:
        route = f"{params.get('date')} {params.get('from_station')} → {params.get('to_station')}"
        if not tickets:
            if isinstance(raw, str) and raw.strip():
                return f"12306 查询结果：{raw.strip()}"
            return f"没有查到 {route} 的可用车次。"

        lines = [f"12306 查询到 {route} 的车次如下（最多显示前 10 条）："]
        for ticket in tickets[:10]:
            train_no = self._pick_ticket_value(ticket, "train_code", "trainNo", "station_train_code", "车次") or "未知车次"
            from_station = self._pick_ticket_value(ticket, "from_station", "fromStation", "from_station_name", "出发站") or params.get("from_station")
            to_station = self._pick_ticket_value(ticket, "to_station", "toStation", "to_station_name", "到达站") or params.get("to_station")
            start_time = self._pick_ticket_value(ticket, "start_time", "startTime", "出发时间") or ""
            arrive_time = self._pick_ticket_value(ticket, "arrive_time", "arriveTime", "arrival_time", "到达时间") or ""
            duration = self._pick_ticket_value(ticket, "duration", "lishi", "历时") or ""
            price = self._pick_ticket_value(ticket, "price", "min_price", "最低票价", "票价") or ""
            seat_text = self._format_train_seats(ticket)
            parts = [str(train_no), f"{from_station}-{to_station}"]
            if start_time or arrive_time:
                parts.append(f"{start_time}-{arrive_time}".strip("-"))
            if duration:
                parts.append(f"历时{duration}")
            if price:
                parts.append(f"参考票价{price}")
            if seat_text:
                parts.append(seat_text)
            lines.append("；".join(parts))
        return "\n".join(lines)

    def _pick_ticket_value(self, ticket: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = ticket.get(key)
            if value not in (None, "", "--"):
                return value
        return ""

    def _format_train_seats(self, ticket: Dict[str, Any]) -> str:
        seat_keys = [
            ("商务座", ("business_seat", "businessSeat", "swz_num")),
            ("一等座", ("first_class_seat", "firstClassSeat", "zy_num")),
            ("二等座", ("second_class_seat", "secondClassSeat", "ze_num")),
            ("硬卧", ("hard_sleeper", "hardSleeper", "yw_num")),
            ("硬座", ("hard_seat", "hardSeat", "yz_num")),
            ("无座", ("no_seat", "noSeat", "wz_num")),
        ]
        parts = []
        for label, keys in seat_keys:
            value = self._pick_ticket_value(ticket, *keys)
            if value:
                parts.append(f"{label}{value}")
        return "余票：" + "，".join(parts[:4]) if parts else ""

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
            amap = AmapService()
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

    async def _web_search(self, query: str) -> Dict[str, Any]:
        """
        网络搜索 - 使用 DDGS（Dux Distributed Global Search），开启 safesearch，过滤可疑来源。

        Args:
            query: 用户查询

        Returns:
            搜索结果
        """
        if not DDGS_AVAILABLE:
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {
                    "message": "搜索库未安装",
                    "note": "请运行：pip install ddgs",
                },
            }

        try:
            ddgs = DDGS()
            # 开启安全搜索，优先 bing 后端（质量更稳定），多取几条再过滤
            search_results = []
            for backend in ("bing", "duckduckgo", "auto"):
                try:
                    raw = ddgs.text(
                        query,
                        max_results=10,
                        safesearch="on",
                        region="cn-zh",
                        backend=backend,
                    )
                    search_results = list(raw)
                    if search_results:
                        break
                except Exception as e:
                    logger.debug(f"DDGS backend {backend} failed: {e}")
                    continue

            results = []
            for result in search_results:
                href = result.get("href", "")
                if _is_suspicious_url(href):
                    continue
                source_meta = _classify_source(href)
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": href,
                    **source_meta,
                })
                if len(results) >= 5:
                    break

            if not results:
                return {
                    "query_type": "网络搜索",
                    "query_success": False,
                    "results": {"message": "未找到相关结果"},
                }

            # 使用 LLM 总结搜索结果
            summary = await self._summarize_search_results(query, results)
            official_sources = [item for item in results if item.get("official")]
            requires_official = _requires_official_source(query)

            return {
                "query_type": "网络搜索",
                "query_success": True,
                "verified": bool(official_sources) if requires_official else True,
                "requires_official_source": requires_official,
                "trust_level": "high" if official_sources else "medium",
                "results": {
                    "summary": summary,
                    "sources": results,
                    "official_sources": official_sources,
                    "verification_note": (
                        "已找到官方/准官方来源。"
                        if official_sources
                        else "未找到官方来源；该结果只能作为普通参考，不能用于确认车次、余票、票价、预约或酒店库存。"
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {"error": f"搜索失败: {str(e)}"},
            }

    async def _trip_info_query(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """针对行程规划场景的专用查询：使用高德获取 POI、天气、地理编码与路线。

        event_data 通常来自 EventCollectionAgent，字段包括：
        - origin
        - destination
        - start_date / end_date
        - duration_days 等
        """

        destination = (event_data.get("destination") or "").strip()
        if not destination:
            raise ValueError("event_data 中缺少 destination 字段，无法进行行程信息查询")

        amap = AmapService()

        # 1) 地理编码：目的地地址转经纬度
        geocodes = await amap.maps_geo(address=destination)

        # 2) 目的地 POI：默认用“景点”关键字获取一批标志性地点。
        # 若 plan 要求补充检索，则追加关键词查询并合并去重。
        search_keywords = ["景点", "博物馆", "公园", "美食 餐厅", "经济型酒店", "火车站 高铁站", "机场"]
        web_search_requests: List[Dict[str, Any]] = []
        for request in refinement_requests or []:
            if not isinstance(request, dict):
                continue
            keywords = str(request.get("keywords") or request.get("query") or "").strip()
            if not keywords:
                continue
            web_search_requests.append(request)
            if keywords and keywords not in search_keywords and "天气" not in keywords:
                search_keywords.append(keywords)

        pois: List[Dict[str, Any]] = []
        pois_by_category: Dict[str, List[Dict[str, Any]]] = {}
        seen_poi_ids = set()

        async def query_keyword(keyword: str) -> tuple[str, List[Dict[str, Any]]]:
            try:
                return keyword, await amap.maps_text_search(city=destination, keywords=keyword)
            except Exception as e:
                logger.warning("Amap POI query failed for %s/%s: %s", destination, keyword, e)
                return keyword, []

        poi_query_results = await asyncio.gather(
            *[query_keyword(keyword) for keyword in search_keywords[:8]]
        )
        for keywords, queried in poi_query_results:
            category_items: List[Dict[str, Any]] = []
            for poi in queried:
                poi_key = poi.get("id") or poi.get("name")
                if not poi_key:
                    continue
                poi["source_keywords"] = keywords
                category_items.append(poi)
                if poi_key in seen_poi_ids:
                    continue
                seen_poi_ids.add(poi_key)
                pois.append(poi)
            pois_by_category[keywords] = category_items[:10]

        # 3) 目的地天气：使用高德天气接口，保留原始结构，方便上游 LLM 使用
        try:
            weather = await amap.maps_weather(city=destination)
        except Exception as e:
            logger.warning("Amap weather query failed for %s: %s", destination, e)
            weather = {"error": str(e)}

        supplemental_search: List[Dict[str, Any]] = []
        for request in web_search_requests[:4]:
            keywords = str(request.get("keywords") or request.get("query") or "").strip()
            if not keywords or keywords in {"景点", "天气"}:
                continue
            query = f"{destination} {keywords}"
            try:
                web_result = await self._web_search(query)
            except Exception as e:
                web_result = {
                    "query_type": "补充检索",
                    "query_success": False,
                    "results": {"error": str(e)},
                }
            supplemental_search.append(
                {
                    "request": request,
                    "query": query,
                    "result": web_result,
                    "verified": bool(web_result.get("verified")) if isinstance(web_result, dict) else False,
                    "trust_level": web_result.get("trust_level", "unknown") if isinstance(web_result, dict) else "unknown",
                }
            )

        # 4) 路线与距离：基于首批核心景点和交通枢纽生成动线参考。
        routes: List[Dict[str, Any]] = []
        distances: List[Dict[str, Any]] = []
        route_candidates = self._select_route_candidates(pois_by_category, pois)
        if len(route_candidates) >= 2:
            route_pairs = list(zip(route_candidates, route_candidates[1:]))[:4]

            async def query_route_pair(pair: tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
                start, end = pair
                origin = self._location_str(start)
                dest = self._location_str(end)
                if not origin or not dest:
                    return {}
                try:
                    route = await amap.maps_direction_walking(origin=origin, destination=dest)
                    return {
                        "from": start.get("name"),
                        "to": end.get("name"),
                        "mode": "walking",
                        "route": route,
                    }
                except Exception as e:
                    logger.warning("Amap route query failed for %s -> %s: %s", start.get("name"), end.get("name"), e)
                    return {}

            routes = [item for item in await asyncio.gather(*[query_route_pair(pair) for pair in route_pairs]) if item]

            destination_point = self._location_str(route_candidates[0])
            origin_points = [self._location_str(item) for item in route_candidates[1:6]]
            origin_points = [item for item in origin_points if item]
            if destination_point and origin_points:
                try:
                    distance_raw = await amap.maps_distance(
                        origins="|".join(origin_points),
                        destination=destination_point,
                        type="1",
                    )
                    distances.append(
                        {
                            "to": route_candidates[0].get("name"),
                            "from": [item.get("name") for item in route_candidates[1:6]],
                            "mode": "driving_distance",
                            "raw": distance_raw,
                        }
                    )
                except Exception as e:
                    logger.warning("Amap distance query failed for %s: %s", destination, e)

        return {
            "query_type": "行程相关信息查询",
            "query_success": True,
            "results": {
                "destination": destination,
                "event_data": event_data,
                "geocodes": geocodes,
                "pois": pois,
                "pois_by_category": pois_by_category,
                "routes": routes,
                "distances": distances,
                "weather": weather,
                "refinement_requests": refinement_requests or [],
                "search_keywords": search_keywords,
                "supplemental_search": supplemental_search,
                "search_trust_policy": {
                    "hard_constraints_require_official_sources": True,
                    "unverified_hard_constraints_must_not_be_written_as_confirmed": True,
                },
            },
        }

    def _select_route_candidates(
        self,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category in ("火车站 高铁站", "机场", "景点", "博物馆", "公园", "美食 餐厅", "经济型酒店"):
            for poi in pois_by_category.get(category, [])[:2]:
                if poi.get("name") and self._location_str(poi):
                    selected.append(poi)
        if not selected:
            selected = [poi for poi in pois if poi.get("name") and self._location_str(poi)]
        seen = set()
        unique = []
        for poi in selected:
            key = poi.get("id") or poi.get("name")
            if key in seen:
                continue
            seen.add(key)
            unique.append(poi)
        return unique[:8]

    def _location_str(self, poi: Dict[str, Any]) -> str:
        loc = poi.get("location") if isinstance(poi, dict) else None
        if not isinstance(loc, dict):
            return ""
        lng = loc.get("longitude")
        lat = loc.get("latitude")
        if lng is None or lat is None:
            return ""
        return f"{lng},{lat}"

    async def _summarize_search_results(self, query: str, results: List[Dict]) -> str:
        """
        使用 LLM 总结搜索结果

        Args:
            query: 用户查询
            results: 搜索结果列表

        Returns:
            总结文本
        """
        if not results:
            return "未找到相关信息"

        # 构建搜索结果文本
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result['title']}\n{result['snippet']}\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("query-info")
        if not skill_instruction:
            skill_instruction = "请直接回答用户的问题，保持简洁。"

        prompt = f"""根据以下搜索结果，简洁地回答用户的问题。

【当前时间】
{current_date} {weekday}
（用户查询中的相对时间请基于此日期理解，如"明天"、"2月28日"等）

【用户问题】
{query}

【搜索结果】
{results_text}

【任务说明】
{skill_instruction}
"""

        try:
            summarized = await self._invoke_summary(prompt)
            return summarized.summary.strip() if summarized.summary else "无法生成摘要"
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "搜索成功，但摘要生成失败"

    async def _invoke_summary(self, prompt: str) -> SummaryOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(SummaryOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, SummaryOutput):
                    return result
                if isinstance(result, dict):
                    return SummaryOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        return SummaryOutput(summary=str(text).strip())
