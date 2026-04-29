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
from utils.json_parser import robust_json_parse
from utils.langchain_runtime import ainvoke_text
from utils.train_service import TrainService

logger = logging.getLogger(__name__)

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
            response_format="json",
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
        if isinstance(raw, str):
            return self._parse_train_ticket_text(raw)
        candidates = raw
        if isinstance(raw, dict):
            candidates = raw.get("tickets") or raw.get("data") or raw.get("results") or raw.get("list") or []
        if isinstance(candidates, dict):
            candidates = candidates.get("tickets") or candidates.get("list") or []
        if not isinstance(candidates, list):
            return []
        return [self._normalize_train_ticket_item(item) for item in candidates if isinstance(item, dict)]

    def _normalize_train_ticket_item(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(ticket)
        normalized["train_code"] = (
            self._pick_ticket_value(ticket, "train_code", "trainNo", "station_train_code", "start_train_code", "车次")
            or ""
        )
        normalized["from_station"] = (
            self._pick_ticket_value(ticket, "from_station", "fromStation", "from_station_name", "出发站")
            or ""
        )
        normalized["to_station"] = (
            self._pick_ticket_value(ticket, "to_station", "toStation", "to_station_name", "到达站")
            or ""
        )
        normalized["from_station_code"] = (
            self._pick_ticket_value(ticket, "from_station_code", "from_station_telecode", "fromStationCode")
            or ""
        )
        normalized["to_station_code"] = (
            self._pick_ticket_value(ticket, "to_station_code", "to_station_telecode", "toStationCode")
            or ""
        )
        normalized["start_time"] = self._pick_ticket_value(ticket, "start_time", "startTime", "出发时间") or ""
        normalized["arrive_time"] = (
            self._pick_ticket_value(ticket, "arrive_time", "arriveTime", "arrival_time", "到达时间")
            or ""
        )
        normalized["duration"] = self._pick_ticket_value(ticket, "duration", "lishi", "历时") or ""

        prices = ticket.get("prices")
        if isinstance(prices, list):
            seats = []
            for price in prices:
                if not isinstance(price, dict):
                    continue
                seat_type = str(price.get("seat_name") or price.get("seat_type") or "").strip()
                if not seat_type:
                    continue
                availability = self._format_train_ticket_status(price.get("num"))
                price_value = price.get("price")
                price_text = "" if price_value in (None, "") else f"{price_value}元"
                seat = {
                    "seat_type": seat_type,
                    "availability": availability,
                    "price": price_text,
                }
                seats.append(seat)
                self._apply_named_seat_fields(normalized, seat)
            if seats:
                normalized["seats"] = seats

        return normalized

    def _format_train_ticket_status(self, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if not text or text in {"--", "无"}:
            return "无票"
        if text in {"有", "充足"}:
            return "有票"
        if text == "候补":
            return "无票需候补"
        if text.isdigit():
            count = int(text)
            return "无票" if count == 0 else f"剩余{count}张票"
        return text if text.endswith("票") else f"{text}票"

    def _parse_train_ticket_text(self, text: str) -> List[Dict[str, Any]]:
        tickets: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None
        train_line_pattern = re.compile(
            r"^(?P<train>[A-Z]\d+)\s+"
            r"(?P<from>.+?)\(telecode:(?P<from_code>[A-Z]+)\)\s*->\s*"
            r"(?P<to>.+?)\(telecode:(?P<to_code>[A-Z]+)\)\s+"
            r"(?P<start>\d{2}:\d{2})\s*->\s*(?P<arrive>\d{2}:\d{2})\s+历时：(?P<duration>\d{2}:\d{2})"
        )
        seat_pattern = re.compile(r"^-\s*(?P<seat>[^:：]+)[:：]\s*(?P<availability>.+?)\s+(?P<price>\d+(?:\.\d+)?)元")

        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("车次|"):
                continue
            train_match = train_line_pattern.match(line)
            if train_match:
                if current:
                    tickets.append(current)
                current = {
                    "train_code": train_match.group("train"),
                    "from_station": train_match.group("from").strip(),
                    "from_station_code": train_match.group("from_code"),
                    "to_station": train_match.group("to").strip(),
                    "to_station_code": train_match.group("to_code"),
                    "start_time": train_match.group("start"),
                    "arrive_time": train_match.group("arrive"),
                    "duration": train_match.group("duration"),
                    "seats": [],
                }
                continue

            seat_match = seat_pattern.match(line)
            if seat_match and current is not None:
                seat = {
                    "seat_type": seat_match.group("seat").strip(),
                    "availability": seat_match.group("availability").strip(),
                    "price": f"{seat_match.group('price')}元",
                }
                current["seats"].append(seat)
                self._apply_named_seat_fields(current, seat)

        if current:
            tickets.append(current)
        return tickets

    def _apply_named_seat_fields(self, ticket: Dict[str, Any], seat: Dict[str, str]) -> None:
        seat_type = seat.get("seat_type", "")
        value = f"{seat.get('availability', '')} {seat.get('price', '')}".strip()
        mapping = {
            "商务座": "business_seat",
            "一等座": "first_class_seat",
            "二等座": "second_class_seat",
            "硬卧": "hard_sleeper",
            "硬座": "hard_seat",
            "无座": "no_seat",
        }
        for label, key in mapping.items():
            if label in seat_type:
                ticket[key] = value
                return

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
        seats = ticket.get("seats")
        if isinstance(seats, list) and seats:
            for seat in seats[:4]:
                if not isinstance(seat, dict):
                    continue
                seat_type = seat.get("seat_type")
                availability = seat.get("availability")
                price = seat.get("price")
                if seat_type and availability:
                    parts.append(f"{seat_type}{availability}{f' {price}' if price else ''}")
            return "余票：" + "，".join(parts) if parts else ""

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
        """针对行程规划场景生成检索计划，并调用垂直工具返回结构化素材包。"""

        destination = (event_data.get("destination") or "").strip()
        if not destination:
            raise ValueError("event_data 中缺少 destination 字段，无法进行行程信息查询")

        amap = AmapService()
        search_plan = await self._build_trip_search_plan(event_data, user_query, refinement_requests or [])

        geocodes_task = asyncio.create_task(amap.maps_geo(address=destination))
        weather_task = asyncio.create_task(amap.maps_weather(city=destination))
        poi_tasks = search_plan.get("tasks_by_type", {}).get("poi_search", [])

        async def query_keyword(keyword: str) -> tuple[str, List[Dict[str, Any]]]:
            try:
                return keyword, await amap.maps_text_search(city=destination, keywords=keyword)
            except Exception as e:
                logger.warning("Amap POI query failed for %s/%s: %s", destination, keyword, e)
                return keyword, []

        poi_query_results = await asyncio.gather(
            *[query_keyword(str(task.get("keywords", ""))) for task in poi_tasks[:10]]
        )

        pois: List[Dict[str, Any]] = []
        pois_by_category: Dict[str, List[Dict[str, Any]]] = {}
        seen_poi_ids = set()
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

        try:
            geocodes = await geocodes_task
        except Exception as e:
            logger.warning("Amap geocode query failed for %s: %s", destination, e)
            geocodes = {"error": str(e)}

        try:
            weather = await weather_task
        except Exception as e:
            logger.warning("Amap weather query failed for %s: %s", destination, e)
            weather = {"error": str(e)}

        transport = await self._execute_transport_tasks(search_plan, user_query)
        selected_scenic = self._select_scenic_pois(pois_by_category, pois)
        nearby = await self._query_nearby_trip_support(amap, destination, selected_scenic)
        routes, distances = await self._query_trip_routes(amap, pois_by_category, pois)

        supplemental_search: List[Dict[str, Any]] = []
        for request in search_plan.get("tasks_by_type", {}).get("web_search", [])[:4]:
            keywords = str(request.get("keywords") or request.get("query") or "").strip()
            if not keywords or keywords in {"景点", "天气"}:
                continue
            query = keywords if destination in keywords else f"{destination} {keywords}"
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

        search_bundle = {
            "planning": {
                "search_strategy": search_plan.get("search_strategy", ""),
                "demand_profile": search_plan.get("demand_profile", {}),
                "must_verify": search_plan.get("must_verify", []),
                "blocking_missing_fields": search_plan.get("blocking_missing_fields", []),
                "non_blocking_gaps": search_plan.get("non_blocking_gaps", []),
            },
            "transport": transport,
            "destination": {
                "name": destination,
                "geocodes": geocodes,
                "weather": weather,
                "pois": pois,
                "pois_by_category": pois_by_category,
                "selected_scenic_pois": selected_scenic,
                "nearby": nearby,
                "routes": routes,
                "distances": distances,
            },
            "quality": self._build_search_quality_report(
                event_data,
                transport,
                pois,
                weather,
                nearby,
                routes,
                search_plan,
                supplemental_search,
            ),
            "sources": [
                {"title": "Amap MCP", "source_type": "official_map", "trust_level": "high", "official": True},
                {"title": "12306 MCP", "source_type": "official_transport", "trust_level": "high", "official": True},
            ],
        }

        summary = self._format_trip_search_summary(search_bundle)

        return {
            "query_type": "行程相关信息查询",
            "query_success": True,
            "results": {
                "summary": summary,
                "destination": destination,
                "event_data": event_data,
                "search_plan": search_plan.get("tasks", []),
                "search_strategy": search_plan.get("search_strategy", ""),
                "demand_profile": search_plan.get("demand_profile", {}),
                "must_verify": search_plan.get("must_verify", []),
                "blocking_missing_fields": search_plan.get("blocking_missing_fields", []),
                "non_blocking_gaps": search_plan.get("non_blocking_gaps", []),
                "search_bundle": search_bundle,
                "transport": transport,
                "geocodes": geocodes,
                "pois": pois,
                "pois_by_category": pois_by_category,
                "nearby": nearby,
                "hotels": nearby.get("hotels", []),
                "restaurants": nearby.get("restaurants", []),
                "stations": nearby.get("stations", []),
                "routes": routes,
                "distances": distances,
                "weather": weather,
                "refinement_requests": refinement_requests or [],
                "search_keywords": [task.get("keywords") for task in poi_tasks],
                "supplemental_search": supplemental_search,
                "search_trust_policy": {
                    "hard_constraints_require_official_sources": True,
                    "unverified_hard_constraints_must_not_be_written_as_confirmed": True,
                },
            },
        }

    async def _build_trip_search_plan(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a demand-aware search plan before executing external tools."""

        base_tasks = self._build_base_trip_search_tasks(event_data, user_query, refinement_requests)
        profile = self._infer_trip_search_profile(event_data, user_query, refinement_requests)
        augmented_tasks = self._augment_tasks_for_profile(base_tasks, profile, event_data)
        planner_output = await self._llm_refine_trip_search_plan(
            event_data=event_data,
            user_query=user_query,
            refinement_requests=refinement_requests,
            base_tasks=augmented_tasks,
            profile=profile,
        )

        tasks = self._merge_search_tasks(augmented_tasks, planner_output.get("tasks", []), event_data)
        tasks_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for task in tasks:
            tasks_by_type.setdefault(str(task.get("type")), []).append(task)

        return {
            "event_data": event_data,
            "search_strategy": planner_output.get("search_strategy") or profile.get("search_strategy", ""),
            "demand_profile": {**profile, **(planner_output.get("demand_profile") or {})},
            "tasks": tasks,
            "tasks_by_type": tasks_by_type,
            "must_verify": planner_output.get("must_verify") or profile.get("must_verify", []),
            "blocking_missing_fields": planner_output.get("blocking_missing_fields") or profile.get("blocking_missing_fields", []),
            "non_blocking_gaps": planner_output.get("non_blocking_gaps") or profile.get("non_blocking_gaps", []),
        }

    def _build_base_trip_search_tasks(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        destination = str(event_data.get("destination") or "").strip()
        origin = str(event_data.get("origin") or "").strip()
        return_location = str(event_data.get("return_location") or origin or "").strip()
        start_date = str(event_data.get("start_date") or "").strip()
        end_date = str(event_data.get("end_date") or "").strip()
        if not end_date:
            end_date = self._derive_end_date(start_date, event_data.get("duration_days"))

        tasks: List[Dict[str, Any]] = [
            {
                "type": "geocode",
                "provider": "amap",
                "target": destination,
                "reason": "获取目的地经纬度，支撑周边和动线查询。",
            },
            {
                "type": "weather",
                "provider": "amap",
                "city": destination,
                "reason": "获取目的地天气，支撑每日安排和雨天备选。",
            },
        ]

        poi_keywords = [
            ("scenic", "景点", "核心景点候选"),
            ("museum", "博物馆", "室内文化景点和雨天备选"),
            ("park", "公园", "轻松节奏和户外候选"),
            ("food", "美食 餐厅", "餐饮和商圈候选"),
            ("lodging", "经济型酒店 地铁站", "住宿位置和通勤便利性"),
            ("rail_station", "火车站 高铁站", "到离站交通枢纽"),
            ("airport", "机场", "备用交通枢纽"),
        ]
        for category, keywords, reason in poi_keywords:
            tasks.append(
                {
                    "type": "poi_search",
                    "provider": "amap",
                    "category": category,
                    "city": destination,
                    "keywords": keywords,
                    "reason": reason,
                }
            )

        transport_preference = f"{user_query} {event_data.get('transportation_preference') or ''}"
        train_filter_flags = self._train_filter_flags(transport_preference)
        if origin and destination and start_date and self._should_query_train(transport_preference):
            tasks.append(
                {
                    "type": "train",
                    "provider": "12306",
                    "direction": "outbound",
                    "date": start_date,
                    "from_station": origin,
                    "to_station": destination,
                    "train_filter_flags": train_filter_flags,
                    "time_window": ["07:00", "13:30"],
                    "reason": "去程交通是硬约束，需要查询当天合适出发时间的车次、余票和价格。",
                }
            )
        if destination and return_location and end_date and self._should_query_train(transport_preference):
            tasks.append(
                {
                    "type": "train",
                    "provider": "12306",
                    "direction": "return",
                    "date": end_date,
                    "from_station": destination,
                    "to_station": return_location,
                    "train_filter_flags": train_filter_flags,
                    "time_window": ["14:00", "21:30"],
                    "reason": "返程交通是硬约束，需要查询当天下午/晚上合适回程车次、余票和价格。",
                }
            )

        for request in refinement_requests:
            if not isinstance(request, dict):
                continue
            keywords = str(request.get("keywords") or request.get("query") or "").strip()
            if not keywords:
                continue
            task_type = "poi_search" if any(word in keywords for word in ("景点", "酒店", "餐厅", "火车站", "机场", "商圈")) else "web_search"
            tasks.append(
                {
                    "type": task_type,
                    "provider": "amap" if task_type == "poi_search" else "web",
                    "category": "refinement",
                    "city": destination,
                    "keywords": keywords,
                    "reason": str(request.get("reason") or "规划智能体要求补充检索"),
                    "expected_output": str(request.get("expected_output") or "补充外部旅行信息"),
                }
            )

        return tasks

    def _infer_trip_search_profile(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        q = " ".join(
            [
                user_query or "",
                str(event_data.get("travel_theme") or ""),
                str(event_data.get("companion_type") or ""),
                str(event_data.get("pace_preference") or ""),
                str(event_data.get("budget_level") or ""),
                " ".join(
                    str(item.get("keywords") or item.get("query") or item)
                    for item in refinement_requests
                    if isinstance(item, (dict, str))
                ),
            ]
        )

        focus_tags = []
        keyword_rules = [
            ("family", ("亲子", "孩子", "儿童", "小孩", "带娃")),
            ("senior_friendly", ("老人", "父母", "长辈", "腿脚", "少走路")),
            ("food", ("美食", "吃", "餐厅", "小吃", "夜市", "老字号")),
            ("hidden_gems", ("小众", "人少", "避开人多", "不排队", "冷门")),
            ("photography", ("拍照", "摄影", "打卡", "出片", "夜景")),
            ("culture", ("历史", "文化", "博物馆", "展览", "古迹")),
            ("shopping", ("购物", "商场", "买东西", "伴手礼")),
            ("nightlife", ("夜游", "夜生活", "酒吧", "夜景")),
            ("rain_backup", ("下雨", "雨天", "室内", "天气不好")),
            ("budget", ("省钱", "经济", "便宜", "预算低", "平价")),
            ("premium", ("高端", "品质", "豪华", "舒适", "预算充足")),
            ("accessibility", ("无障碍", "轮椅", "婴儿车", "少走路")),
        ]
        for tag, words in keyword_rules:
            if any(word in q for word in words):
                focus_tags.append(tag)

        must_verify = ["destination_pois", "weather"]
        hard_constraints = []
        if event_data.get("origin") and event_data.get("destination") and event_data.get("start_date"):
            hard_constraints.append("transport_options")
            if self._should_query_train(f"{user_query} {event_data.get('transportation_preference') or ''}"):
                must_verify.append("train_tickets")
        if any(word in q for word in ("开放时间", "营业时间", "门票", "预约", "闭馆", "限流")):
            hard_constraints.extend(["opening_hours", "ticket_or_reservation_policy"])
            must_verify.append("official_attraction_policy")
        if any(word in q for word in ("酒店", "住宿", "住哪", "民宿")):
            hard_constraints.append("lodging_area")
            must_verify.append("lodging_candidates")
        if any(word in q for word in ("路线", "交通", "地铁", "打车", "步行", "动线")):
            hard_constraints.append("route_distance")
            must_verify.append("routes")

        blocking_missing_fields = []
        for field in ("destination", "start_date", "duration_days"):
            if event_data.get(field) in (None, "", []):
                blocking_missing_fields.append(field)

        search_strategy = "先核验硬约束，再按用户偏好补充 POI、周边服务和路线；未核验的门票、预约、库存、余票不作为确定事实。"
        return {
            "focus_tags": focus_tags,
            "hard_constraints": hard_constraints,
            "must_verify": list(dict.fromkeys(must_verify)),
            "blocking_missing_fields": blocking_missing_fields,
            "non_blocking_gaps": [],
            "search_strategy": search_strategy,
            "requires_official_sources": bool(hard_constraints),
        }

    def _augment_tasks_for_profile(
        self,
        tasks: List[Dict[str, Any]],
        profile: Dict[str, Any],
        event_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        destination = str(event_data.get("destination") or "").strip()
        focus_tags = set(profile.get("focus_tags") or [])
        augmented = list(tasks)

        tag_tasks = {
            "family": [
                ("poi_search", "亲子 景点 儿童乐园 科技馆", "根据亲子需求补充适合儿童的景点。"),
                ("web_search", "亲子游 注意事项 预约 门票", "核验亲子出行的预约、门票和年龄限制。"),
            ],
            "senior_friendly": [
                ("poi_search", "公园 博物馆 休闲 景点", "根据长辈同行补充低强度景点。"),
                ("web_search", "老人友好 无障碍 少走路 路线", "补充长辈友好和无障碍信息。"),
            ],
            "food": [
                ("poi_search", "特色美食 老字号 餐厅", "根据美食需求补充餐饮候选。"),
                ("poi_search", "夜市 小吃街", "补充夜间餐饮和小吃街候选。"),
            ],
            "hidden_gems": [
                ("poi_search", "小众景点 人少", "根据小众需求补充游客较少的地点。"),
                ("web_search", "小众路线 避开人多", "补充普通 POI 搜索难覆盖的小众路线信息。"),
            ],
            "photography": [
                ("poi_search", "摄影 打卡 观景台 夜景", "根据拍照需求补充出片地点。"),
            ],
            "culture": [
                ("poi_search", "历史文化 博物馆 展览 古迹", "根据文化需求补充文化类地点。"),
            ],
            "shopping": [
                ("poi_search", "商圈 购物中心 伴手礼", "根据购物需求补充商圈和伴手礼地点。"),
            ],
            "nightlife": [
                ("poi_search", "夜景 夜游 夜市", "根据夜游需求补充夜间活动地点。"),
            ],
            "rain_backup": [
                ("poi_search", "室内景点 博物馆 商场", "根据雨天需求补充室内备选。"),
            ],
            "budget": [
                ("poi_search", "平价餐厅 经济型酒店 青年旅舍", "根据低预算需求补充平价候选。"),
            ],
            "premium": [
                ("poi_search", "高端酒店 精品酒店 黑珍珠 餐厅", "根据品质需求补充高端候选。"),
            ],
            "accessibility": [
                ("web_search", "无障碍 婴儿车 轮椅 交通", "核验无障碍和少步行信息。"),
            ],
        }

        for tag in focus_tags:
            for task_type, keywords, reason in tag_tasks.get(tag, []):
                augmented.append(
                    {
                        "type": task_type,
                        "provider": "amap" if task_type == "poi_search" else "web",
                        "category": tag,
                        "city": destination,
                        "keywords": keywords,
                        "reason": reason,
                        "expected_output": "补充与用户明确需求相关的外部信息。",
                    }
                )

        if "official_attraction_policy" in set(profile.get("must_verify") or []):
            augmented.append(
                {
                    "type": "web_search",
                    "provider": "web",
                    "category": "official_policy",
                    "city": destination,
                    "keywords": "官方 开放时间 门票 预约",
                    "reason": "门票、预约、开放时间属于硬约束，需要优先寻找官方或准官方来源。",
                    "expected_output": "官方开放时间、预约和票务政策。",
                    "requires_official_source": True,
                    "blocking": False,
                }
            )

        return augmented

    async def _llm_refine_trip_search_plan(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]],
        base_tasks: List[Dict[str, Any]],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.model:
            return {}

        compact_tasks = [
            {
                "type": task.get("type"),
                "provider": task.get("provider"),
                "category": task.get("category"),
                "keywords": task.get("keywords") or task.get("target") or task.get("city"),
                "reason": task.get("reason"),
            }
            for task in base_tasks[:18]
        ]
        prompt = f"""你是 TravelFlow 的 Search Planner。请先判断用户需求需要搜什么，再补充少量高价值检索任务。

只返回 JSON，不要写 Markdown。

【用户需求】
{user_query}

【结构化行程信息】
{json.dumps(event_data, ensure_ascii=False)}

【规划智能体追加检索请求】
{json.dumps(refinement_requests, ensure_ascii=False)}

【当前规则画像】
{json.dumps(profile, ensure_ascii=False)}

【已有基础任务】
{json.dumps(compact_tasks, ensure_ascii=False)}

输出 JSON schema：
{{
  "search_strategy": "一句话说明搜索策略",
  "demand_profile": {{
    "focus_tags": ["family|food|hidden_gems|photography|culture|shopping|nightlife|rain_backup|senior_friendly|accessibility|budget|premium"],
    "hard_constraints": ["transport_options|opening_hours|ticket_or_reservation_policy|lodging_area|route_distance"]
  }},
  "tasks": [
    {{
      "type": "poi_search|web_search",
      "provider": "amap|web",
      "category": "短类别名",
      "keywords": "要搜索的关键词，不要超过18个汉字",
      "reason": "为什么需要搜",
      "expected_output": "期望得到什么",
      "requires_official_source": false,
      "blocking": false
    }}
  ],
  "must_verify": ["必须核验的事实字段"],
  "blocking_missing_fields": ["如果缺失就应阻断规划的字段"],
  "non_blocking_gaps": ["可以提示但不阻断的问题"]
}}

规则：
1. 不要重复已有基础任务。
2. 最多新增 6 个任务。
3. 门票、预约、开放时间、余票、票价、酒店库存必须标记 requires_official_source=true。
4. 不确定的内容不要写成已确认事实，只作为检索目标。
"""
        try:
            planner = await self._invoke_search_planner(prompt)
            return {
                "search_strategy": planner.search_strategy,
                "demand_profile": planner.demand_profile or {},
                "tasks": planner.tasks or [],
                "must_verify": planner.must_verify or [],
                "blocking_missing_fields": planner.blocking_missing_fields or [],
                "non_blocking_gaps": planner.non_blocking_gaps or [],
            }
        except Exception as e:
            logger.warning("Search planner refinement failed, using rule-based plan: %s", e)
            return {}

    async def _invoke_search_planner(self, prompt: str) -> SearchPlanOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(SearchPlanOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, SearchPlanOutput):
                    return result
                if isinstance(result, dict):
                    return SearchPlanOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured planner output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured planner output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        parsed = robust_json_parse(text, fallback={})
        if not isinstance(parsed, dict):
            parsed = {}
        return SearchPlanOutput.model_validate(parsed)

    def _merge_search_tasks(
        self,
        base_tasks: List[Dict[str, Any]],
        extra_tasks: List[Dict[str, Any]],
        event_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        destination = str(event_data.get("destination") or "").strip()
        merged: List[Dict[str, Any]] = []
        seen = set()

        def normalize_task(task: Dict[str, Any]) -> Dict[str, Any] | None:
            if not isinstance(task, dict):
                return None
            task_type = str(task.get("type") or "").strip()
            if task_type not in {"geocode", "weather", "poi_search", "train", "web_search"}:
                return None
            normalized = dict(task)
            normalized["type"] = task_type
            if task_type == "poi_search":
                normalized["provider"] = "amap"
                normalized["city"] = str(normalized.get("city") or destination)
                normalized["keywords"] = str(normalized.get("keywords") or normalized.get("query") or "").strip()
                if not normalized["keywords"]:
                    return None
            elif task_type == "web_search":
                normalized["provider"] = "web"
                normalized["city"] = str(normalized.get("city") or destination)
                normalized["keywords"] = str(normalized.get("keywords") or normalized.get("query") or "").strip()
                if not normalized["keywords"]:
                    return None
            return normalized

        for raw_task in list(base_tasks) + list(extra_tasks or [])[:6]:
            task = normalize_task(raw_task)
            if not task:
                continue
            key = (
                task.get("type"),
                task.get("provider"),
                task.get("city"),
                task.get("keywords") or task.get("target") or task.get("direction"),
                task.get("date"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(task)

        return merged

    def _derive_end_date(self, start_date: str, duration_days: Any) -> str:
        if not start_date or not duration_days:
            return ""
        try:
            start = date.fromisoformat(str(start_date))
            return (start + timedelta(days=max(1, int(duration_days)) - 1)).isoformat()
        except Exception:
            return ""

    def _should_query_train(self, query: str) -> bool:
        q = query or ""
        if any(word in q for word in ("飞机", "航班", "自驾", "开车")) and not any(word in q for word in ("高铁", "火车", "动车", "12306")):
            return False
        return True

    async def _execute_transport_tasks(self, search_plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        train_tasks = search_plan.get("tasks_by_type", {}).get("train", [])
        transport = {
            "outbound_trains": [],
            "return_trains": [],
            "queries": [],
            "errors": [],
        }
        if not train_tasks:
            return transport

        async def query_train_task(task: Dict[str, Any]) -> Dict[str, Any]:
            params = {
                "date": str(task.get("date") or ""),
                "from_station": str(task.get("from_station") or ""),
                "to_station": str(task.get("to_station") or ""),
            }
            earliest_start_time, latest_start_time = self._time_window_hours(task.get("time_window"))
            try:
                raw = await TrainService().get_tickets(
                    date=params["date"],
                    from_station=params["from_station"],
                    to_station=params["to_station"],
                    train_filter_flags=str(task.get("train_filter_flags") or self._train_filter_flags(user_query or "")),
                    earliest_start_time=earliest_start_time,
                    latest_start_time=latest_start_time,
                    sort_flag="startTime",
                    sort_reverse=task.get("direction") == "return",
                    limited_num=80,
                    response_format="json",
                )
                tickets = self._normalize_train_tickets(raw)
                selected = self._select_time_fit_trains(tickets, task.get("time_window"))
                return {
                    "task": task,
                    "params": params,
                    "tickets": selected,
                    "raw_count": len(tickets),
                    "summary": self._format_train_ticket_summary(params, selected, raw),
                    "source": "12306 MCP get-tickets",
                }
            except Exception as e:
                logger.warning("Train task failed %s: %s", params, e)
                return {
                    "task": task,
                    "params": params,
                    "tickets": [],
                    "error": str(e),
                    "source": "12306 MCP get-tickets",
                }

        results = await asyncio.gather(*[query_train_task(task) for task in train_tasks])
        for item in results:
            direction = item.get("task", {}).get("direction")
            transport["queries"].append(item)
            if item.get("error"):
                transport["errors"].append(item)
            elif direction == "return":
                transport["return_trains"] = item.get("tickets", [])
            else:
                transport["outbound_trains"] = item.get("tickets", [])
        return transport

    def _time_window_hours(self, time_window: Any) -> tuple[int | None, int | None]:
        if not isinstance(time_window, list) or len(time_window) != 2:
            return None, None

        def parse_hour(value: Any, round_up: bool = False) -> int | None:
            match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", str(value or "").strip())
            if not match:
                return None
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if round_up and minute > 0:
                hour += 1
            if 0 <= hour <= 24:
                return hour
            return None

        return parse_hour(time_window[0]), parse_hour(time_window[1], round_up=True)

    def _select_time_fit_trains(self, tickets: List[Dict[str, Any]], time_window: Any, limit: int = 6) -> List[Dict[str, Any]]:
        if not tickets:
            return []
        if not isinstance(time_window, list) or len(time_window) != 2:
            return tickets[:limit]
        start, end = str(time_window[0]), str(time_window[1])
        matched = [
            ticket
            for ticket in tickets
            if start <= str(ticket.get("start_time") or "") <= end
        ]
        return (matched or tickets)[:limit]

    def _select_scenic_pois(
        self,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category in ("景点", "博物馆", "公园"):
            selected.extend(pois_by_category.get(category, [])[:3])
        if not selected:
            selected = [poi for poi in pois if poi.get("name")][:8]
        seen = set()
        unique = []
        for poi in selected:
            key = poi.get("id") or poi.get("name")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(poi)
        return unique[:8]

    async def _query_nearby_trip_support(
        self,
        amap: AmapService,
        destination: str,
        selected_scenic: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        support = {
            "by_poi": [],
            "hotels": [],
            "restaurants": [],
            "stations": [],
        }

        async def search_near_poi(poi: Dict[str, Any]) -> Dict[str, Any]:
            name = str(poi.get("name") or "").strip()
            if not name:
                return {}
            try:
                hotels, restaurants, stations = await asyncio.gather(
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 经济型酒店"),
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 餐厅 美食"),
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 地铁站 火车站"),
                )
                return {
                    "poi": self._compact_poi(poi),
                    "hotels": hotels[:5],
                    "restaurants": restaurants[:5],
                    "stations": stations[:5],
                }
            except Exception as e:
                logger.warning("Nearby query failed for %s: %s", name, e)
                return {"poi": self._compact_poi(poi), "error": str(e)}

        nearby_items = await asyncio.gather(*[search_near_poi(poi) for poi in selected_scenic[:5]])
        seen = {"hotels": set(), "restaurants": set(), "stations": set()}
        for item in nearby_items:
            if not item:
                continue
            support["by_poi"].append(item)
            for key in ("hotels", "restaurants", "stations"):
                for poi in item.get(key, []) or []:
                    poi_key = poi.get("id") or poi.get("name")
                    if not poi_key or poi_key in seen[key]:
                        continue
                    seen[key].add(poi_key)
                    support[key].append(poi)
        support["hotels"] = support["hotels"][:15]
        support["restaurants"] = support["restaurants"][:15]
        support["stations"] = support["stations"][:15]
        return support

    async def _query_trip_routes(
        self,
        amap: AmapService,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        routes: List[Dict[str, Any]] = []
        distances: List[Dict[str, Any]] = []
        route_candidates = self._select_route_candidates(pois_by_category, pois)
        if len(route_candidates) < 2:
            return routes, distances

        route_pairs = list(zip(route_candidates, route_candidates[1:]))[:6]

        async def query_route_pair(pair: tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
            start, end = pair
            origin = self._location_str(start)
            dest = self._location_str(end)
            if not origin or not dest:
                return {}
            route_item = {
                "from": start.get("name"),
                "to": end.get("name"),
                "from_location": origin,
                "to_location": dest,
            }
            try:
                walking, driving = await asyncio.gather(
                    amap.maps_direction_walking(origin=origin, destination=dest),
                    amap.maps_direction_driving(origin=origin, destination=dest),
                )
                route_item["walking"] = walking
                route_item["driving"] = driving
                route_item["recommended_modes"] = ["walking", "driving"]
                return route_item
            except Exception as e:
                logger.warning("Amap route query failed for %s -> %s: %s", start.get("name"), end.get("name"), e)
                route_item["error"] = str(e)
                return route_item

        routes = [item for item in await asyncio.gather(*[query_route_pair(pair) for pair in route_pairs]) if item]

        destination_point = self._location_str(route_candidates[0])
        origin_points = [self._location_str(item) for item in route_candidates[1:8]]
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
                        "from": [item.get("name") for item in route_candidates[1:8]],
                        "mode": "driving_distance",
                        "raw": distance_raw,
                    }
                )
            except Exception as e:
                logger.warning("Amap distance query failed for %s: %s", route_candidates[0].get("name"), e)
        return routes, distances

    def _compact_poi(self, poi: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "type": poi.get("type"),
            "location": poi.get("location"),
            "source_keywords": poi.get("source_keywords"),
        }

    def _build_search_quality_report(
        self,
        event_data: Dict[str, Any],
        transport: Dict[str, Any],
        pois: List[Dict[str, Any]],
        weather: Any,
        nearby: Dict[str, Any],
        routes: List[Dict[str, Any]],
        search_plan: Dict[str, Any] | None = None,
        supplemental_search: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        search_plan = search_plan or {}
        supplemental_search = supplemental_search or []
        missing = []
        warnings = []
        verified = []
        if event_data.get("origin") and event_data.get("destination") and event_data.get("start_date"):
            if transport.get("outbound_trains"):
                verified.append("outbound_train_options")
            else:
                missing.append("outbound_train_options")
        has_return_query = any(
            item.get("task", {}).get("direction") == "return"
            for item in transport.get("queries", [])
        )
        if has_return_query or (event_data.get("end_date") and event_data.get("return_location")):
            if transport.get("return_trains"):
                verified.append("return_train_options")
            else:
                missing.append("return_train_options")
        if pois:
            verified.append("destination_pois")
        else:
            missing.append("destination_pois")
        if isinstance(weather, dict) and weather.get("error"):
            warnings.append(f"weather_failed: {weather.get('error')}")
        elif weather:
            verified.append("weather")
        else:
            missing.append("weather")
        if nearby.get("hotels"):
            verified.append("nearby_hotels")
        else:
            missing.append("nearby_hotels")
        if nearby.get("restaurants"):
            verified.append("nearby_restaurants")
        if routes:
            verified.append("poi_routes")
        else:
            missing.append("poi_routes")
        for error in transport.get("errors", []):
            warnings.append(f"train_query_failed: {error.get('params')} {error.get('error')}")

        verification_aliases = set(verified)
        if transport.get("outbound_trains") or transport.get("return_trains"):
            verification_aliases.update({"train_tickets", "transport_options"})
        if nearby.get("hotels"):
            verification_aliases.update({"lodging_candidates", "lodging_area"})
        if routes:
            verification_aliases.update({"routes", "route_distance"})
        if any(item.get("verified") for item in supplemental_search):
            verification_aliases.update({"official_attraction_policy", "opening_hours", "ticket_or_reservation_policy"})

        planned_must_verify = [
            str(item)
            for item in search_plan.get("must_verify", [])
            if str(item) not in verification_aliases
        ]
        blocking_missing = [
            str(item)
            for item in search_plan.get("blocking_missing_fields", [])
            if event_data.get(str(item)) in (None, "", [])
        ]
        non_blocking_gaps = [
            str(item)
            for item in search_plan.get("non_blocking_gaps", [])
            if item
        ]
        for item in planned_must_verify:
            if item not in missing:
                missing.append(item)
        for item in blocking_missing:
            if item not in missing:
                missing.append(item)

        return {
            "verified_fields": verified,
            "missing": missing,
            "warnings": warnings,
            "must_verify": search_plan.get("must_verify", []),
            "unverified_must_verify": planned_must_verify,
            "blocking_missing_fields": blocking_missing,
            "non_blocking_gaps": non_blocking_gaps,
            "hard_constraints_require_official_sources": True,
        }

    def _format_trip_search_summary(self, bundle: Dict[str, Any]) -> str:
        destination = bundle.get("destination", {}).get("name", "")
        transport = bundle.get("transport", {})
        dest = bundle.get("destination", {})
        parts = [f"已完成 {destination} 行程检索素材收集。"]
        if transport.get("outbound_trains"):
            parts.append(f"去程候选高铁/火车 {len(transport['outbound_trains'])} 条。")
        if transport.get("return_trains"):
            parts.append(f"返程候选高铁/火车 {len(transport['return_trains'])} 条。")
        if dest.get("pois"):
            parts.append(f"目的地 POI {len(dest['pois'])} 个，含景点、餐饮、住宿和交通枢纽。")
        nearby = dest.get("nearby") or {}
        if nearby.get("hotels"):
            parts.append(f"景点周边住宿候选 {len(nearby['hotels'])} 个。")
        if nearby.get("restaurants"):
            parts.append(f"景点周边餐饮候选 {len(nearby['restaurants'])} 个。")
        if dest.get("routes"):
            parts.append(f"景点间路线/交通方式 {len(dest['routes'])} 组。")
        missing = bundle.get("quality", {}).get("missing") or []
        if missing:
            parts.append("仍缺少：" + "、".join(missing) + "。")
        return "".join(parts)

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
