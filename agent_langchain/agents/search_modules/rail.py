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

class RailSearchMixin:
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
        service = self._new_train_service()
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

