"""
事项收集智能体
职责：收集用户的出发地/事项地点/事项时间/返程地
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Dict, List, Optional

from utils.llm_json import parse_json_text
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)
from utils.langchain_runtime import ainvoke_text
from utils.budget_utils import detect_budget_level, detect_lodging_budget

logger = logging.getLogger(__name__)

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")
ConfigDict = getattr(_pydantic, "ConfigDict")


class EventCollectionOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    origin: Optional[str] = Field(default=None)
    destination: Optional[str] = Field(default=None)
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    duration_days: Optional[int] = Field(default=None)
    budget_level: Optional[str] = Field(default=None)
    lodging_budget_per_night: Optional[int] = Field(default=None)
    lodging_budget_per_night_min: Optional[int] = Field(default=None)
    lodging_budget_per_night_max: Optional[int] = Field(default=None)
    meal_budget_preference: Optional[str] = Field(default=None)
    transport_budget_preference: Optional[str] = Field(default=None)
    pace_preference: Optional[str] = Field(default=None)
    return_location: Optional[str] = Field(default=None)
    trip_purpose: Optional[str] = Field(default=None)
    missing_info: List[str] = Field(default_factory=list)
    extracted_count: int = Field(default=0)
    summary: str = Field(default="")
    suggested_options: List[Dict[str, str]] = Field(default_factory=list)


class EventCollectionAgent:
    """事项收集智能体"""

    def __init__(self, name: str = "EventCollectionAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", {})
        user_query = context.get("rewritten_query", "") or state.get("user_query", "")
        user_preferences = context.get("user_preferences", {})

        direct_result = self._try_direct_guidance(user_query)
        if direct_result:
            return direct_result

        extracted_result = self._try_rule_extract(user_query, user_preferences)
        if extracted_result:
            return extracted_result

        background_info = ""
        if user_preferences:
            bg_parts = ["【用户背景信息】（可用于推断缺失信息）"]
            if user_preferences.get("home_location"):
                bg_parts.append(f"• 家庭住址: {user_preferences['home_location']}")
            if user_preferences.get("hotel_brands"):
                bg_parts.append(f"• 酒店偏好: {', '.join(user_preferences['hotel_brands'])}")
            if user_preferences.get("airlines"):
                bg_parts.append(f"• 航空偏好: {', '.join(user_preferences['airlines'])}")
            if len(bg_parts) > 1:
                background_info = "\n".join(bg_parts) + "\n\n"

        from datetime import datetime

        current_date = datetime.now().strftime("%Y年%m月%d日")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        prompt = f"""你是事项收集专家，负责提取旅行的基础信息。

【当前时间】
{current_date} {weekday}

{background_info}【用户输入】
{user_query}

【提取要求】
请尽可能提取以下信息：
1. origin - 出发地
2. destination - 目的地
3. start_date - 出发日期（YYYY-MM-DD格式）
4. end_date - 返程日期
5. duration_days - 行程天数
6. budget_level - 本次行程预算偏好
7. pace_preference - 本次行程节奏偏好
8. return_location - 返程地
9. trip_purpose - 行程目的

【日期处理规则】
- 当前时间是{current_date}
- 用户说"明天"、"后天"、"下周"时，请推断具体日期
- 日期统一输出 YYYY-MM-DD

【特殊处理】
- 北京一日游：destination 和 origin 均可为北京
- 一日游：duration_days=1
- 若用户未说出发地且有家庭住址，可推断 origin
"""

        try:
            result = await self._invoke_structured(prompt)
            result_data = result.model_dump()

            # 兜底补充
            if not result_data.get("missing_info"):
                result_data["missing_info"] = [
                    k
                    for k in [
                        "origin",
                        "destination",
                        "start_date",
                        "end_date",
                        "duration_days",
                        "budget_level",
                        "pace_preference",
                        "return_location",
                        "trip_purpose",
                    ]
                    if result_data.get(k) in (None, "")
                ]
            if not result_data.get("extracted_count"):
                result_data["extracted_count"] = 7 - len(result_data.get("missing_info", []))
        except Exception as e:
            logger.error("Event collection failed: %s", e)
            result_data = {
                "missing_info": ["所有信息"],
                "extracted_count": 0,
                "error": str(e),
            }

        return result_data

    def _try_direct_guidance(self, user_query: str) -> Optional[Dict[str, Any]]:
        query = (user_query or "").strip()
        asks_required_fields = (
            ("需要" in query or "要" in query)
            and any(word in query for word in ("提供", "填写", "告诉", "补充"))
            and any(word in query for word in ("什么", "哪些", "啥"))
            and any(word in query for word in ("意向", "信息", "内容", "资料", "字段"))
        )
        if not asks_required_fields:
            return None

        missing_info = [
            "目的地",
            "出发地",
            "出发日期或大致时间",
            "行程天数",
            "预算范围",
            "同行人情况",
            "旅行偏好（自然风景/美食/亲子/轻松/紧凑等）",
        ]
        return {
            "origin": None,
            "destination": None,
            "start_date": None,
            "end_date": None,
            "duration_days": None,
            "budget_level": None,
            "pace_preference": None,
            "return_location": None,
            "trip_purpose": None,
            "missing_info": missing_info,
            "extracted_count": 0,
            "summary": "为了继续规划行程，请提供目的地、出发地、时间、天数、预算、同行人和旅行偏好等信息。",
            "suggested_options": self._build_suggested_options(missing_info),
        }

    def _try_rule_extract(self, user_query: str, user_preferences: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = (user_query or "").strip()
        if not query:
            return None

        places = self._extract_destination(query)
        dates = self._extract_dates(query)
        budget = self._extract_budget(query)

        origin = places.get("origin")
        destination = places.get("destination")
        start_date = dates.get("start_date")
        end_date = dates.get("end_date")
        duration_days = self._extract_duration(query)
        trip_purpose = self._extract_trip_purpose(query)
        budget_level = budget.get("budget_level")
        lodging_budget = budget.get("lodging_budget")
        meal_budget_preference = budget.get("meal_budget_preference")
        transport_budget_preference = budget.get("transport_budget_preference")
        pace_preference = self._extract_pace(query)

        if not any(
            [
                origin,
                destination,
                start_date,
                duration_days,
                trip_purpose,
                budget_level,
                lodging_budget,
                meal_budget_preference,
                transport_budget_preference,
                pace_preference,
            ]
        ):
            return None

        if not origin and isinstance(user_preferences, dict):
            origin = user_preferences.get("home_location")
        if not budget_level and isinstance(user_preferences, dict):
            budget_level = user_preferences.get("budget_level")
        if not pace_preference and isinstance(user_preferences, dict):
            pace_preference = user_preferences.get("pace_preference")

        fields = {
            "origin": origin,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": duration_days,
            "budget_level": budget_level,
            "lodging_budget_per_night": lodging_budget.get("value") if lodging_budget else None,
            "lodging_budget_per_night_min": lodging_budget.get("min") if lodging_budget else None,
            "lodging_budget_per_night_max": lodging_budget.get("max") if lodging_budget else None,
            "meal_budget_preference": meal_budget_preference,
            "transport_budget_preference": transport_budget_preference,
            "pace_preference": pace_preference,
            "return_location": origin,
            "trip_purpose": trip_purpose,
        }
        missing_info = [
            key
            for key in ["origin", "destination", "start_date", "duration_days", "budget_level", "pace_preference"]
            if fields.get(key) in (None, "")
        ]
        summary_parts = []
        if origin:
            summary_parts.append(f"出发地为{origin}")
        if destination:
            summary_parts.append(f"目的地为{destination}")
        if start_date:
            summary_parts.append(f"出发日期为{start_date}")
        if duration_days:
            summary_parts.append(f"行程{duration_days}天")
        if budget_level:
            summary_parts.append(f"预算为{budget_level}")
        if lodging_budget:
            if lodging_budget.get("min") is not None and lodging_budget.get("max") is not None:
                summary_parts.append(f"住宿每晚{lodging_budget['min']}到{lodging_budget['max']}元")
            elif lodging_budget.get("max") is not None:
                summary_parts.append(f"住宿每晚{lodging_budget['max']}元以内")
        if meal_budget_preference:
            summary_parts.append(f"餐饮{meal_budget_preference}")
        if transport_budget_preference:
            summary_parts.append(f"交通{transport_budget_preference}")
        if pace_preference:
            summary_parts.append(f"节奏为{pace_preference}")

        return {
            **fields,
            "missing_info": missing_info,
            "extracted_count": 7 - len([v for v in fields.values() if v in (None, "")]),
            "summary": "已提取：" + "，".join(summary_parts) if summary_parts else "已提取部分行程信息。",
            "suggested_options": self._build_suggested_options(missing_info),
        }

    def _extract_destination(self, text: str) -> Dict[str, Optional[str]]:
        """提取出发地和目的地。"""
        origin = None
        destination = None
        movement_match = re.search(r"从(?P<origin>[\u4e00-\u9fa5A-Za-z0-9·（）()]+?)(?:出发)?(?:去|到|前往)(?P<dest>[\u4e00-\u9fa5A-Za-z0-9·（）()]+)", text)
        if movement_match:
            origin = self._clean_place(movement_match.group("origin"))
            destination = self._clean_place(movement_match.group("dest"))

        if not origin or not destination:
            compact_match = re.search(r"(?P<origin>[\u4e00-\u9fa5]{2,8})(?:出发)?(?:去|到|前往)(?P<dest>[\u4e00-\u9fa5]{2,8})", text)
            if compact_match:
                origin = origin or self._clean_place(compact_match.group("origin"))
                destination = destination or self._clean_place(compact_match.group("dest"))
        if self._is_invalid_origin_candidate(origin):
            origin = None

        city_origin, city_destination = self._extract_places_by_city_names(text)
        if city_origin and city_destination:
            origin = city_origin
            destination = city_destination
        elif city_destination and not destination:
            destination = city_destination

        if not destination:
            dest_match = re.search(r"(?:去|到|前往)(?P<dest>[\u4e00-\u9fa5]{2,8})", text)
            if dest_match:
                destination = self._clean_place(dest_match.group("dest"))

        return {"origin": origin, "destination": destination}

    def _extract_dates(self, text: str) -> Dict[str, Optional[str]]:
        """提取出发和返程日期。"""
        start_date = self._extract_start_date(text)
        duration_days = self._extract_duration_days(text)
        end_date = None
        if start_date and duration_days:
            from datetime import datetime, timedelta

            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = (start_dt + timedelta(days=duration_days - 1)).isoformat()
        return {"start_date": start_date, "end_date": end_date}

    def _extract_duration(self, text: str) -> Optional[int]:
        """提取行程天数。"""
        return self._extract_duration_days(text)

    def _extract_budget(self, text: str) -> Dict[str, Any]:
        """提取本次行程预算、住宿、餐饮和交通预算偏好。"""
        return {
            "budget_level": self._extract_budget_level(text),
            "lodging_budget": self._extract_lodging_budget(text),
            "meal_budget_preference": self._extract_meal_budget_preference(text),
            "transport_budget_preference": self._extract_transport_budget_preference(text),
        }

    def _extract_pace(self, text: str) -> Optional[str]:
        """提取行程节奏偏好。"""
        return self._extract_pace_preference(text)

    def _extract_trip_purpose(self, text: str) -> Optional[str]:
        if any(word in text for word in ("玩", "旅游", "旅行")):
            return "旅游"
        if "出差" in text:
            return "出差"
        return None

    def _clean_place(self, value: str) -> str:
        place = (value or "").strip()
        place = re.sub(r"^(帮我|请|麻烦|规划|安排|下周|明天|后天|今天|从|我想|我要|我想要|打算|计划)", "", place).strip()
        place = re.sub(
            r"(出差|旅游|旅行|玩|游玩|行程|规划|安排|三天|两天|一天|\d+\s*天|[一二三四五六七八九十]+天|之旅|之行|的行程|的旅行|的旅游|的出差|期间.*)",
            "",
            place,
        ).strip()
        return place

    def _is_invalid_origin_candidate(self, value: Optional[str]) -> bool:
        if not value:
            return False
        normalized = value.strip()
        return normalized in {"我", "我想", "我要", "想", "想要", "打算", "计划", "帮我", "请帮我", "麻烦"}

    def _extract_budget_level(self, query: str) -> Optional[str]:
        return detect_budget_level(query)

    def _extract_lodging_budget(self, query: str) -> Optional[Dict[str, int]]:
        min_value, max_value = detect_lodging_budget(query)
        if min_value is None and max_value is None:
            return None
        if min_value is not None and max_value is not None:
            return {"min": min_value, "max": max_value}
        if max_value is not None:
            return {"value": max_value, "max": max_value}
        return {"min": min_value}

    def _extract_meal_budget_preference(self, query: str) -> Optional[str]:
        if any(word in query for word in ("餐饮", "吃饭", "美食")) and any(word in query for word in ("节省", "省钱", "性价比")):
            return "节省" if any(word in query for word in ("节省", "省钱")) else "性价比"
        return None

    def _extract_transport_budget_preference(self, query: str) -> Optional[str]:
        if "交通" in query and any(word in query for word in ("节省", "省钱", "性价比")):
            return "节省" if any(word in query for word in ("节省", "省钱")) else "性价比"
        return None

    def _extract_pace_preference(self, query: str) -> Optional[str]:
        if any(word in query for word in ("轻松", "不要太赶", "慢一点")):
            return "轻松"
        if any(word in query for word in ("紧凑", "多看", "多安排")):
            return "紧凑"
        if "均衡" in query:
            return "均衡"
        return None

    def _build_suggested_options(self, missing_info: List[str]) -> List[Dict[str, str]]:
        options: List[Dict[str, str]] = []
        missing = set(str(item) for item in missing_info)
        if "budget_level" in missing:
            options.extend(
                [
                    {
                        "label": "经济型预算",
                        "description": "住宿每晚300元以内，整体更省钱。",
                        "message": "本次行程预算选择经济型，住宿每晚300元以内，餐饮和交通尽量节省。",
                    },
                    {
                        "label": "舒适型预算",
                        "description": "住宿每晚300到600元，兼顾体验和性价比。",
                        "message": "本次行程预算选择舒适型，住宿每晚300到600元，兼顾体验和性价比。",
                    },
                    {
                        "label": "品质型预算",
                        "description": "住宿每晚600元以上，优先便利和体验。",
                        "message": "本次行程预算选择品质型，住宿每晚600元以上，优先体验和便利。",
                    },
                ]
            )
        if "pace_preference" in missing:
            options.extend(
                [
                    {
                        "label": "轻松节奏",
                        "description": "每天2到3个核心地点，留出休息时间。",
                        "message": "本次行程节奏选择轻松，不要太赶，每天安排2到3个核心地点。",
                    },
                    {
                        "label": "均衡节奏",
                        "description": "景点、餐饮和休息都兼顾。",
                        "message": "本次行程节奏选择均衡，景点和休息都要兼顾。",
                    },
                    {
                        "label": "紧凑节奏",
                        "description": "尽量多安排景点，行程更充实。",
                        "message": "本次行程节奏选择紧凑，希望尽量多看一些地点。",
                    },
                ]
            )
        return options

    def _extract_places_by_city_names(self, query: str) -> tuple[Optional[str], Optional[str]]:
        common_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "苏州",
            "天津", "重庆", "厦门", "青岛", "大连", "宁波", "无锡", "长沙", "郑州", "济南",
            "哈尔滨", "沈阳", "昆明", "合肥", "福州", "石家庄", "南昌", "贵阳", "太原", "南宁",
        ]
        hits = []
        for city in common_cities:
            idx = query.find(city)
            if idx >= 0:
                hits.append((idx, city))
        hits.sort(key=lambda item: item[0])
        if len(hits) >= 2:
            return hits[0][1], hits[1][1]
        if len(hits) == 1 and any(word in query for word in ("去", "到", "前往")):
            return None, hits[0][1]
        return None, None

    def _extract_duration_days(self, query: str) -> Optional[int]:
        # "半天" / "半日" → 1 day (minimum unit for itinerary)
        if "半天" in query or "半日" in query:
            return 1
        digit_match = re.search(r"(\d+)\s*天", query)
        if digit_match:
            return int(digit_match.group(1))
        cn_nums = {
            "一": 1,
            "两": 2,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        for text, value in cn_nums.items():
            if f"{text}天" in query:
                return value
        return None

    def _extract_start_date(self, query: str) -> Optional[str]:
        from datetime import datetime, timedelta

        today = datetime.now().date()

        # "下周X" — specific weekday next week
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        next_weekday_match = re.search(r"下周([一二三四五六日天])", query)
        if next_weekday_match:
            target_wd = weekday_map[next_weekday_match.group(1)]
            days_ahead = (target_wd - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()

        if "下周" in query:
            days_until_next_monday = (7 - today.weekday()) % 7
            if days_until_next_monday == 0:
                days_until_next_monday = 7
            return (today + timedelta(days=days_until_next_monday)).isoformat()
        if "明天" in query:
            return (today + timedelta(days=1)).isoformat()
        if "后天" in query:
            return (today + timedelta(days=2)).isoformat()

        iso_match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", query)
        if iso_match:
            try:
                year = int(iso_match.group(1))
                month = int(iso_match.group(2))
                day = int(iso_match.group(3))
                return datetime(year, month, day).date().isoformat()
            except ValueError:
                return None

        date_match = re.search(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日?", query)
        if date_match:
            try:
                year = int(date_match.group(1) or today.year)
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                return datetime(year, month, day).date().isoformat()
            except ValueError:
                return None
        return None

    async def _invoke_structured(self, prompt: str) -> EventCollectionOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(EventCollectionOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, EventCollectionOutput):
                    return result
                if isinstance(result, dict):
                    return EventCollectionOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        # fallback
        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        return EventCollectionOutput.model_validate(parse_json_text(str(text)))
