"""
行程规划智能体
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Dict, List

from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)
from utils.langchain_runtime import ainvoke_text

logger = logging.getLogger(__name__)

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")
ConfigDict = getattr(_pydantic, "ConfigDict")


class ItineraryOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    itinerary: Dict[str, Any] = Field(default_factory=dict)
    planning_complete: bool = Field(default=False)
    summary: str = Field(default="")
    search_requests: List[Dict[str, Any]] = Field(default_factory=list)
    verification_summary: Dict[str, Any] = Field(default_factory=dict)


class ItineraryPlanningAgent:
    """行程规划智能体（主协调）"""

    def __init__(self, name: str = "ItineraryPlanningAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader

        self.skill_loader = SkillLoader()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query = ""
        context_info: Dict[str, Any] = {}
        previous_results: List[Dict[str, Any]] = []
        user_preferences: Dict[str, Any] = {}

        context_info = state.get("context", {})
        user_query = context_info.get("rewritten_query", "") or state.get("user_query", "")
        previous_results = state.get("previous_results", [])
        user_preferences = context_info.get("user_preferences", {})

        all_info: Dict[str, Any] = {"user_query": user_query, "context": context_info}
        for prev in previous_results:
            agent_name = prev.get("agent_name", "")
            result_data = prev.get("result", {}).get("data", {})
            if result_data and agent_name:
                all_info[agent_name] = result_data
                if agent_name == "clarification":
                    all_info["event_collection"] = result_data
                elif agent_name == "search":
                    all_info["information_query"] = result_data
                elif agent_name == "memory":
                    all_info["memory_query"] = result_data
                    if isinstance(result_data.get("preferences"), dict):
                        merged = dict(user_preferences or {})
                        merged.update({k: v for k, v in result_data["preferences"].items() if v})
                        user_preferences = merged

        incomplete = self._check_missing_required_trip_fields(all_info)
        if incomplete:
            return {
                "itinerary": {},
                "planning_complete": False,
                "summary": f"行程关键信息不足，请先补充：{', '.join(incomplete)}。",
                "missing_info": incomplete,
            }

        external_gaps = self._external_search_gaps(all_info)
        if external_gaps:
            return {
                "itinerary": {},
                "planning_complete": False,
                "summary": "当前外部信息不足，先补充检索后再生成行程。",
                "search_requests": external_gaps,
            }

        preferences_info = ""
        if user_preferences:
            pref_parts = ["【用户偏好】（规划时优先考虑）"]
            if user_preferences.get("home_location"):
                pref_parts.append(f"• 家庭住址: {user_preferences['home_location']}")
            if user_preferences.get("hotel_brands"):
                pref_parts.append(f"• 酒店偏好: {', '.join(user_preferences['hotel_brands'])}")
            if user_preferences.get("airlines"):
                pref_parts.append(f"• 航空偏好: {', '.join(user_preferences['airlines'])}")
            if user_preferences.get("seat_preference"):
                pref_parts.append(f"• 座位偏好: {user_preferences['seat_preference']}")
            if len(pref_parts) > 1:
                preferences_info = "\n".join(pref_parts) + "\n\n"

        memory_info = all_info.get("memory") or all_info.get("memory_query") or {}
        preference_follow_up = ""
        if isinstance(memory_info, dict) and memory_info.get("follow_up_question"):
            preference_follow_up = str(memory_info.get("follow_up_question"))

        from datetime import datetime

        current_date = datetime.now().strftime("%Y年%m月%d日")
        current_month = datetime.now().month
        current_season = (
            "冬季"
            if current_month in [12, 1, 2]
            else "春季" if current_month in [3, 4, 5] else "夏季" if current_month in [6, 7, 8] else "秋季"
        )
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        skill_instruction = self.skill_loader.get_skill_content("plan-trip")
        if not skill_instruction:
            skill_instruction = "请根据用户需求和偏好生成行程规划。"

        prompt = f"""你是一个高级行程规划专家。

【当前时间】
{current_date} {weekday}，当前季节是{current_season}

【用户需求】
{user_query}

{preferences_info}【所有收集的信息】
{json.dumps(all_info, ensure_ascii=False, indent=2)}

【任务说明与指南】
{skill_instruction}

如果已有信息足够，请返回结构化行程规划，至少包含 itinerary 与 planning_complete。
如果外部信息不足以支撑具体规划，不要编造；请返回 planning_complete=false、itinerary={{}}，
并在 search_requests 中列出需要补充检索的请求，每项包含 keywords、reason、expected_output。

【硬约束规则】
- 不得编造高铁/航班车次、出发到达时间、票价、余票、酒店门店、预约状态。
- 只有在 search 结果中有明确来源时，才能把硬约束写成“已核验”。
- 交通、酒店、预约、总预算若未核验，必须放入 itinerary.hard_constraints，status 标为 unverified 或 needs_official_check。
- 行程动线必须服务于用户节奏偏好；轻松节奏每天最多 2 个核心景点，故宫这类高体力景点不得和过多景点硬串。
- 必须提供约不到/买不到时的备选方案。
"""

        try:
            result = await self._invoke_structured(prompt)
            result_data = result.model_dump()
            if "itinerary" not in result_data:
                result_data["itinerary"] = {}
            if "planning_complete" not in result_data:
                result_data["planning_complete"] = bool(result_data.get("itinerary"))
        except Exception as e:
            logger.error("Itinerary planning failed: %s", e)
            result_data = self._fallback_itinerary(all_info, user_preferences)
            result_data["summary"] = "模型结构化输出不可用，已基于已收集信息生成兜底行程。"

        result_data = self._enforce_hard_constraint_disclosure(result_data, all_info)

        if preference_follow_up and isinstance(result_data.get("itinerary"), dict):
            notes = result_data["itinerary"].setdefault("notes", [])
            if isinstance(notes, list) and preference_follow_up not in notes:
                notes.append(preference_follow_up)
        if preference_follow_up:
            result_data["preference_follow_up"] = preference_follow_up

        return result_data

    def _external_search_gaps(self, all_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = all_info.get("context") if isinstance(all_info.get("context"), dict) else {}
        if context.get("search_refinement_count", 0):
            return []

        event = all_info.get("clarification") or all_info.get("event_collection") or {}
        search = all_info.get("search") or all_info.get("information_query") or {}
        search_results = search.get("results") if isinstance(search, dict) else {}
        if not isinstance(search_results, dict):
            search_results = {}

        destination = event.get("destination") or search_results.get("destination") or ""
        try:
            duration_days = int(event.get("duration_days") or 1)
        except Exception:
            duration_days = 1

        pois = [p for p in (search_results.get("pois") or []) if isinstance(p, dict) and p.get("name")]
        pois_by_category = search_results.get("pois_by_category") if isinstance(search_results.get("pois_by_category"), dict) else {}
        requests: List[Dict[str, Any]] = []
        if len(pois) < max(4, duration_days * 2):
            requests.append(
                {
                    "keywords": "景点",
                    "reason": "当前 POI 数量不足，无法支撑每天上午/下午的具体景点安排。",
                    "expected_output": f"{destination}更多可安排景点、地址、区域和类型",
                }
            )
        if not search_results.get("weather"):
            requests.append(
                {
                    "keywords": "天气",
                    "reason": "缺少目的地天气，无法给出雨天替代和出行提醒。",
                    "expected_output": f"{destination}天气信息",
                }
            )
        food_candidates = pois_by_category.get("美食 餐厅") or [
            poi for poi in pois if "餐" in str(poi.get("source_keywords", "")) or "餐" in str(poi.get("type", ""))
        ]
        if destination and not food_candidates and not any("餐" in str(req.get("keywords", "")) for req in requests):
            requests.append(
                {
                    "keywords": "美食 餐厅",
                    "reason": "缺少餐饮候选，行程只能给出笼统用餐建议。",
                    "expected_output": f"{destination}本地餐饮或商圈候选",
                }
            )
        hotel_candidates = pois_by_category.get("经济型酒店") or [
            poi for poi in pois if "酒店" in str(poi.get("source_keywords", "")) or "酒店" in str(poi.get("type", ""))
        ]
        if destination and not hotel_candidates:
            requests.append(
                {
                    "keywords": "经济型酒店 地铁站",
                    "reason": "缺少住宿候选，无法评估经济型住宿位置和通勤。",
                    "expected_output": f"{destination}经济型酒店候选和地铁便利性",
                }
            )
        supplemental = search_results.get("supplemental_search") or []
        if not supplemental:
            origin = event.get("origin") or ""
            start_date = event.get("start_date") or ""
            if origin and destination and start_date:
                requests.extend(
                    [
                        {
                            "keywords": f"{origin} 到 {destination} {start_date} 高铁 火车 12306",
                            "reason": "交通是硬约束，不能编造车次、时间、票价或余票。",
                            "expected_output": "官方或可核验的往返交通查询入口、车站、车次核验方式",
                        },
                        {
                            "keywords": f"{destination} {start_date} 故宫 天安门 预约 官方",
                            "reason": "核心景点预约是硬约束，需要核验预约入口、放票时间和替代方案。",
                            "expected_output": "官方预约入口、预约规则、约不到时的替代方案",
                        },
                        {
                            "keywords": f"{destination} 经济型 酒店 地铁站 附近 {start_date}",
                            "reason": "酒店位置和价格会影响动线与预算，需要核验具体门店和区位。",
                            "expected_output": "经济型酒店候选、位置、预算区间、交通便利性",
                        },
                    ]
                )
        return requests[:3]

    def _check_missing_required_trip_fields(self, all_info: Dict[str, Any]) -> List[str]:
        event = all_info.get("clarification") or all_info.get("event_collection") or {}
        if not isinstance(event, dict):
            return ["destination", "start_date", "duration_days"]

        required = ["destination", "start_date", "duration_days"]
        missing = []
        existing_missing = {str(item) for item in event.get("missing_info", [])}
        for field in required:
            if field in existing_missing or event.get(field) in (None, ""):
                missing.append(field)
        return missing

    def _fallback_itinerary(self, all_info: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        event = all_info.get("clarification") or all_info.get("event_collection") or {}
        search = all_info.get("search") or all_info.get("information_query") or {}
        search_results = search.get("results") if isinstance(search, dict) else {}
        if not isinstance(search_results, dict):
            search_results = {}

        origin = event.get("origin") or user_preferences.get("home_location") or "出发地"
        destination = event.get("destination") or search_results.get("destination") or "目的地"
        start_date = event.get("start_date") or "出发日"
        duration_days = event.get("duration_days") or 3
        try:
            duration_days = int(duration_days)
        except Exception:
            duration_days = 3
        duration_days = max(1, min(duration_days, 7))

        pois = [p for p in (search_results.get("pois") or []) if isinstance(p, dict) and p.get("name")]
        if not pois:
            pois = [
                {"name": "城市核心景区", "adname": destination, "type": "景点"},
                {"name": "代表性街区", "adname": destination, "type": "休闲街区"},
                {"name": "本地特色餐饮区", "adname": destination, "type": "餐饮"},
                {"name": "博物馆或室内展馆", "adname": destination, "type": "文化场馆"},
                {"name": "公园或观景点", "adname": destination, "type": "公园"},
                {"name": "夜景区域", "adname": destination, "type": "夜间休闲"},
            ]
        route_hint = self._route_hint(search_results.get("routes") or [])
        weather_hint = self._weather_hint(search_results.get("weather"))

        daily_plans = []
        for day in range(1, duration_days + 1):
            idx = (day - 1) * 2
            morning = pois[idx % len(pois)]
            afternoon = pois[(idx + 1) % len(pois)]
            evening_area = self._poi_area(afternoon) or self._poi_area(morning) or destination
            daily_plans.append(
                {
                    "day": day,
                    "activities": [
                        {
                            "time": "上午",
                            "activity": morning.get("name", ""),
                            "description": self._poi_description(morning, "上午优先安排核心游览点，避开午后高峰。"),
                            "transport": "从酒店出发优先选择地铁；若跨区或携带行李，改用打车。",
                        },
                        {
                            "time": "下午",
                            "activity": afternoon.get("name", ""),
                            "description": self._poi_description(afternoon, "下午安排相邻或同城热门地点，控制换乘次数。"),
                            "transport": route_hint or "上午和下午地点距离较近时步行或骑行，跨区时使用地铁/打车。",
                        },
                        {
                            "time": "晚上",
                            "activity": f"{evening_area}周边晚餐与轻松散步",
                            "description": f"晚餐建议选择{evening_area}附近评分较高、排队压力较低的本地餐厅；轻松节奏下不再追加远距离景点。",
                            "transport": "晚间优先选择酒店附近或同一区域活动，减少夜间通勤。",
                        },
                    ],
                    "meals": {
                        "lunch": f"午餐放在{self._poi_area(morning) or morning.get('name', destination)}附近，减少中午折返。",
                        "dinner": f"晚餐放在{evening_area}附近，方便饭后返回酒店。",
                    },
                }
            )

        notes = [
            f"建议从{origin}前往{destination}，出发日期：{start_date}。",
            "本方案只把高德 POI/路线作为动线参考；交通车次、酒店价格、门票预约必须以官方渠道实时核验为准。",
        ]
        if weather_hint:
            notes.append(weather_hint)
        if pois and pois[0].get("name") != "城市核心景区":
            notes.append("景点候选来自高德 MCP POI 检索；开放时间、门票和预约状态出发前仍需确认。")

        return {
            "itinerary": {
                "title": f"{destination}{duration_days}日旅行计划",
                "duration": f"{duration_days}天",
                "hard_constraints": self._build_hard_constraints(event, search_results),
                "budget_estimate": self._build_budget_estimate(event),
                "daily_plans": daily_plans,
                "fallback_options": self._build_fallback_options(destination),
                "notes": notes,
            },
            "planning_complete": True,
            "verification_summary": {
                "status": "needs_official_check",
                "message": "交通、酒店和预约未接入官方实时库存，已按待核验硬约束处理。",
            },
        }

    def _enforce_hard_constraint_disclosure(self, result_data: Dict[str, Any], all_info: Dict[str, Any]) -> Dict[str, Any]:
        itinerary = result_data.get("itinerary")
        if not isinstance(itinerary, dict) or not itinerary:
            return result_data

        event = all_info.get("clarification") or all_info.get("event_collection") or {}
        search = all_info.get("search") or all_info.get("information_query") or {}
        search_results = search.get("results") if isinstance(search, dict) else {}
        if not isinstance(search_results, dict):
            search_results = {}

        itinerary.setdefault("hard_constraints", self._build_hard_constraints(event, search_results))
        itinerary.setdefault("budget_estimate", self._build_budget_estimate(event))
        itinerary.setdefault("fallback_options", self._build_fallback_options(event.get("destination") or search_results.get("destination") or "目的地"))
        notes = itinerary.setdefault("notes", [])
        if isinstance(notes, list):
            warning = "交通车次、酒店价格和景点预约未接入官方实时库存时，均需按待核验处理，不能直接视为已确认安排。"
            if warning not in notes:
                notes.insert(0, warning)

        result_data.setdefault(
            "verification_summary",
            {
                "status": "needs_official_check",
                "message": "已强制标注硬约束核验状态，避免把未验证信息包装成可执行结论。",
            },
        )
        return result_data

    def _build_hard_constraints(self, event: Dict[str, Any], search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        origin = event.get("origin") or "出发地"
        destination = event.get("destination") or search_results.get("destination") or "目的地"
        start_date = event.get("start_date") or "出发日期"
        reservation_watch = self._reservation_watch_date(start_date)
        supplemental = search_results.get("supplemental_search") or []
        has_transport_verified = any(
            item.get("verified") and ("高铁" in str(item) or "12306" in str(item))
            for item in supplemental
            if isinstance(item, dict)
        )
        has_reservation_verified = any(
            item.get("verified") and ("预约" in str(item) or "故宫" in str(item))
            for item in supplemental
            if isinstance(item, dict)
        )
        has_hotel_verified = any(
            item.get("verified") and "酒店" in str(item)
            for item in supplemental
            if isinstance(item, dict)
        )
        return [
            {
                "name": "往返交通",
                "status": "verified" if has_transport_verified else "needs_official_check",
                "action": f"请在铁路12306按 {start_date} 查询 {origin} 到 {destination} 的真实车次、到达站、票价和余票；本系统不得编造 G/D 车次。",
            },
            {
                "name": "景点预约",
                "status": "verified" if has_reservation_verified else "needs_official_check",
                "action": f"请使用故宫博物院官方渠道、天安门广场预约参观服务平台核验。{reservation_watch}",
            },
            {
                "name": "住宿位置",
                "status": "verified" if has_hotel_verified else "needs_official_check",
                "action": "请在酒店平台核验具体门店、价格、取消政策和到地铁站/火车站距离；未核验前只建议住在地铁站附近的经济型连锁酒店。",
            },
        ]

    def _reservation_watch_date(self, start_date: Any) -> str:
        from datetime import datetime, timedelta

        try:
            watch = datetime.strptime(str(start_date), "%Y-%m-%d").date() - timedelta(days=7)
            return f"若按提前7天放票，建议从 {watch.isoformat()} 起关注；实际规则以官方为准。"
        except Exception:
            return "请根据实际出发日期倒推官方放票时间；实际规则以官方为准。"

    def _build_budget_estimate(self, event: Dict[str, Any]) -> Dict[str, Any]:
        try:
            days = int(event.get("duration_days") or 3)
        except Exception:
            days = 3
        nights = max(days - 1, 1)
        return {
            "status": "rough_estimate",
            "items": [
                {"name": "往返大交通", "range": "待 12306/航司实时核验"},
                {"name": "住宿", "range": f"经济型约 {nights * 250}-{nights * 450} 元（按每晚250-450粗估）"},
                {"name": "门票/预约", "range": "约 100-300 元，需按实际预约景点核验"},
                {"name": "市内交通", "range": f"约 {days * 30}-{days * 80} 元"},
                {"name": "餐饮", "range": f"约 {days * 100}-{days * 180} 元"},
            ],
            "note": "预算为粗估，不含未核验的大交通价格；最终以购票、酒店和预约平台为准。",
        }

    def _build_fallback_options(self, destination: str) -> List[Dict[str, str]]:
        return [
            {
                "scenario": "故宫或核心景点约不到",
                "option": f"改成{destination}博物馆/公园/历史街区组合，保留同一区域轻松动线。",
            },
            {
                "scenario": "理想车次无票",
                "option": "优先调整出发时段或到达站，再压缩第一天景点，不反向增加行程强度。",
            },
            {
                "scenario": "酒店价格超预算",
                "option": "改住地铁沿线外扩1-3站的经济型酒店，减少打车次数来控制总预算。",
            },
        ]

    def _poi_area(self, poi: Dict[str, Any]) -> str:
        return str(poi.get("adname") or poi.get("business_area") or poi.get("address") or "").strip()

    def _poi_description(self, poi: Dict[str, Any], fallback: str) -> str:
        parts = []
        poi_type = str(poi.get("type") or "").strip()
        area = self._poi_area(poi)
        address = str(poi.get("address") or "").strip()
        if area:
            parts.append(f"位置参考：{area}")
        if address and address != area:
            parts.append(f"地址：{address}")
        if poi_type:
            parts.append(f"类型：{poi_type}")
        parts.append(fallback)
        return "；".join(parts)

    def _route_hint(self, routes: List[Any]) -> str:
        for route in routes:
            if isinstance(route, str) and route.strip():
                return route.strip()[:120]
            if not isinstance(route, dict):
                continue
            route_text = json.dumps(route, ensure_ascii=False)
            if route_text:
                return f"高德路线参考：{route_text[:120]}"
        return ""

    def _weather_hint(self, weather: Any) -> str:
        if not weather:
            return ""
        if isinstance(weather, dict) and weather.get("error"):
            return "天气查询暂未成功；若遇雨天，优先把室外景点替换为博物馆、展馆或商圈。"
        if isinstance(weather, str):
            return f"天气参考：{weather[:160]}"
        return "天气信息已纳入参考；雨天可把室外景点替换为博物馆、展馆或商圈。"

    async def _invoke_structured(self, prompt: str) -> ItineraryOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(ItineraryOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, ItineraryOutput):
                    return result
                if isinstance(result, dict):
                    return ItineraryOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        return ItineraryOutput.model_validate(self._parse_json_text(str(text)))

    @staticmethod
    def _parse_json_text(text: str) -> Dict[str, Any]:
        clean = text.strip()
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
