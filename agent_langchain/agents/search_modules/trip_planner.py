from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List

from utils.json_parser import robust_json_parse
from utils.langchain_runtime import ainvoke_text
from utils.budget_utils import detect_budget_level, infer_budget_profile
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

class TripSearchPlannerMixin:
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
            ("accessibility", ("无障碍", "轮椅", "婴儿车", "少走路")),
        ]
        for tag, words in keyword_rules:
            if any(word in q for word in words):
                focus_tags.append(tag)
        budget_profile = infer_budget_profile(str(event_data.get("budget_level") or detect_budget_level(q) or ""))
        if budget_profile:
            focus_tags.append(budget_profile)

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
