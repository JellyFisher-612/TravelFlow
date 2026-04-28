"""TravelFlow canonical agent adapters.

These classes expose the five-agent architecture while reusing the existing
skill implementations under ``.claude/skills``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from agents.preference_agent import PreferenceAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_skill_agent(skill_name: str):
    script_path = PROJECT_ROOT / ".claude" / "skills" / skill_name / "script" / "agent.py"
    module_name = f"travelflow_skill_{skill_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load skill agent from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for obj in module.__dict__.values():
        if isinstance(obj, type) and (hasattr(obj, "run") or hasattr(obj, "reply")):
            return obj
    raise ValueError(f"No runnable agent class found in {script_path}")


class SearchAgent:
    """信息检索智能体：通过高德 API 和必要的外部查询获取景点、天气、路线等信息。"""

    def __init__(self, name: str = "SearchAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self._delegate = _load_skill_agent("query-info")(name=name, model=model, **kwargs)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._delegate.run(state)


class PlanAgent:
    """行程规划智能体：基于偏好、行程要素和检索数据生成个性化计划。"""

    def __init__(self, name: str = "PlanAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self._delegate = _load_skill_agent("plan-trip")(name=name, model=model, **kwargs)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._delegate.run(state)


class ClarificationAgent:
    """事项收集智能体：提取缺失字段并生成需要追问的信息。"""

    def __init__(self, name: str = "ClarificationAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self._delegate = _load_skill_agent("event-collection")(name=name, model=model, **kwargs)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await self._delegate.run(state)


class MemoryAgent:
    """记忆与偏好智能体：管理长期偏好、历史行程和会话记忆查询。"""

    def __init__(self, name: str = "MemoryAgent", model=None, memory_manager=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        self._preference_agent = PreferenceAgent(
            name="MemoryPreferenceAgent",
            model=model,
            memory_manager=memory_manager,
        )

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", {})
        intents = context.get("intents", []) or []
        query = context.get("rewritten_query", "") or state.get("user_query", "")

        if self._is_preference_task(intents, query):
            return await self._preference_agent.run(state)

        return self._query_memory(query, context)

    def _parse_input(self, content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"context": {"rewritten_query": content}}
        return {"context": {"rewritten_query": str(content)}}

    def _is_preference_task(self, intents: List[Dict[str, Any]], query: str) -> bool:
        normalized_types = {
            str(item.get("type", "")).strip().lower()
            for item in intents
            if isinstance(item, dict)
        }
        if normalized_types & {"plan", "travel_plan", "itinerary", "itinerary_planning", "clarification"}:
            return False

        for item in intents:
            if isinstance(item, dict) and item.get("type") in {"preference", "preference_management"}:
                return True
        return self._is_explicit_long_term_preference(query)

    def _is_explicit_long_term_preference(self, query: str) -> bool:
        q = query or ""
        explicit_markers = (
            "我喜欢",
            "我不喜欢",
            "我的偏好",
            "我偏好",
            "记住",
            "帮我记住",
            "以后",
            "以后都",
            "长期",
            "平时",
            "通常",
            "每次",
            "下次",
        )
        preference_words = ("偏好", "喜欢", "不喜欢", "预算", "酒店", "交通方式", "节奏", "餐饮", "常住", "住在")
        return any(marker in q for marker in explicit_markers) and any(word in q for word in preference_words)

    def _query_memory(self, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"answer": "当前未启用记忆系统。", "query": query}

        context = context or {}
        prefs = self.memory_manager.long_term.get_preference()
        trips = self.memory_manager.long_term.get_trip_history(limit=20)
        feedback = self.memory_manager.long_term.get_behavior_feedback(limit=20)
        preference_gaps = self._preference_gaps(prefs)
        event_data = context.get("event_data") if isinstance(context.get("event_data"), dict) else {}
        if event_data.get("budget_level") and "budget_level" in preference_gaps:
            preference_gaps.remove("budget_level")
        if event_data.get("pace_preference") and "pace_preference" in preference_gaps:
            preference_gaps.remove("pace_preference")
        blocking_preference_gaps = self._blocking_preference_gaps(preference_gaps)
        follow_up_question = self._build_preference_follow_up(preference_gaps)
        suggested_options = self._build_preference_options(blocking_preference_gaps or preference_gaps[:1])

        if any(word in query for word in ("我是谁", "知道我是谁", "认识我")):
            known = []
            if prefs.get("name") or prefs.get("nickname"):
                known.append(f"你的称呼是{prefs.get('name') or prefs.get('nickname')}")
            else:
                known.append("我目前不知道你的姓名或真实身份")
            if prefs.get("home_location"):
                known.append(f"你常从{prefs['home_location']}出发")
            if prefs.get("hotel_brands"):
                known.append(f"你偏好的酒店品牌包括{self._format_value(prefs['hotel_brands'])}")
            if prefs.get("transportation_preference"):
                known.append(f"你偏好的交通方式是{self._format_value(prefs['transportation_preference'])}")
            if trips:
                destinations = []
                for trip in trips[-5:]:
                    dest = trip.get("destination")
                    if dest and dest not in destinations:
                        destinations.append(dest)
                if destinations:
                    known.append(f"你近期关注或规划过这些目的地：{'、'.join(destinations[:5])}")
            return {
                "query": query,
                "answer": "；".join(known) + "。如果你愿意，可以告诉我你的名字或称呼，我会记住。",
                "preferences": prefs,
                "preference_gaps": preference_gaps,
                "blocking_preference_gaps": blocking_preference_gaps,
                "follow_up_question": follow_up_question,
                "suggested_options": suggested_options,
                "trip_history": trips[:5],
                "behavior_feedback": feedback[:5],
            }

        answer_parts = []
        if prefs:
            pref_lines = []
            for key, value in prefs.items():
                if value:
                    pref_lines.append(f"{key}: {value}")
            if pref_lines:
                answer_parts.append("已记录偏好：" + "；".join(pref_lines))

        if trips:
            trip_lines = []
            for trip in trips[:5]:
                origin = trip.get("origin") or "未知出发地"
                destination = trip.get("destination") or "未知目的地"
                start = trip.get("start_date") or ""
                trip_lines.append(f"{origin}到{destination}{f'（{start}）' if start else ''}")
            answer_parts.append("近期行程：" + "；".join(trip_lines))

        if feedback:
            answer_parts.append("行为反馈：" + "；".join(str(item.get("feedback", item)) for item in feedback[:5]))
        if preference_gaps:
            answer_parts.append("尚未记录：" + "、".join(self._display_preference_name(item) for item in preference_gaps[:4]))
        if follow_up_question:
            answer_parts.append(follow_up_question)

        return {
            "query": query,
            "answer": "；".join(answer_parts) if answer_parts else "目前没有找到相关的长期记忆。",
            "preferences": prefs,
            "preference_gaps": preference_gaps,
            "blocking_preference_gaps": blocking_preference_gaps,
            "follow_up_question": follow_up_question,
            "suggested_options": suggested_options,
            "trip_history": trips[:5],
            "behavior_feedback": feedback[:5],
        }

    def _format_value(self, value: Any) -> str:
        if isinstance(value, list):
            return "、".join(str(item) for item in value if item)
        return str(value)

    def _preference_gaps(self, prefs: Dict[str, Any]) -> List[str]:
        required_for_planning = [
            "budget_level",
            "pace_preference",
            "hotel_brands",
            "transportation_preference",
            "food_preference",
            "interest_tags",
        ]
        gaps = []
        for key in required_for_planning:
            value = prefs.get(key) if isinstance(prefs, dict) else None
            if value in (None, "", [], {}):
                gaps.append(key)
        return gaps

    def _blocking_preference_gaps(self, gaps: List[str]) -> List[str]:
        required_before_plan = {"budget_level", "pace_preference"}
        return [item for item in gaps if item in required_before_plan]

    def _build_preference_follow_up(self, gaps: List[str]) -> str:
        if not gaps:
            return ""
        selected = [self._display_preference_name(item) for item in gaps[:3]]
        return f"如果你愿意，可以补充{('、'.join(selected))}，我后续规划会更贴合你的习惯。"

    def _build_preference_options(self, gaps: List[str]) -> List[Dict[str, str]]:
        options: List[Dict[str, str]] = []
        if "budget_level" in gaps:
            options.extend(
                [
                    {
                        "label": "经济型预算",
                        "message": "我的预算偏好是经济型，住宿每晚300元以内，餐饮和交通尽量节省。",
                    },
                    {
                        "label": "舒适型预算",
                        "message": "我的预算偏好是舒适型，住宿每晚300到600元，兼顾体验和性价比。",
                    },
                    {
                        "label": "品质型预算",
                        "message": "我的预算偏好是品质型，住宿每晚600元以上，优先体验和便利。",
                    },
                ]
            )
        if "pace_preference" in gaps:
            options.extend(
                [
                    {
                        "label": "轻松节奏",
                        "message": "我的行程节奏偏好是轻松，不要太赶，每天安排2到3个核心地点。",
                    },
                    {
                        "label": "均衡节奏",
                        "message": "我的行程节奏偏好是均衡，景点和休息都要兼顾。",
                    },
                    {
                        "label": "紧凑节奏",
                        "message": "我的行程节奏偏好是紧凑，希望尽量多看一些地点。",
                    },
                ]
            )
        return options

    def _display_preference_name(self, key: str) -> str:
        names = {
            "budget_level": "预算偏好",
            "pace_preference": "行程节奏",
            "hotel_brands": "酒店偏好",
            "transportation_preference": "交通偏好",
            "food_preference": "餐饮偏好",
            "interest_tags": "景点兴趣",
        }
        return names.get(key, key)
