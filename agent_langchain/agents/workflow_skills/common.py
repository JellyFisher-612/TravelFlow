"""Common TravelFlow workflow skills.

Each workflow skill owns a reusable workflow plan: which child agents
participate, their order, the execution mode, and the blocking policy. The legacy
``agent_schedule`` is derived from that plan so the current scheduler can keep
running the workflow.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agents.workflow_skills.base import SkillMatch, WorkflowSkill


def _query(state: Dict[str, Any]) -> str:
    query = str(state.get("user_query") or "").strip()
    if query:
        return query
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "").strip()
        content = getattr(last, "content", "")
        return str(content or "").strip()
    return ""


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _schedule(agent_name: str, priority: int, reason: str, expected_output: str) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "priority": priority,
        "reason": reason,
        "expected_output": expected_output,
    }


def _child_agent(agent_name: str, adapter: str, implementation: str, responsibility: str) -> Dict[str, str]:
    return {
        "agent_name": agent_name,
        "adapter": adapter,
        "implementation": implementation,
        "responsibility": responsibility,
    }


def _tool(name: str, provider: str, operations: list[str], purpose: str) -> Dict[str, Any]:
    return {
        "name": name,
        "provider": provider,
        "operations": operations,
        "purpose": purpose,
    }


def _memory_dependency(name: str, operations: list[str], purpose: str) -> Dict[str, Any]:
    return {
        "name": name,
        "operations": operations,
        "purpose": purpose,
    }


def _clarification_agent() -> Dict[str, str]:
    return _child_agent(
        "clarification",
        "agents.travelflow_agents.ClarificationAgent",
        "agents.clarification_agent.EventCollectionAgent",
        "抽取或追问目的地、出发地、日期、天数、预算、节奏等行程字段。",
    )


def _search_agent() -> Dict[str, str]:
    return _child_agent(
        "search",
        "agents.travelflow_agents.SearchAgent",
        "agents.search_agent.InformationQueryAgent",
        "调用外部工具获取天气、POI、路线、车次和网页兜底信息。",
    )


def _plan_agent() -> Dict[str, str]:
    return _child_agent(
        "plan",
        "agents.travelflow_agents.PlanAgent",
        "agents.plan_agent.ItineraryPlanningAgent",
        "整合事项字段、检索数据和记忆上下文生成结构化行程。",
    )


def _amap_tools(operations: list[str], purpose: str) -> Dict[str, Any]:
    return _tool("AmapService", "utils.amap_service", operations, purpose)


def _train_tool() -> Dict[str, Any]:
    return _tool(
        "TrainService",
        "utils.train_service",
        ["get_tickets"],
        "通过 12306 MCP 查询火车/高铁车次、时间、票价和余票。",
    )


def _ddgs_tool(purpose: str = "当结构化工具不足时进行通用网页搜索兜底。") -> Dict[str, Any]:
    return _tool("DDGS", "ddgs / duckduckgo_search", ["text"], purpose)


def _scheduled_workflow_plan(
    name: str,
    description: str,
    steps: list[Dict[str, Any]],
    child_agents: list[Dict[str, Any]],
    tools: list[Dict[str, Any]],
    blocking_policy: Dict[str, Any] | None = None,
    memory_policy: Dict[str, Any] | None = None,
    memory_dependencies: list[Dict[str, Any]] | None = None,
    data_flow: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "execution_mode": "scheduled_agents",
        "description": description,
        "child_agents": child_agents,
        "tools": tools,
        "memory_dependencies": memory_dependencies or [],
        "data_flow": data_flow or [],
        "steps": steps,
        "blocking_policy": blocking_policy or {},
        "memory_policy": memory_policy or {"inject_runtime_context": True},
    }


def _direct_workflow_plan(name: str, description: str, operation: str) -> Dict[str, Any]:
    return {
        "name": name,
        "execution_mode": "main_agent_direct_action",
        "description": description,
        "child_agents": [],
        "tools": [
            _tool(
                "MemoryManager",
                "context.memory_manager",
                ["get_runtime_context", "query_memory", "apply_agent_results"],
                "读取个人偏好、历史行程、行为反馈，并按策略提交显式偏好更新。",
            ),
            _tool(
                "PreferenceAgent",
                "agents.preference_agent",
                ["run"],
                "当 operation=profile_update 时抽取用户显式长期偏好候选。",
            ),
        ],
        "memory_dependencies": [
            _memory_dependency(
                "LongTermMemory",
                ["get_preference", "get_preference_records", "get_trip_history", "get_behavior_feedback", "save_preference"],
                "长期用户画像、历史行程和行为反馈。",
            ),
            _memory_dependency(
                "ShortTermMemory",
                ["get_recent_context"],
                "提供最近对话作为记忆查询上下文。",
            ),
        ],
        "data_flow": [
            "user_query -> MainAgent direct_action",
            "MemoryManager.get_runtime_context -> query/update decision",
            "PreferenceAgent.run -> MemoryManager.apply_agent_results（仅显式长期偏好更新）",
        ],
        "steps": [
            {
                "name": "memory_manager",
                "owner": "MainAgent",
                "operation": operation,
                "reason": "显式个人记忆请求不进入业务智能体调度。",
            }
        ],
        "blocking_policy": {},
        "memory_policy": {"inject_runtime_context": True, "allow_profile_update": operation == "profile_update"},
    }


def _agent_schedule_from_plan(workflow_plan: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "agent_name": step["agent_name"],
            "priority": step["priority"],
            "reason": step.get("reason", ""),
            "expected_output": step.get("expected_output", ""),
        }
        for step in workflow_plan.get("steps", [])
        if isinstance(step, dict) and step.get("agent_name")
    ]


def _planning_workflow_plan(name: str = "travel_planning") -> Dict[str, Any]:
    steps = [
        _schedule("clarification", 1, "提取出发地、目的地、时间、天数、预算和节奏", "结构化行程字段"),
        _schedule("search", 2, "基于目的地、日期和交通需求检索外部旅行信息", "目的地 POI、天气、路线等外部数据"),
        _schedule("plan", 3, "整合事项字段、外部信息和 MainAgent 注入的记忆上下文生成旅行计划", "结构化旅行计划"),
    ]
    steps[0]["required_fields"] = ["destination", "start_date", "duration_days", "budget_level", "pace_preference"]
    steps[1]["requires"] = ["clarification.event_data"]
    steps[1]["tools"] = ["AmapService", "TrainService", "DDGS"]
    steps[2]["requires"] = ["clarification.event_data", "search.search_data", "memory_context"]
    steps[2]["tools"] = []
    return _scheduled_workflow_plan(
        name=name,
        description="行程规划复合 workflow：先收集事项，再检索外部信息，最后生成计划。",
        steps=steps,
        child_agents=[_clarification_agent(), _search_agent(), _plan_agent()],
        tools=[
            _amap_tools(
                [
                    "maps_geo",
                    "maps_weather",
                    "maps_text_search",
                    "maps_around_search",
                    "maps_direction_walking",
                    "maps_direction_driving",
                    "maps_distance",
                ],
                "为规划提供目的地地理编码、天气、POI、周边信息、路线和距离。",
            ),
            _train_tool(),
            _ddgs_tool("门票、预约、开放时间等结构化工具未覆盖的信息使用网页兜底检索。"),
        ],
        memory_dependencies=[
            _memory_dependency(
                "MemoryManager",
                ["get_runtime_context", "apply_agent_results"],
                "规划前注入用户偏好、历史行程和反馈；规划成功后按策略写入 Trip History。",
            ),
            _memory_dependency(
                "ShortTermMemory",
                ["set_pending_plan", "clear_pending_plan", "working_state"],
                "缺字段时保存 pending_plan，补充信息后恢复 workflow。",
            ),
        ],
        data_flow=[
            "user_query -> clarification.event_data",
            "clarification.event_data -> search.search_bundle",
            "memory_context + event_data + search_bundle -> plan.itinerary",
            "completed itinerary -> MemoryManager Trip History policy",
        ],
        blocking_policy={
            "missing_required_fields": "ask_user",
            "search_failure_before_plan": "stop",
            "max_search_refinement_rounds": 1,
        },
        memory_policy={"inject_runtime_context": True, "write_trip_history_after_completed_plan": True},
    )


def _search_workflow_plan(
    name: str,
    description: str,
    search_profile: str,
    reason: str,
    expected_output: str,
    tools: list[Dict[str, Any]],
    data_flow: list[str],
) -> Dict[str, Any]:
    steps = [_schedule("search", 1, reason, expected_output)]
    steps[0]["search_profile"] = search_profile
    steps[0]["tools"] = [tool["name"] for tool in tools]
    return _scheduled_workflow_plan(
        name=name,
        description=description,
        steps=steps,
        child_agents=[_search_agent()],
        tools=tools,
        data_flow=data_flow,
        blocking_policy={"search_failure": "return_failure"},
        memory_policy={"inject_runtime_context": False},
    )


def _extract_trip_entities(query: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}

    destination_tail = r"(?=旅游|旅行|游玩|出游|自由行|度假|玩|[一二两三四五六七八九十\d]+\s*天|，|。|！|？|\s|$)"
    origin_destination = re.search(
        rf"从(?P<origin>[^，。！？\s]{{1,20}}?)(?:出发)?(?:去|到|前往)(?P<destination>[^，。！？\s]{{1,20}}?){destination_tail}",
        query,
    )
    if origin_destination:
        entities["origin"] = origin_destination.group("origin")
        entities["destination"] = origin_destination.group("destination")
    else:
        destination = re.search(rf"(?:去|到|前往)(?P<destination>[^，。！？\s]{{1,20}}?){destination_tail}", query)
        if destination:
            entities["destination"] = destination.group("destination")

    duration = re.search(r"(?P<days>[一二两三四五六七八九十\d]+)\s*天", query)
    if duration:
        entities["duration_days_text"] = duration.group("days")

    if _contains_any(query, ("明天", "后天", "下周", "周末", "五一", "十一", "春节", "暑假", "寒假")):
        entities["date_expression"] = next(
            marker for marker in ("明天", "后天", "下周", "周末", "五一", "十一", "春节", "暑假", "寒假") if marker in query
        )

    if "预算" in query:
        entities["budget_mentioned"] = True
    if _contains_any(query, ("轻松", "紧凑", "慢节奏", "特种兵", "亲子", "老人", "情侣")):
        entities["style_mentioned"] = True

    return entities


def _memory_operation(query: str) -> str:
    query_markers = ("什么", "哪些", "吗", "么", "有没有", "是多少", "怎么看", "查询", "查看", "告诉我")
    if _contains_any(query, query_markers):
        return "query"
    update_markers = ("记住", "帮我记住", "以后", "下次", "长期", "我喜欢", "我不喜欢", "我偏好", "我的偏好是")
    return "profile_update" if _contains_any(query, update_markers) else "query"


class PendingPlanSkill(WorkflowSkill):
    name = "pending_plan_completion"
    priority = 1

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        pending_plan = state.get("pending_plan") if isinstance(state.get("pending_plan"), dict) else None
        if not query or not pending_plan:
            return None

        supplement_markers = (
            "预算",
            "出发",
            "从",
            "日期",
            "时间",
            "天",
            "轻松",
            "紧凑",
            "酒店",
            "交通",
            "人均",
            "带孩子",
            "老人",
        )
        if not _contains_any(query, supplement_markers):
            return None

        base_query = str(pending_plan.get("query") or "").strip()
        rewritten_query = f"{base_query}；补充信息：{query}" if base_query else query
        intention_data = {
            "reasoning": "用户正在补充上一轮被阻断的行程规划信息，直接恢复规划工作流。",
            "intents": [
                {
                    "type": "plan",
                    "confidence": 0.95,
                    "description": "继续未完成的行程规划",
                    "reason": "存在 pending_plan，且本轮输入是行程字段补充。",
                }
            ],
            "key_entities": _extract_trip_entities(query),
            "rewritten_query": rewritten_query,
        }
        workflow_plan = _planning_workflow_plan(self.name)
        intention_data["agent_schedule"] = _agent_schedule_from_plan(workflow_plan)
        intention_data["direct_action"] = {}
        intention_data["workflow_plan"] = workflow_plan
        return SkillMatch(self.name, 0.95, "补充 pending_plan", intention_data, workflow_plan)


class TravelPlanningSkill(WorkflowSkill):
    name = "travel_planning"
    priority = 10

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        if not query:
            return None

        planning_words = ("规划", "安排", "行程", "路线", "旅游", "旅行", "出游", "自由行", "游玩", "玩")
        trip_verbs = ("去", "到", "前往")
        has_planning_word = _contains_any(query, planning_words)
        has_destination_movement = _contains_any(query, trip_verbs) and not _contains_any(query, ("天气", "气温", "预报"))
        has_duration_or_date = bool(re.search(r"[一二两三四五六七八九十\d]+\s*天", query)) or _contains_any(
            query, ("明天", "后天", "下周", "周末", "五一", "十一", "春节", "暑假", "寒假")
        )

        if not (has_planning_word and (has_destination_movement or has_duration_or_date)):
            return None

        workflow_plan = _planning_workflow_plan(self.name)
        intention_data = {
            "reasoning": "用户提出常见行程规划请求，命中规划 workflow skill，按固定层级调度事项收集、检索和规划。",
            "intents": [
                {
                    "type": "plan",
                    "confidence": 0.95,
                    "description": "旅行行程规划",
                    "reason": "用户明确表达出游/旅游/规划需求。",
                }
            ],
            "key_entities": _extract_trip_entities(query),
            "rewritten_query": query,
            "agent_schedule": _agent_schedule_from_plan(workflow_plan),
            "direct_action": {},
            "workflow_plan": workflow_plan,
        }
        return SkillMatch(self.name, 0.95, "常见行程规划请求", intention_data, workflow_plan, intention_data["key_entities"])


class WeatherQuerySkill(WorkflowSkill):
    name = "weather_query"
    priority = 20

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        if not query or not _contains_any(query, ("天气", "气温", "下雨", "预报", "冷不冷", "热不热")):
            return None

        workflow_plan = _search_workflow_plan(
            name=self.name,
            description="天气查询 workflow：直接调用信息检索智能体获取天气。",
            search_profile="weather",
            reason="调用高德 MCP maps_weather 查询天气",
            expected_output="城市天气预报",
            tools=[
                _amap_tools(
                    ["maps_weather"],
                    "根据城市或目的地查询天气、气温和预报。",
                )
            ],
            data_flow=[
                "user_query -> search weather intent",
                "search -> AmapService.maps_weather",
                "weather result -> final_result",
            ],
        )
        intention_data = {
            "reasoning": "用户询问天气，命中天气查询 workflow skill，调度 search 使用外部天气数据。",
            "intents": [
                {
                    "type": "search",
                    "confidence": 0.98,
                    "description": "天气查询",
                    "reason": "用户明确询问天气、气温或预报。",
                }
            ],
            "key_entities": {},
            "rewritten_query": query,
            "agent_schedule": _agent_schedule_from_plan(workflow_plan),
            "direct_action": {},
            "workflow_plan": workflow_plan,
        }
        return SkillMatch(self.name, 0.98, "天气查询", intention_data, workflow_plan)


class TrainQuerySkill(WorkflowSkill):
    name = "train_query"
    priority = 25

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        if not query or not _contains_any(query, ("火车", "高铁", "动车", "城际", "车次", "12306", "余票", "票价", "火车票", "高铁票")):
            return None

        workflow_plan = _search_workflow_plan(
            name=self.name,
            description="铁路查询 workflow：直接调用信息检索智能体使用 12306 数据。",
            search_profile="rail",
            reason="调用 12306 MCP 查询车次、时间、票价或余票",
            expected_output="铁路车次时间、余票与票价信息",
            tools=[_train_tool()],
            data_flow=[
                "user_query -> search rail intent",
                "search -> TrainService.get_tickets",
                "tickets -> final_result",
            ],
        )
        intention_data = {
            "reasoning": "用户询问铁路出行信息，命中车次查询 workflow skill，调度 search 使用 12306 数据。",
            "intents": [
                {
                    "type": "search",
                    "confidence": 0.98,
                    "description": "火车车次查询",
                    "reason": "用户明确询问车次、票价、余票或铁路交通。",
                }
            ],
            "key_entities": {},
            "rewritten_query": query,
            "agent_schedule": _agent_schedule_from_plan(workflow_plan),
            "direct_action": {},
            "workflow_plan": workflow_plan,
        }
        return SkillMatch(self.name, 0.98, "铁路信息查询", intention_data, workflow_plan)


class MemoryProfileSkill(WorkflowSkill):
    name = "memory_profile"
    priority = 30

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        if not query:
            return None

        identity_memory = _contains_any(query, ("我是谁", "知道我是谁", "认识我"))
        personal_markers = ("我的", "我过去", "我以前", "我历史", "我去过", "我喜欢", "我不喜欢", "我偏好", "我常")
        memory_targets = ("偏好", "喜好", "历史", "行程", "去过", "记录", "记得", "预算", "节奏", "酒店", "交通方式", "常住")
        is_personal_memory = _contains_any(query, personal_markers) and _contains_any(query, memory_targets)

        if not identity_memory and not is_personal_memory:
            return None

        operation = _memory_operation(query)
        workflow_plan = _direct_workflow_plan(self.name, "偏好与个人历史 workflow：由 MainAgent 驱动 MemoryManager。", operation)
        intention_data = {
            "reasoning": "用户询问或更新自己的长期偏好、历史行程或身份记忆，由 MainAgent 驱动 MemoryManager 处理。",
            "intents": [
                {
                    "type": "memory",
                    "confidence": 0.98,
                    "description": "个人记忆/偏好处理",
                    "reason": "请求对象是用户自己的偏好、历史或已记录信息。",
                }
            ],
            "key_entities": {},
            "rewritten_query": query,
            "direct_action": {
                "type": "memory",
                "operation": operation,
                "reason": "读取或更新用户长期偏好、历史行程和行为反馈",
            },
            "agent_schedule": [],
            "workflow_plan": workflow_plan,
        }
        return SkillMatch(self.name, 0.98, "个人记忆请求", intention_data, workflow_plan)


class InformationQuerySkill(WorkflowSkill):
    name = "information_query"
    priority = 90

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        query = _query(state)
        if not query:
            return None

        info_markers = (
            "怎么样",
            "有什么好玩",
            "推荐",
            "攻略",
            "门票",
            "开放时间",
            "地址",
            "路线",
            "附近",
            "查一下",
            "查查",
            "介绍一下",
            "在哪里",
            "怎么去",
        )
        if not _contains_any(query, info_markers):
            return None

        workflow_plan = _search_workflow_plan(
            name=self.name,
            description="通用旅行信息查询 workflow：调用 search 获取可核验信息。",
            search_profile="travel_info",
            reason="检索目的地、景点、门票、开放时间、路线或攻略信息",
            expected_output="可核验的信息查询结果",
            tools=[
                _amap_tools(
                    ["maps_geo", "maps_text_search", "maps_around_search", "maps_search_detail", "maps_direction_walking", "maps_direction_driving"],
                    "查询地点、景点、周边设施、详情和基础路线。",
                ),
                _ddgs_tool("查询门票、开放时间、预约规则、攻略等网页信息兜底。"),
            ],
            data_flow=[
                "user_query -> search travel_info intent",
                "search -> AmapService structured POI/route lookup",
                "search -> DDGS fallback when needed",
                "verified information -> final_result",
            ],
        )
        intention_data = {
            "reasoning": "用户提出目的地、景点或交通信息查询，命中通用信息查询 workflow skill。",
            "intents": [
                {
                    "type": "search",
                    "confidence": 0.86,
                    "description": "通用旅行信息查询",
                    "reason": "用户询问地点、景点、攻略、门票、开放时间或路线信息。",
                }
            ],
            "key_entities": {},
            "rewritten_query": query,
            "agent_schedule": _agent_schedule_from_plan(workflow_plan),
            "direct_action": {},
            "workflow_plan": workflow_plan,
        }
        return SkillMatch(self.name, 0.86, "通用旅行信息查询", intention_data, workflow_plan)


DEFAULT_WORKFLOW_SKILLS: tuple[WorkflowSkill, ...] = (
    PendingPlanSkill(),
    TravelPlanningSkill(),
    WeatherQuerySkill(),
    TrainQuerySkill(),
    MemoryProfileSkill(),
    InformationQuerySkill(),
)
