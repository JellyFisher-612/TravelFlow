"""MainAgent 的意图识别能力。

该模块负责理解用户输入并生成 agent_schedule。它不是独立面向用户的
业务智能体，而是 MainAgent 内部的认知/路由能力。
"""

from __future__ import annotations

import json
import logging
import importlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from utils.skill_loader import SkillLoader
from utils.langchain_runtime import ainvoke_text, message_to_dict, to_lc_messages
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

logger = logging.getLogger(__name__)

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")


class IntentItem(BaseModel):
    type: str = Field(default="search", description="意图类型")
    confidence: float = Field(default=0.5, description="置信度，0-1")
    description: str = Field(default="", description="意图说明")
    reason: str = Field(default="", description="识别原因")


class AgentScheduleItem(BaseModel):
    agent_name: str = Field(default="search", description="子智能体名称")
    priority: int = Field(default=1, description="优先级")
    reason: str = Field(default="默认查询", description="调用原因")
    expected_output: str = Field(default="查询结果", description="期望输出")


class IntentionOutput(BaseModel):
    reasoning: str = Field(default="", description="推理过程")
    intents: List[IntentItem] = Field(default_factory=list)
    key_entities: Dict[str, Any] = Field(default_factory=dict)
    rewritten_query: str = Field(default="", description="标准化查询")
    agent_schedule: List[AgentScheduleItem] = Field(default_factory=list)
    direct_answer: str = Field(default="", description="无需调度业务智能体时的直接回复")
    direct_action: Dict[str, Any] = Field(default_factory=dict, description="MainAgent 内部动作，不进入业务智能体调度")


class IntentRecognition:
    """主智能体的意图识别与任务计划能力（LangChain Structured Output 版本）。"""

    def __init__(self, name: str = "IntentRecognition", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.conversation_history: List[str] = []
        self.skill_loader = SkillLoader()
        self._valid_agent_names = {
            "memory",
            "search",
            "plan",
            "clarification",
        }

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        user_query = state.get("user_query", "")
        pending_plan = state.get("pending_plan") if isinstance(state.get("pending_plan"), dict) else None

        if messages:
            lc_messages = to_lc_messages(messages)
            user_query = str(lc_messages[-1].content) if lc_messages else user_query
            self.conversation_history = []
            for msg in lc_messages[:-1]:
                msg_dict = message_to_dict(msg)
                if msg_dict["role"] == "system":
                    self.conversation_history.append(f"[系统记忆]\n{msg_dict['content']}")
                else:
                    role_name = "用户" if msg_dict["role"] == "user" else "助手"
                    content = msg_dict["content"][:800] if len(msg_dict["content"]) > 800 else msg_dict["content"]
                    if len(msg_dict["content"]) > 800:
                        content += "..."
                    self.conversation_history.append(f"{role_name}: {content}")

        direct_result = self._try_pending_plan_completion(user_query, pending_plan)
        if direct_result:
            return {**state, "intention_data": direct_result}

        direct_result = self._try_direct_intent(user_query)
        if direct_result:
            return {**state, "intention_data": direct_result}

        context_str = self._build_context_str()
        now = datetime.now()
        current_time = now.strftime("%Y年%m月%d日 %H:%M")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]

        skill_mapping = {
            "memory-query": "memory",
            "preference": "memory",
            "query-info": "search",
            "plan-trip": "plan",
        }
        dynamic_skills_prompt = self.skill_loader.get_skill_prompt(skill_mapping)

        prompt = f"""你是 TravelFlow 旅游出行助手 MainAgent 内部的意图识别能力（IntentRecognition）。请分析用户查询并返回结构化调度决策。

【当前时间】
{current_time} {weekday}
（重要：当用户说"2月28日"或"明天"等相对时间时，请根据当前时间推断完整日期）

【用户Query】
{user_query}

【对话历史上下文】
{context_str}

【TravelFlow 可调度子智能体】
{dynamic_skills_prompt}

固定只能调度以下三类业务智能体；MainAgent、IntentRecognition、AgentScheduler 和记忆系统都不是业务智能体，不要放入 agent_schedule：
- search：信息检索智能体，通过高德开放平台 API 获取景点、天气、路线、POI 等外部数据。
- plan：行程规划智能体，基于用户偏好、外部数据和行程要素生成个性化旅行计划。
- clarification：事项收集智能体，在需求不明确时提取/追问目的地、时间、预算、同行人等关键字段。

显式记忆查询或长期偏好更新不是业务子智能体调度。遇到“我的偏好/历史/我是谁/记住/以后默认”等请求时：
- intents 使用 type="memory"
- agent_schedule 必须为空
- direct_action 使用 {{"type": "memory", "operation": "query" | "profile_update"}}

【重要 - 意图区分原则】
请基于语义理解判断意图，不要机械匹配关键词。
- "我去过北京吗？" → memory
- "以后我都喜欢轻松一点的行程" / "帮我记住我的预算不高" → memory
- "北京怎么样？" / "北京有什么好玩的？" / "杭州明天天气怎么样？" → search
- "我想去北京玩三天" → clarification → search → plan

优先级规则：
- 如果用户明确询问"我的"、"我过去的"，必须识别为 memory，并返回 direct_action，不调度业务智能体
- 需要规划行程时，必须按层级调度：clarification 先收集字段；search 检索外部信息；最后 plan 生成方案。用户长期偏好和历史由 MainAgent 注入上下文，不要把 memory 放入规划链路。

【多轮对话与更正规则】
你必须结合【对话历史上下文】理解当前这句话是在纠正/补充之前的意图，还是开启一个全新的话题：

1. 更正之前的行程要素
- 如果上一轮或最近几轮正在讨论出行/行程规划（例如之前识别过 plan，或用户刚说「我要去A」「从B到A」等），
    而当前用户只说类似：
    - "我在C点" / "出发地是C" / "不是B，我在C" / "改成从C出发" / "目的地改成A" 等，
    那么：
    - 这应视为**对现有行程要素的补充或修正**，而不是新的信息查询；
    - intents 中的主意图应为 plan（可同时保留 clarification 以便重新收集字段）；
    - key_entities 中需要更新 origin/destination 等字段，覆盖掉之前错误的值；
    - rewritten_query 必须整合历史语境，生成一条**自洽、完整**的出行需求描述（包含最新的出发地/目的地/日期等），而不是仅重复"我在C点"。

2. 单独地点/日期陈述的处理
- 当用户只说一个地点或日期（如"我在C点"、"出发地是上海"、"时间改到5月1号"），
    - 如果【对话历史上下文】中最近的话题是行程规划，则默认这是对该行程的补充/修改，继续走 clarification/search/plan 流程；
    - 只有在**没有任何行程相关上下文**，且用户明确是在问天气/信息（如"C点天气怎么样"、"帮我查一下C点"）时，才将其归为 search。

3. rewritten_query 的职责
- 在多轮对话里，rewritten_query 不是简单地重复本轮话语，而是要：
    - 综合对话历史中的关键信息（出发地、目的地、日期、行程目的等）；
    - 应用用户本轮的补充/更正；
    - 输出用一句话就能看懂的、完整的旅行需求描述；
例如：
    - 历史："我要从上海去北京玩两天" → 模型误以为出发地是广州；
    - 用户本轮："我在苏州"；
    - 正确 rewritten_query 应类似："用户想从苏州出发，前往北京游玩2天，请规划行程"。

【任务要求】
请返回以下结构字段：
1. reasoning: 推理过程
2. intents: 识别的多意图（包含 type, confidence, description, reason）
3. key_entities: 关键实体
4. rewritten_query: 标准化查询
5. agent_schedule: 业务智能体调度策略（agent_name, priority, reason, expected_output）
6. direct_action: MainAgent 内部动作；显式记忆请求使用 {{"type": "memory", "operation": "query" | "profile_update"}}，普通业务调度留空

【优先级设置规则】
优先级数字相同的智能体会并行执行，不同优先级按顺序执行。

Priority 1:
- clarification（规划类请求先收集基础字段）

Priority 2:
- search（外部信息检索，依赖 clarification 的目的地、日期等信息调用高德 API）

Priority 3:
- plan（依赖 MainAgent 注入的记忆上下文、clarification 和 search 的结果生成完整行程）

请确保输出可被结构化解析，并尽量完整。"""

        try:
            result = await self._invoke_structured(prompt)
        except Exception as e:
            logger.error("Intent recognition failed: %s", e)
            result = self._build_error_fallback_intention(user_query, e)

        result_dict = result.model_dump()
        if not result_dict.get("rewritten_query"):
            result_dict["rewritten_query"] = str(user_query)

        if self._is_conversation_intent(result_dict, user_query):
            result_dict["intents"] = result_dict.get("intents") or [
                {
                    "type": "conversation",
                    "confidence": 1.0,
                    "description": "普通对话",
                    "reason": "不需要调用业务智能体。",
                }
            ]
            result_dict["direct_answer"] = result_dict.get("direct_answer") or self._build_direct_conversation_answer(user_query)
            result_dict["agent_schedule"] = []
            return {**state, "intention_data": result_dict}

        if not result_dict.get("intents"):
            result_dict["intents"] = [
                {
                    "type": "search",
                    "confidence": 0.5,
                    "description": "默认查询意图",
                    "reason": "空意图结果，使用默认策略",
                }
            ]
        result_dict["agent_schedule"] = self._sanitize_agent_schedule(result_dict.get("agent_schedule", []))
        result_dict["agent_schedule"] = self._normalize_layered_planning_schedule(result_dict)
        if self._is_conversation_intent(result_dict, user_query):
            result_dict["direct_answer"] = result_dict.get("direct_answer") or self._build_direct_conversation_answer(user_query)
            result_dict["agent_schedule"] = []
            return {**state, "intention_data": result_dict}

        if not result_dict.get("agent_schedule"):
            result_dict["agent_schedule"] = [
                {
                    "agent_name": "search",
                    "priority": 1,
                    "reason": "默认查询",
                    "expected_output": "查询结果",
                }
            ]

        return {**state, "intention_data": result_dict}

    def _normalize_layered_planning_schedule(self, result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        schedule = result_dict.get("agent_schedule") or []
        if not self._is_planning_intention(result_dict):
            return schedule

        by_name = {item.get("agent_name"): item for item in schedule if isinstance(item, dict)}
        layered = [
            {
                **by_name.get("clarification", {}),
                "agent_name": "clarification",
                "priority": 1,
                "reason": by_name.get("clarification", {}).get("reason", "提取出发地、目的地、时间和天数"),
                "expected_output": by_name.get("clarification", {}).get("expected_output", "结构化行程字段"),
            },
            {
                **by_name.get("search", {}),
                "agent_name": "search",
                "priority": 2,
                "reason": by_name.get("search", {}).get("reason", "调用高德 MCP 检索外部旅行信息"),
                "expected_output": by_name.get("search", {}).get("expected_output", "目的地 POI、天气、路线等外部数据"),
            },
            {
                **by_name.get("plan", {}),
                "agent_name": "plan",
                "priority": 3,
                "reason": by_name.get("plan", {}).get("reason", "整合事项字段、外部信息和 MainAgent 注入的记忆上下文生成旅行计划"),
                "expected_output": by_name.get("plan", {}).get("expected_output", "结构化旅行计划"),
            },
        ]
        return layered

    def _is_planning_intention(self, result_dict: Dict[str, Any]) -> bool:
        intents = result_dict.get("intents") or []
        for item in intents:
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() in {"plan", "travel_plan", "itinerary", "itinerary_planning"}:
                return True
        schedule = result_dict.get("agent_schedule") or []
        return any(isinstance(item, dict) and item.get("agent_name") == "plan" for item in schedule)

    def _is_conversation_intent(self, result_dict: Dict[str, Any], user_query: str) -> bool:
        """Return True for meta/chat turns that the main agent should answer itself."""
        query = (user_query or "").strip().lower()
        if self._is_direct_conversation_query(query):
            return True

        intents = result_dict.get("intents") or []
        conversation_types = {
            "conversation",
            "chat",
            "greeting",
            "identity",
            "capability",
            "greeting_or_identity",
            "assistant_identity",
            "smalltalk",
            "none",
        }
        business_types = {"search", "plan", "clarification", "memory", "preference", "weather"}
        normalized_types = []
        for item in intents:
            if not isinstance(item, dict):
                continue
            intent_type = str(item.get("type", "")).strip().lower()
            if intent_type:
                normalized_types.append(intent_type)

        if normalized_types and all(intent_type in conversation_types for intent_type in normalized_types):
            return True
        if normalized_types and any(intent_type in conversation_types for intent_type in normalized_types):
            if not any(intent_type in business_types for intent_type in normalized_types):
                return True
        return False

    def _is_direct_conversation_query(self, query: str) -> bool:
        if not query:
            return False
        normalized_query = self._normalize_direct_query(query)
        if any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我")):
            return False
        exact_queries = {
            "你好",
            "您好",
            "hello",
            "hi",
            "嗨",
            "你是谁",
            "你是啥",
            "你叫什么",
            "你叫什么名字",
            "你是什么",
            "你能做什么",
            "你会做什么",
            "你可以做什么",
            "你有什么功能",
            "怎么用",
            "在吗",
            "在不在",
            "有人吗",
            "谢谢",
            "谢谢你",
            "多谢",
            "感谢",
            "thanks",
            "thankyou",
            "thx",
            "好的",
            "好",
            "ok",
            "okay",
            "嗯",
            "明白",
            "收到",
        }
        if normalized_query in exact_queries:
            return True
        return any(
            phrase in query
            for phrase in (
                "介绍一下你自己",
                "你是一个什么",
                "你是干什么的",
                "这个系统怎么用",
            )
        )

    def _build_direct_conversation_answer(self, user_query: str) -> str:
        query = (user_query or "").strip().lower()
        normalized_query = self._normalize_direct_query(query)
        if any(phrase in query for phrase in ("你能做什么", "你会做什么", "你可以做什么", "你有什么功能", "怎么用")):
            return (
                "我可以帮你做旅行规划：收集目的地、出发地、时间、预算和同行人信息；"
                "查询高德 MCP 提供的天气、地点和路线数据；结合你的历史偏好生成个性化行程。"
                "你可以直接说：帮我规划下周从上海去北京玩三天。"
            )
        if normalized_query in {"你好", "您好", "hello", "hi", "嗨"}:
            return "你好，我是 TravelFlow 旅游出行助手。你可以直接告诉我想去哪里、从哪里出发、什么时候去和玩几天。"
        if normalized_query in {"在吗", "在不在", "有人吗"}:
            return "我在。你可以告诉我目的地、出发地、时间、天数和偏好，我来帮你规划行程。"
        if normalized_query in {"谢谢", "谢谢你", "多谢", "感谢", "thanks", "thankyou", "thx"}:
            return "不客气。后续如果要继续调整目的地、时间、预算或节奏，可以直接告诉我。"
        if normalized_query in {"好的", "好", "ok", "okay", "嗯", "明白", "收到"}:
            return "好的。你可以继续补充目的地、出发地、时间、天数、预算或偏好。"
        return (
            "我是 TravelFlow 旅游出行助手，一个基于 LangChain/LangGraph 多智能体架构的旅行规划系统。"
            "我可以帮你收集出行意向、查询目的地信息和天气、结合你的偏好生成行程，也能记住你的长期旅行偏好。"
        )

    def _normalize_direct_query(self, query: str) -> str:
        """Normalize short meta/chat queries for deterministic routing."""
        normalized = (query or "").strip().lower()
        normalized = re.sub(r"[\s\ufeff\u200b]+", "", normalized)
        normalized = re.sub(r"[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+$", "", normalized)
        normalized = re.sub(r"^[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+", "", normalized)
        return normalized

    def _build_error_fallback_intention(self, user_query: str, error: Exception) -> IntentionOutput:
        """Use deterministic routing when the LLM intent recognizer is unavailable."""
        direct_result = self._try_direct_intent(user_query)
        if direct_result:
            return IntentionOutput.model_validate(
                {
                    **direct_result,
                    "reasoning": f"{direct_result.get('reasoning', '')}（LLM 意图识别不可用，使用规则兜底。错误: {error}）",
                }
            )

        query = (user_query or "").strip()
        if not query:
            return IntentionOutput(
                reasoning=f"意图识别出错且用户输入为空，直接提示用户补充需求。错误: {error}",
                intents=[
                    IntentItem(
                        type="conversation",
                        confidence=1.0,
                        description="空输入",
                        reason="没有可用于搜索或规划的用户需求。",
                    )
                ],
                key_entities={},
                rewritten_query=query,
                direct_answer="我在。你可以告诉我想去哪里、从哪里出发、什么时候去和玩几天。",
                agent_schedule=[],
            )

        return IntentionOutput(
            reasoning=f"意图识别出错，使用保守信息查询兜底。错误: {error}",
            intents=[
                IntentItem(
                    type="search",
                    confidence=0.4,
                    description="保守信息查询兜底",
                    reason="无法调用 LLM 识别意图，且规则未命中直接回复、天气、车次或规划请求。",
                )
            ],
            key_entities={},
            rewritten_query=query,
            agent_schedule=[
                AgentScheduleItem(
                    agent_name="search",
                    priority=1,
                    reason="保守信息查询兜底",
                    expected_output="查询结果",
                )
            ],
        )

    def _try_direct_intent(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Handle deterministic meta-intents before calling the LLM."""

        query = (user_query or "").strip()
        if not query:
            return None

        identity_questions = (
            "你是谁",
            "你是啥",
            "你叫什么",
            "你叫什么名字",
            "介绍一下你自己",
            "你是什么",
        )
        if any(question in query for question in identity_questions) and not any(
            phrase in query for phrase in ("我是谁", "知道我是谁", "认识我")
        ):
            return {
                "reasoning": "用户在询问当前助手身份，属于对话型元问题，应由系统直接回答，不需要调用外部搜索。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "系统身份问答",
                        "reason": "用户询问 TravelFlow 助手是谁。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": (
                    "我是 TravelFlow 旅游出行助手，一个基于 LangChain/LangGraph 多智能体架构的旅行规划系统。"
                    "我可以帮你收集出行意向、查询目的地信息和天气、结合你的偏好生成行程，也能记住你的长期旅行偏好。"
                ),
                "agent_schedule": [],
            }

        greeting_words = ("你好", "您好", "hello", "hi", "嗨")
        if self._normalize_direct_query(query) in greeting_words:
            return {
                "reasoning": "用户只是打招呼，属于普通对话，不需要调用外部搜索或规划智能体。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "普通问候",
                        "reason": "用户发送问候语。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": "你好，我是 TravelFlow 旅游出行助手。你可以告诉我目的地、出发地、时间、天数和偏好，我来帮你规划行程。",
                "agent_schedule": [],
            }

        presence_queries = ("在吗", "在不在", "有人吗")
        if self._normalize_direct_query(query) in presence_queries:
            return {
                "reasoning": "用户在确认助手是否在线，属于普通对话，不需要调用外部搜索或规划智能体。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "在线确认",
                        "reason": "用户询问助手是否在。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": "我在。你可以告诉我目的地、出发地、时间、天数和偏好，我来帮你规划行程。",
                "agent_schedule": [],
            }

        thanks_queries = ("谢谢", "谢谢你", "多谢", "感谢", "thanks", "thankyou", "thx")
        if self._normalize_direct_query(query) in thanks_queries:
            return {
                "reasoning": "用户表达感谢，属于普通对话，不需要调用外部搜索或规划智能体。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "感谢回应",
                        "reason": "用户表达感谢。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": "不客气。后续如果要继续调整目的地、时间、预算或节奏，可以直接告诉我。",
                "agent_schedule": [],
            }

        confirmation_queries = ("好的", "好", "ok", "okay", "嗯", "明白", "收到")
        if self._normalize_direct_query(query) in confirmation_queries:
            return {
                "reasoning": "用户只是确认或承接上一轮对话，属于普通对话，不需要调用外部搜索或规划智能体。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "确认回应",
                        "reason": "用户发送简短确认。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": "好的。你可以继续补充目的地、出发地、时间、天数、预算或偏好。",
                "agent_schedule": [],
            }

        current_trip_constraints = self._try_current_trip_constraints(query)
        if current_trip_constraints:
            return current_trip_constraints

        memory_result = self._try_direct_memory_intent(query)
        if memory_result:
            return memory_result

        constraints_result = self._try_ambiguous_trip_constraints(query)
        if constraints_result:
            return constraints_result

        capability_questions = (
            "你能做什么",
            "你会做什么",
            "你可以做什么",
            "怎么用",
            "你有什么功能",
        )
        if any(question in query for question in capability_questions):
            return {
                "reasoning": "用户询问系统能力，属于产品能力说明，应直接回答并引导用户继续对话。",
                "intents": [
                    {
                        "type": "conversation",
                        "confidence": 1.0,
                        "description": "能力说明",
                        "reason": "用户询问 TravelFlow 能提供哪些帮助。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": (
                    "我可以帮你做旅行规划：收集目的地、出发地、时间、预算和同行人信息；"
                    "查询高德 MCP 提供的天气、地点和路线数据；结合你的历史偏好生成个性化行程。"
                    "你可以直接说：帮我规划下周从上海去北京玩三天。"
                ),
                "agent_schedule": [],
            }

        asks_required_fields = (
            ("需要" in query or "要" in query)
            and any(word in query for word in ("提供", "填写", "告诉", "补充"))
            and any(word in query for word in ("什么", "哪些", "啥"))
            and any(word in query for word in ("意向", "信息", "内容", "资料", "字段"))
        )
        if asks_required_fields:
            return {
                "reasoning": "用户在询问规划旅行前需要提供哪些信息，属于事项收集说明，不应查询或更新记忆。",
                "intents": [
                    {
                        "type": "clarification",
                        "confidence": 1.0,
                        "description": "说明旅行规划所需信息",
                        "reason": "用户询问需要提供哪些意向/信息。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [
                    {
                        "agent_name": "clarification",
                        "priority": 1,
                        "reason": "告知用户需要补充的旅行规划信息",
                        "expected_output": "需要用户提供的信息清单",
                    }
                ],
            }

        if any(word in query for word in ("火车", "高铁", "动车", "城际", "车次", "12306", "余票", "票价", "火车票", "高铁票")):
            return {
                "reasoning": "用户询问火车/高铁车次、时间、票价或余票，直接调度信息检索智能体使用 12306 MCP。",
                "intents": [
                    {
                        "type": "search",
                        "confidence": 1.0,
                        "description": "火车车次查询",
                        "reason": "用户明确询问铁路出行信息。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [
                    {
                        "agent_name": "search",
                        "priority": 1,
                        "reason": "调用 12306 MCP 查询车次、时间、票价或余票",
                        "expected_output": "铁路车次时间、余票与票价信息",
                    }
                ],
            }

        if any(word in query for word in ("天气", "气温", "下雨", "预报")):
            return {
                "reasoning": "用户询问天气，直接调度信息检索智能体使用高德 MCP maps_weather。",
                "intents": [
                    {
                        "type": "search",
                        "confidence": 1.0,
                        "description": "天气查询",
                        "reason": "用户明确询问天气或天气预报。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [
                    {
                        "agent_name": "search",
                        "priority": 1,
                        "reason": "调用高德 MCP maps_weather 查询天气",
                        "expected_output": "城市天气预报",
                    }
                ],
            }

        planning_words = ("规划", "安排", "行程", "路线", "旅游", "玩", "旅行")
        has_trip_movement = ("从" in query and any(word in query for word in ("去", "到", "前往"))) or any(
            word in query for word in ("三天", "两天", "一天", "下周", "明天", "后天")
        )
        if any(word in query for word in planning_words) and has_trip_movement:
            return {
                "reasoning": "用户提出明确行程规划请求，按层级执行：先收集行程字段，再检索外部信息，最后结合 MainAgent 注入的记忆上下文生成计划。",
                "intents": [
                    {
                        "type": "plan",
                        "confidence": 1.0,
                        "description": "旅行行程规划",
                        "reason": "用户明确要求规划行程。",
                    }
                ],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [
                    {
                        "agent_name": "clarification",
                        "priority": 1,
                        "reason": "提取出发地、目的地、时间和天数",
                        "expected_output": "结构化行程字段",
                    },
                    {
                        "agent_name": "search",
                        "priority": 2,
                        "reason": "基于目的地调用高德 API 检索景点、天气和路线",
                        "expected_output": "目的地 POI、天气、路线等外部数据",
                    },
                    {
                        "agent_name": "plan",
                        "priority": 3,
                        "reason": "整合事项字段、外部信息和 MainAgent 注入的记忆上下文生成旅行计划",
                        "expected_output": "三天结构化旅行计划",
                    },
                ],
            }

        return None

    def _try_direct_memory_intent(self, query: str) -> Optional[Dict[str, Any]]:
        personal_markers = ("我的", "我过去", "我以前", "我历史", "我去过", "我喜欢", "我不喜欢", "我偏好", "我常")
        memory_targets = ("偏好", "喜好", "历史", "行程", "去过", "记录", "记得", "预算", "节奏", "酒店", "交通方式", "常住")
        identity_memory = any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我"))
        is_personal_memory_query = any(marker in query for marker in personal_markers) and any(
            target in query for target in memory_targets
        )
        is_preference_statement = any(phrase in query for phrase in ("我喜欢", "我不喜欢", "我偏好", "我的预算", "我的行程节奏"))

        if not identity_memory and not is_personal_memory_query and not is_preference_statement:
            return None

        operation = "profile_update" if is_preference_statement else "query"
        return {
            "reasoning": "用户询问或补充自己的长期偏好、历史行程或身份记忆，由 MainAgent 内部记忆能力处理，不调度业务子智能体。",
            "intents": [
                {
                    "type": "memory",
                    "confidence": 1.0,
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
        }

    def _try_current_trip_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        constraint_words = (
            "预算",
            "经济型",
            "舒适型",
            "品质型",
            "住宿",
            "酒店",
            "每晚",
            "餐饮",
            "吃饭",
            "交通",
            "节省",
            "省钱",
            "轻松",
            "均衡",
            "紧凑",
            "节奏",
        )
        current_trip_markers = (
            "本次行程",
            "这次行程",
            "此次行程",
            "本次旅行",
            "这次旅行",
            "此次旅行",
            "这趟",
            "这一次",
            "这次",
        )
        long_term_markers = ("以后", "长期", "平时", "通常", "一直", "默认", "记住")

        if not any(word in query for word in constraint_words):
            return None
        if not any(marker in query for marker in current_trip_markers):
            return None
        if any(marker in query for marker in long_term_markers):
            return None

        key_entities: Dict[str, Any] = {"trip_scope": "current"}
        if any(word in query for word in ("经济型", "省钱", "节省")):
            key_entities["budget_level"] = "经济型"
        elif "舒适型" in query:
            key_entities["budget_level"] = "舒适型"
        elif "品质型" in query:
            key_entities["budget_level"] = "品质型"

        lodging_range = re.search(
            r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*(?:到|至|-|~|－|—)\s*(\d+)\s*元?",
            query,
        )
        if lodging_range:
            low = int(lodging_range.group(1))
            high = int(lodging_range.group(2))
            key_entities["lodging_budget_per_night_min"] = min(low, high)
            key_entities["lodging_budget_per_night_max"] = max(low, high)
        else:
            lodging_budget = re.search(r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*元?(?:以内|以下|内)?", query)
            if lodging_budget:
                key_entities["lodging_budget_per_night"] = int(lodging_budget.group(1))
                key_entities["lodging_budget_per_night_max"] = int(lodging_budget.group(1))

        if any(word in query for word in ("轻松", "慢节奏")):
            key_entities["pace_preference"] = "轻松"
        elif "紧凑" in query:
            key_entities["pace_preference"] = "紧凑"
        elif "均衡" in query:
            key_entities["pace_preference"] = "均衡"

        if any(word in query for word in ("餐饮", "吃饭")) and any(word in query for word in ("节省", "省钱")):
            key_entities["meal_budget_preference"] = "节省"
        if "交通" in query and any(word in query for word in ("节省", "省钱")):
            key_entities["transport_budget_preference"] = "节省"

        return {
            "reasoning": "用户明确在补充本次行程的预算、住宿、餐饮、交通或节奏约束，应作为当前行程规划字段处理，而不是外部信息查询。",
            "intents": [
                {
                    "type": "plan",
                    "confidence": 1.0,
                    "description": "本次行程约束补充",
                    "reason": "输入包含“本次/这次行程”等当前行程标记和预算/住宿/餐饮/交通/节奏约束。",
                }
            ],
            "key_entities": key_entities,
            "rewritten_query": query,
            "agent_schedule": [
                {
                    "agent_name": "clarification",
                    "priority": 1,
                    "reason": "提取并合并本次行程的预算、住宿、餐饮、交通和节奏约束",
                    "expected_output": "更新后的结构化行程字段",
                },
                {
                    "agent_name": "search",
                    "priority": 2,
                    "reason": "在行程字段完整后检索符合预算约束的外部旅行信息",
                    "expected_output": "目的地 POI、路线、天气和预算相关外部数据",
                },
                {
                    "agent_name": "plan",
                    "priority": 3,
                    "reason": "按最新预算、节省约束和 MainAgent 注入的记忆上下文生成或调整行程计划",
                    "expected_output": "符合约束的结构化旅行计划",
                },
            ],
        }

    def _try_ambiguous_trip_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        has_budget_or_pace = any(word in query for word in ("经济型", "舒适型", "品质型", "预算", "轻松", "均衡", "紧凑", "节奏"))
        has_trip_anchor = any(word in query for word in ("从", "去", "到", "前往", "出发", "玩", "旅游", "行程", "天", "月", "日"))
        has_long_term_marker = any(word in query for word in ("我喜欢", "我不喜欢", "我的偏好", "记住", "以后", "长期", "平时", "通常"))
        if not has_budget_or_pace or has_trip_anchor or has_long_term_marker:
            return None

        return {
            "reasoning": "用户只提供了预算/节奏等约束，但没有说明这是本次行程信息还是长期偏好，需要先确认。",
            "intents": [
                {
                    "type": "conversation",
                    "confidence": 1.0,
                    "description": "澄清约束用途",
                    "reason": "预算和节奏可能是本次行程约束，也可能是长期偏好。",
                }
            ],
            "key_entities": {},
            "rewritten_query": query,
            "direct_answer": (
                "你说的“"
                + query
                + "”是这次行程的要求，还是希望我保存为长期偏好？"
                "如果是这次行程，请告诉我目的地、出发地、日期和天数；"
                "如果是长期偏好，可以说“以后都按经济型预算、轻松节奏来安排”。"
            ),
            "agent_schedule": [],
        }

    def _try_pending_plan_completion(self, user_query: str, pending_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pending_plan:
            return None
        query = (user_query or "").strip()
        pending_query = str(pending_plan.get("query") or "").strip()
        if not query or not pending_query:
            return None
        supplement_markers = (
            "预算",
            "经济",
            "舒适",
            "品质",
            "节奏",
            "轻松",
            "均衡",
            "紧凑",
            "出发",
            "从",
            "在",
            "去",
            "到",
            "目的地",
            "日期",
            "时间",
            "明天",
            "后天",
            "下周",
            "月",
            "日",
            "天",
        )
        if not any(word in query for word in supplement_markers):
            return None

        return {
            "reasoning": "用户正在补充上一轮规划所缺的行程字段；先合并补充信息，再恢复原规划流程。",
            "intents": [
                {
                    "type": "plan",
                    "confidence": 1.0,
                    "description": "补充字段后继续规划",
                    "reason": "存在待恢复的行程规划请求，当前输入是对缺失行程字段的补充。",
                }
            ],
            "key_entities": {},
            "rewritten_query": f"{pending_query}；{query}",
            "agent_schedule": [
                {
                    "agent_name": "clarification",
                    "priority": 1,
                    "reason": "合并原始规划请求和用户刚补充的行程字段",
                    "expected_output": "完整结构化行程字段",
                },
                {
                    "agent_name": "search",
                    "priority": 2,
                    "reason": "基于完整字段调用高德 MCP 检索外部旅行信息",
                    "expected_output": "目的地 POI、天气、路线等外部数据",
                },
                {
                    "agent_name": "plan",
                    "priority": 3,
                    "reason": "整合补齐后的事项、外部信息和 MainAgent 注入的记忆上下文生成行程",
                    "expected_output": "结构化旅行计划",
                },
            ],
        }

    def _build_context_str(self) -> str:
        context_parts: List[str] = []
        system_memory = None
        dialogue_history: List[str] = []

        for item in self.conversation_history:
            if item.startswith("[系统记忆]"):
                system_memory = item
            else:
                dialogue_history.append(item)

        if system_memory:
            context_parts.append(system_memory)
        if dialogue_history:
            context_parts.extend(dialogue_history)

        return "\n".join(context_parts) if context_parts else "无历史对话"

    async def _invoke_structured(self, prompt: str) -> IntentionOutput:
        # 优先使用 LangChain ChatOpenAI 的结构化输出能力
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                lc_messages = importlib.import_module("langchain_core.messages")
                SystemMessage = getattr(lc_messages, "SystemMessage")
                HumanMessage = getattr(lc_messages, "HumanMessage")

                structured_llm = lc_model.with_structured_output(IntentionOutput)
                result = await structured_llm.ainvoke(
                    [
                        SystemMessage(content="你是一个高级意图识别专家。请严格按结构化字段输出。"),
                        HumanMessage(content=prompt),
                    ]
                )
                if isinstance(result, IntentionOutput):
                    return result
                if isinstance(result, dict):
                    return IntentionOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(
            self.model,
            [
                {"role": "system", "content": "你是一个高级意图识别专家。请仅返回 JSON。"},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = self._parse_json_text(str(text))
        return IntentionOutput.model_validate(parsed)

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

    def _sanitize_agent_schedule(self, schedule: Any) -> List[Dict[str, Any]]:
        if not isinstance(schedule, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in schedule:
            if not isinstance(item, dict):
                continue

            raw_name = item.get("agent_name")
            if raw_name is None:
                continue

            agent_name = str(raw_name).strip().lower()
            agent_name = {
                "information_query": "search",
                "itinerary_planning": "plan",
                "event_collection": "clarification",
                "memory_query": "memory",
                "preference": "memory",
            }.get(agent_name, agent_name)
            if not agent_name or agent_name in {"none", "null", "n/a", "na", "unknown"}:
                continue

            if agent_name not in self._valid_agent_names:
                continue

            normalized.append(
                {
                    "agent_name": agent_name,
                    "priority": int(item.get("priority", 1) or 1),
                    "reason": str(item.get("reason", "默认查询") or "默认查询"),
                    "expected_output": str(item.get("expected_output", "查询结果") or "查询结果"),
                }
            )

        return normalized


# Backward compatibility: older code/tests may still import IntentionAgent.
IntentionAgent = IntentRecognition
