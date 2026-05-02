"""MainAgent 的意图识别能力。

该模块负责一轮意图识别流程编排：上下文整理、prompt 构造、LLM 调用、
规则兜底/护栏和业务智能体调度规范化。规则细节在 intent_rules.py，
LLM 调用和 schema 在 intent_llm.py。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from agents.intent_llm import AgentScheduleItem, IntentItem, IntentLLMClient, IntentionOutput
from agents.intent_rules import IntentRuleEngine
from utils.langchain_runtime import message_to_dict, to_lc_messages
from utils.skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class IntentRecognition(IntentRuleEngine):
    """主智能体的意图识别与任务计划能力。"""

    def __init__(self, name: str = "IntentRecognition", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.llm_client = IntentLLMClient(model)
        self.conversation_history: List[str] = []
        self.skill_loader = SkillLoader()
        self._valid_agent_names = {"memory", "search", "plan", "clarification"}

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query = self._load_turn_context(state)
        pending_plan = state.get("pending_plan") if isinstance(state.get("pending_plan"), dict) else None

        direct_result = self._try_pending_plan_completion(user_query, pending_plan)
        if direct_result:
            return {**state, "intention_data": direct_result}

        direct_result = self._try_direct_intent(user_query)
        if direct_result:
            return {**state, "intention_data": direct_result}

        prompt = self._build_prompt(user_query)
        try:
            result = await self.llm_client.invoke_structured(prompt)
        except Exception as e:
            logger.error("Intent recognition failed: %s", e)
            result = self._build_error_fallback_intention(user_query, e)

        result_dict = result.model_dump()
        return {**state, "intention_data": self._postprocess_result(result_dict, user_query)}

    def _load_turn_context(self, state: Dict[str, Any]) -> str:
        messages = state.get("messages", [])
        user_query = state.get("user_query", "")
        if not messages:
            return user_query

        lc_messages = to_lc_messages(messages)
        user_query = str(lc_messages[-1].content) if lc_messages else user_query
        self.conversation_history = []
        for msg in lc_messages[:-1]:
            msg_dict = message_to_dict(msg)
            if msg_dict["role"] == "system":
                self.conversation_history.append(f"[系统记忆]\n{msg_dict['content']}")
                continue
            role_name = "用户" if msg_dict["role"] == "user" else "助手"
            content = msg_dict["content"][:800] if len(msg_dict["content"]) > 800 else msg_dict["content"]
            if len(msg_dict["content"]) > 800:
                content += "..."
            self.conversation_history.append(f"{role_name}: {content}")
        return user_query

    def _build_prompt(self, user_query: str) -> str:
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
        context_str = self._build_context_str()

        return f"""你是 TravelFlow 旅游出行助手 MainAgent 内部的意图识别能力（IntentRecognition）。请分析用户查询并返回结构化调度决策。

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
- 历史行程只能作为参考，不能自动填入本次行程的出发地、日期、天数等核心字段；除非用户明确说“继续刚才/上次/这个行程”，否则本轮没说的核心字段必须留给 clarification 追问。

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

    def _postprocess_result(self, result_dict: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        if not result_dict.get("rewritten_query"):
            result_dict["rewritten_query"] = str(user_query)
        result_dict = self._remove_unsupported_history_inferences(result_dict, user_query)

        if self._is_conversation_intent(result_dict, user_query):
            result_dict["intents"] = result_dict.get("intents") or [
                {"type": "conversation", "confidence": 1.0, "description": "普通对话", "reason": "不需要调用业务智能体。"}
            ]
            result_dict["direct_answer"] = result_dict.get("direct_answer") or self._build_direct_conversation_answer(user_query)
            result_dict["agent_schedule"] = []
            return result_dict

        if not result_dict.get("intents"):
            result_dict["intents"] = [
                {"type": "search", "confidence": 0.5, "description": "默认查询意图", "reason": "空意图结果，使用默认策略"}
            ]

        result_dict["agent_schedule"] = self._sanitize_agent_schedule(result_dict.get("agent_schedule", []))
        result_dict["agent_schedule"] = self._normalize_layered_planning_schedule(result_dict)

        if self._is_conversation_intent(result_dict, user_query):
            result_dict["direct_answer"] = result_dict.get("direct_answer") or self._build_direct_conversation_answer(user_query)
            result_dict["agent_schedule"] = []
            return result_dict

        if not result_dict.get("agent_schedule") and not (result_dict.get("direct_answer") or result_dict.get("direct_action")):
            result_dict["agent_schedule"] = [
                {"agent_name": "search", "priority": 1, "reason": "默认查询", "expected_output": "查询结果"}
            ]
        return result_dict

    def _normalize_layered_planning_schedule(self, result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        schedule = result_dict.get("agent_schedule") or []
        if not self._is_planning_intention(result_dict):
            return schedule

        by_name = {item.get("agent_name"): item for item in schedule if isinstance(item, dict)}
        return [
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


# Backward compatibility: older code/tests may still import these from intention_agent.
IntentionAgent = IntentRecognition

__all__ = [
    "AgentScheduleItem",
    "IntentItem",
    "IntentRecognition",
    "IntentionAgent",
    "IntentionOutput",
]
