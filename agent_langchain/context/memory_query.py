"""
记忆查询智能体 MemoryQueryAgent
职责：基于用户长期记忆回答历史相关问题
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


class MemoryAnswerOutput(BaseModel):
    answer: str = Field(default="无法基于记忆生成回答")


class MemoryQueryAgent:
    """记忆查询智能体 - 基于长期记忆回答用户问题"""

    def __init__(
        self,
        name: str = "MemoryQueryAgent",
        model=None,
        memory_manager=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        from utils.skill_loader import SkillLoader

        self.skill_loader = SkillLoader()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", {})
        user_query = context.get("rewritten_query", "")
        if not user_query:
            recent_dialogue = context.get("recent_dialogue", [])
            if recent_dialogue:
                for msg in reversed(recent_dialogue):
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break

        if not user_query:
            return {"status": "error", "message": "无法获取用户查询"}

        trip_history = []
        preferences = {}
        chat_summary = ""

        if self.memory_manager:
            trip_history = self.memory_manager.long_term.get_trip_history(limit=50)
            preferences = self.memory_manager.long_term.get_preference()
            if self._should_consult_chat_logs(user_query, trip_history):
                chat_summary = self._load_chat_log_context()

        trip_text = self._format_trip_history(trip_history)
        pref_text = self._format_preferences(preferences)

        skill_instruction = self.skill_loader.get_skill_content("memory-query")
        if not skill_instruction:
            skill_instruction = "请基于用户的历史记忆回答问题，如无相关记录请诚实说明。"

        prompt = f"""你是一个个人记忆助手，请基于用户历史记忆回答问题。

【用户问题】
{user_query}

【用户旅行历史】
{trip_text}

【用户偏好】
{pref_text}

【历史聊天日志】
{chat_summary if chat_summary else "（默认不读取旧聊天日志；当前问题可由结构化记忆回答，或没有可用日志摘要）"}

【任务说明】
{skill_instruction}
"""

        try:
            answer_obj = await self._invoke_structured_answer(prompt)
            answer = answer_obj.answer.strip() if answer_obj.answer else "无法基于记忆生成回答"

            result = {
                "status": "success",
                "query": user_query,
                "answer": answer,
                "memory_sources": {
                    "trip_count": len(trip_history),
                    "has_preferences": any(v for v in preferences.values() if v),
                    "used_chat_logs": bool(chat_summary),
                },
            }
            return result

        except Exception as e:
            logger.error("Memory query failed: %s", e)
            return {"status": "error", "message": f"记忆查询失败: {str(e)}", "query": user_query}

    def _should_consult_chat_logs(self, query: str, trip_history: List[Dict[str, Any]]) -> bool:
        q = query or ""
        explicit_chat_request = any(
            marker in q
            for marker in (
                "聊天记录",
                "对话记录",
                "以前聊",
                "之前聊",
                "上次聊",
                "说过什么",
                "问过什么",
                "我提到过",
                "我们聊过",
            )
        )
        if not explicit_chat_request:
            return False

        trip_only_request = any(
            marker in q
            for marker in ("历史行程", "过去行程", "去过哪里", "去过哪些", "旅行历史", "旅游历史")
        )
        if trip_only_request and trip_history:
            return False
        return True

    def _load_chat_log_context(self) -> str:
        long_term = getattr(self.memory_manager, "long_term", None)
        if not long_term:
            return ""

        if hasattr(long_term, "get_session_summaries"):
            current_session = getattr(self.memory_manager, "session_id", None)
            summaries = long_term.get_session_summaries(limit=5)
            if summaries:
                return "\n".join(
                    f"- {item.get('summary', '')}"
                    for item in summaries
                    if item.get("session_id") != current_session and item.get("summary")
                )

        current_session = getattr(self.memory_manager, "session_id", None)
        logs = long_term.get_chat_history(limit=40)
        lines = []
        for msg in logs:
            if msg.get("session_id") == current_session:
                continue
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", "") or "")
            metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
            if role == "assistant" and metadata.get("display"):
                content = str(metadata.get("display"))
            if content.strip():
                lines.append(f"{role}: {content.strip()[:500]}")
        return "\n".join(lines[-20:])

    async def _invoke_structured_answer(self, prompt: str) -> MemoryAnswerOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(MemoryAnswerOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, MemoryAnswerOutput):
                    return result
                if isinstance(result, dict):
                    return MemoryAnswerOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(
            self.model,
            [
                {"role": "system", "content": "你是一个个人记忆助手，帮助用户查询和理解历史记录。"},
                {"role": "user", "content": prompt},
            ]
        )
        text = text.strip()
        return MemoryAnswerOutput(answer=text or "无法基于记忆生成回答")

    def _format_trip_history(self, trip_history: List[Dict]) -> str:
        if not trip_history:
            return "（暂无旅行记录）"

        lines = []
        for i, trip in enumerate(trip_history, 1):
            origin = trip.get("origin", "未知")
            destination = trip.get("destination", "未知")
            start_date = trip.get("start_date", "")
            end_date = trip.get("end_date", "")
            purpose = trip.get("purpose", "旅游")
            timestamp = trip.get("timestamp", "")

            if start_date and end_date:
                lines.append(f"{i}. {origin} → {destination} ({start_date} 至 {end_date}) - {purpose}")
            elif start_date:
                lines.append(f"{i}. {origin} → {destination} ({start_date}) - {purpose}")
            else:
                lines.append(f"{i}. {origin} → {destination} - {purpose} (记录时间: {timestamp})")

        return "\n".join(lines)

    def _format_preferences(self, preferences: Dict) -> str:
        if not preferences or not any(v for v in preferences.values() if v):
            return "（暂无偏好记录）"

        lines = []
        pref_names = {
            "budget": "预算偏好",
            "accommodation": "住宿偏好",
            "transportation": "交通偏好",
            "food": "餐饮偏好",
            "activity": "活动偏好",
            "other": "其他偏好",
        }

        for key, value in preferences.items():
            if value and key in pref_names:
                lines.append(f"- {pref_names[key]}: {value}")

        return "\n".join(lines) if lines else "（暂无偏好记录）"
