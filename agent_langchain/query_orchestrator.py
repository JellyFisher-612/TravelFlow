"""Shared one-turn query execution for CLI and Web entrypoints."""

from __future__ import annotations

import json
import re
import asyncio
from typing import Any, Optional

from config import RESILIENCE_CONFIG
from utils.circuit_breaker import CircuitOpenError
from utils.llm_resilience import retry_with_backoff


class QueryOrchestrator:
    """Owns non-UI execution steps for a TravelFlow user turn."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    async def execute_query(self, user_input: str) -> tuple[Optional[dict], Optional[str]]:
        rt = self.runtime
        rt._web_trace_events = []

        if rt.circuit_breaker:
            try:
                rt.circuit_breaker.raise_if_open()
            except CircuitOpenError:
                return None, "服务暂时不可用，请稍后再试。"

        rc = RESILIENCE_CONFIG
        max_retries = rc.get("max_retries", 3)

        long_term_summary = ""
        if should_skip_long_term_summary(user_input):
            rt._emit_runtime_event("当前是普通对话，跳过长期记忆摘要。")
        else:
            rt._emit_runtime_event("正在读取会话上下文和长期记忆...")
            long_term_summary = await rt._get_long_term_summary(user_input)

        recent_context = rt.memory_manager.short_term.get_recent_context(n_turns=5)
        context_messages = []
        if long_term_summary:
            context_messages.append({"role": "system", "content": long_term_summary})
        for msg in recent_context:
            context_messages.append({"role": msg["role"], "content": msg["content"]})
        context_messages.append({"role": "user", "content": user_input})

        shared_state = {
            "user_id": rt.user_id,
            "session_id": rt.session_id,
            "user_query": user_input,
            "pending_plan": rt.memory_manager.short_term.get_pending_plan(),
            "messages": context_messages,
            "context": {},
            "results": [],
        }

        # The entry layer owns turn-level chat persistence. MainAgent only
        # understands and delegates the request.
        rt.memory_manager.add_message("user", user_input)
        shared_state["_user_message_recorded"] = True

        try:
            main_result = await retry_with_backoff(
                lambda: rt.main_agent.run(shared_state),
                max_retries=max_retries,
                base_delay_sec=rc.get("retry_base_delay_sec", 1.0),
                max_delay_sec=rc.get("retry_max_delay_sec", 30.0),
            )
            if rt.circuit_breaker:
                rt.circuit_breaker.record_success()
        except CircuitOpenError:
            return None, "服务暂时不可用，请稍后再试。"
        except Exception:
            if rt.circuit_breaker:
                rt.circuit_breaker.record_failure()
            raise

        result_data = main_result.get("final_result") or {"error": "解析结果失败"}
        assistant_display = ""
        try:
            assistant_display = rt.render_result_text(result_data)
        except Exception:
            assistant_display = ""
        rt.memory_manager.add_message(
            "assistant",
            json.dumps(result_data, ensure_ascii=False),
            metadata={"display": assistant_display} if assistant_display else None,
        )
        if hasattr(rt.memory_manager, "summarize_current_session_async"):
            async def _summarize_session_log():
                try:
                    await rt.memory_manager.summarize_current_session_async()
                except Exception:
                    pass

            asyncio.create_task(_summarize_session_log())
        rt._emit_runtime_event("✅ 结果已生成")
        return result_data, None


def should_skip_long_term_summary(user_input: str) -> bool:
    """Avoid unnecessary LLM summarization for short meta/chat turns."""
    query = (user_input or "").strip().lower()
    if not query:
        return True
    if any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我")):
        return False
    normalized = re.sub(r"[\s\ufeff\u200b]+", "", query)
    normalized = re.sub(r"[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+$", "", normalized)
    normalized = re.sub(r"^[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+", "", normalized)
    exact_meta = {
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
    if normalized in exact_meta:
        return True
    return any(
        phrase in query
        for phrase in (
            "介绍一下你自己",
            "你是一个什么",
            "你是干什么的",
            "你是做什么的",
            "你做什么的",
            "这个系统怎么用",
        )
    )
