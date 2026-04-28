"""User-facing TravelFlow main agent.

MainAgent owns the conversation turn. Intent recognition is an internal
capability, and business-agent orchestration is an internal tool.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from agents.agent_scheduler import AgentScheduler
from agents.intent_recognition import IntentRecognition

logger = logging.getLogger(__name__)


class MainAgent:
    """TravelFlow 主智能体：面向用户沟通，并按需调用业务智能体。"""

    def __init__(
        self,
        name: str = "MainAgent",
        model=None,
        intent_recognition: Optional[IntentRecognition] = None,
        agent_scheduler: Optional[AgentScheduler] = None,
        intention_agent: Optional[IntentRecognition] = None,
        orchestrator: Optional[AgentScheduler] = None,
        memory_manager=None,
        event_callback: Optional[Callable[[Any], None]] = None,
        intention_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.intent_recognition = intent_recognition or intention_agent or IntentRecognition(model=model)
        self.agent_scheduler = agent_scheduler or orchestrator
        self.memory_manager = memory_manager
        self.event_callback = event_callback
        self.intention_callback = intention_callback

        # Backward-compatible attribute names for older callers.
        self.intention_agent = self.intent_recognition
        self.orchestrator = self.agent_scheduler

    def set_event_callback(self, callback):
        self.event_callback = callback
        if self.agent_scheduler and hasattr(self.agent_scheduler, "set_event_callback"):
            self.agent_scheduler.set_event_callback(callback)

    def _emit_event(self, event: Any):
        if not self.event_callback:
            return
        try:
            self.event_callback(event)
        except Exception:
            logger.debug("MainAgent event callback failed", exc_info=True)

    def _emit_intention(self, intention_data: Dict[str, Any]):
        if not self.intention_callback:
            return
        try:
            self.intention_callback(intention_data)
        except Exception:
            logger.debug("MainAgent intention callback failed", exc_info=True)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle one user turn and return the orchestrated state."""

        if not self.agent_scheduler:
            return {**state, "final_result": {"status": "error", "message": "主智能体未配置调度器"}}

        self._emit_event("🧠 主智能体正在理解用户需求...")
        intention_state = await self.intent_recognition.run(state)
        intention_data = intention_state.get("intention_data")
        if not isinstance(intention_data, dict):
            return {**state, "final_result": {"status": "error", "message": "无法理解您的需求，请重新描述。"}}

        self._emit_intention(intention_data)

        if self.memory_manager and not state.get("_user_message_recorded"):
            user_query = state.get("user_query")
            if user_query:
                self.memory_manager.add_message("user", str(user_query))
                state["_user_message_recorded"] = True

        schedule = intention_data.get("agent_schedule") or []
        if schedule:
            self._emit_event("🧩 主智能体正在按需调度业务智能体...")
        else:
            self._emit_event("💬 主智能体直接回复当前问题...")

        return await self.agent_scheduler.run(intention_state)
