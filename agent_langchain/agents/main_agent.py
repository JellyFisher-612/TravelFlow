"""User-facing TravelFlow main agent.

MainAgent owns the whole turn: intent recognition, child-agent supervision,
observation, stop/continue decisions, and final aggregation. Conversation
persistence is handled by the entry layer and MemoryManager.
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

        schedule = intention_data.get("agent_schedule") or []
        if schedule:
            self._emit_event("🧩 主智能体正在监督业务智能体执行...")
        else:
            self._emit_event("💬 主智能体直接回复当前问题...")

        return await self._supervise_business_agents(intention_state)

    async def _supervise_business_agents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run child agents while MainAgent keeps ownership of the turn state."""

        scheduler = self.agent_scheduler
        graph_state = scheduler.create_orchestration_state(state)
        graph_state.update(await scheduler.prepare(graph_state))

        max_batches = 20
        while scheduler.has_runnable_batches(graph_state):
            current_index = graph_state.get("batch_index", 0)
            graph_state.update(await scheduler.run_next_batch(graph_state))

            decision = self._observe_after_batch(graph_state)
            if decision == "stop":
                break

            if graph_state.get("batch_index", 0) == current_index:
                logger.error("MainAgent supervisor made no progress at batch index %s", current_index)
                break
            max_batches -= 1
            if max_batches <= 0:
                logger.error("MainAgent supervisor stopped after too many batch iterations")
                break

        graph_state.update(await scheduler.aggregate(graph_state))
        if graph_state.get("supervisor_decision") and isinstance(graph_state.get("final_result"), dict):
            graph_state["final_result"]["supervisor_decision"] = graph_state["supervisor_decision"]
        memory_policy = self._build_memory_policy(graph_state)
        self._apply_memory_policy(graph_state.get("results", []), memory_policy)
        if isinstance(graph_state.get("final_result"), dict):
            graph_state["final_result"]["memory_policy"] = memory_policy

        return {
            **state,
            "context": graph_state.get("context", {}),
            "results": graph_state.get("results", []),
            "final_result": graph_state.get("final_result", {"status": "error", "message": "Unknown error"}),
        }

    def _observe_after_batch(self, graph_state: Dict[str, Any]) -> str:
        scheduler = self.agent_scheduler
        blocking_reason = scheduler.get_blocking_reason(graph_state)
        if blocking_reason:
            self._handle_supervisor_blocking_decision(blocking_reason, graph_state)
            scheduler.truncate_remaining_batches(graph_state)
            return "stop"

        if scheduler.has_runnable_batches(graph_state):
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "supervisor_observation",
                    "title": "主智能体决定继续执行",
                    "message": "上一批智能体已完成，继续执行下一批任务。",
                }
            )
            return "continue"

        return "stop"

    def _blocking_reason_message(self, reason: str) -> str:
        return {
            "clarification_missing_info": "事项信息不足，需要先等待用户补充。",
            "search_failure": "外部检索失败，暂停规划以避免生成未核验方案。",
            "memory_preference_gaps": "关键偏好缺失，需要先确认预算或节奏。",
        }.get(reason, "当前结果不满足继续执行条件。")

    def _handle_supervisor_blocking_decision(self, reason: str, graph_state: Dict[str, Any]):
        """Own user-facing stop/ask decisions after child agents make suggestions."""

        decision = {
            "action": "ask_user" if reason in {"clarification_missing_info", "memory_preference_gaps"} else "stop",
            "reason": reason,
            "message": self._blocking_reason_message(reason),
        }
        graph_state["supervisor_decision"] = decision

        if reason in {"clarification_missing_info", "memory_preference_gaps"}:
            self._save_pending_plan(reason, graph_state)

        self._emit_event(
            {
                "type": "chain",
                "stage": "supervisor_decision",
                "title": self._blocking_reason_title(reason),
                "message": decision["message"],
                "decision": decision,
            }
        )

    def _save_pending_plan(self, reason: str, graph_state: Dict[str, Any]):
        if not self.memory_manager:
            return
        query = graph_state.get("intention_data", {}).get("rewritten_query", "")
        if not query:
            return
        self.memory_manager.short_term.set_pending_plan(query, {"reason": reason})

    def _blocking_reason_title(self, reason: str) -> str:
        return {
            "clarification_missing_info": "主智能体决定追问缺失事项",
            "search_failure": "主智能体决定暂停规划",
            "memory_preference_gaps": "主智能体决定追问关键偏好",
        }.get(reason, "主智能体决定暂停")

    def _build_memory_policy(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide which memory writes are allowed for this supervised turn."""

        decision = graph_state.get("supervisor_decision") if isinstance(graph_state.get("supervisor_decision"), dict) else {}
        stop_reason = decision.get("reason")
        results = graph_state.get("results", [])

        has_completed_plan = any(
            result.get("agent_name") in {"plan", "itinerary_planning"}
            and isinstance(result.get("result"), dict)
            and result.get("result", {}).get("status") == "success"
            and isinstance(result.get("result", {}).get("data"), dict)
            and bool(result.get("result", {}).get("data", {}).get("itinerary"))
            for result in results
            if isinstance(result, dict)
        )

        explicit_memory_update = any(
            result.get("agent_name") in {"memory", "preference"}
            and isinstance(result.get("result"), dict)
            and isinstance(result.get("result", {}).get("data"), dict)
            and bool(result.get("result", {}).get("data", {}).get("preferences"))
            for result in results
            if isinstance(result, dict)
        )

        allow_preference_writes = explicit_memory_update and stop_reason not in {"search_failure"}
        allow_trip_history_writes = has_completed_plan and not stop_reason

        return {
            "owner": "MainAgent",
            "allow_preference_writes": allow_preference_writes,
            "allow_trip_history_writes": allow_trip_history_writes,
            "clear_pending_plan_on_trip": allow_trip_history_writes,
            "reason": stop_reason or "turn_completed",
        }

    def _apply_memory_policy(self, results: list, policy: Dict[str, Any]):
        if not self.memory_manager or not hasattr(self.memory_manager, "apply_agent_results"):
            return
        self.memory_manager.apply_agent_results(results, policy=policy)
        self._emit_event(
            {
                "type": "chain",
                "stage": "memory_policy",
                "title": "主智能体应用记忆策略",
                "message": "已根据本轮状态决定是否写入长期偏好和历史行程。",
                "policy": policy,
            }
        )
