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
from agents.workflow_skills.router import SkillRouter

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
        skill_router: Optional[Any] = None,
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
        self.skill_router = SkillRouter() if skill_router is None else skill_router

        # Backward-compatible attribute names for older callers.
        self.intention_agent = self.intent_recognition
        self.orchestrator = self.agent_scheduler

    def set_event_callback(self, callback: Optional[Callable[[Any], None]]) -> None:
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
        workflow_match = self._match_workflow_skill(state)
        if workflow_match:
            return await self._run_workflow_skill(state, workflow_match)

        intention_state = await self.intent_recognition.run(state)
        return await self._run_intention_state(intention_state)

    async def _run_intention_state(self, intention_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run a prepared intention through MainAgent supervision."""

        intention_data = intention_state.get("intention_data")
        if not isinstance(intention_data, dict):
            return {**intention_state, "final_result": {"status": "error", "message": "无法理解您的需求，请重新描述。"}}

        self._emit_intention(intention_data)

        schedule = intention_data.get("agent_schedule") or []
        if self._is_main_agent_memory_turn(intention_data):
            self._emit_event("🧠 主智能体正在处理记忆请求...")
            return await self._handle_memory_turn(intention_state)

        if schedule:
            self._emit_event("🧩 主智能体正在监督业务智能体执行...")
        else:
            self._emit_event("💬 主智能体直接回复当前问题...")

        return await self._supervise_business_agents(intention_state)

    def _match_workflow_skill(self, state: Dict[str, Any]):
        """Match common business workflow skills before LLM intent recognition."""

        if not self.skill_router or not hasattr(self.skill_router, "match"):
            return None
        match = self.skill_router.match(state)
        if not match:
            return None
        self._emit_event(
            {
                "type": "chain",
                "stage": "workflow_skill",
                "title": "命中常见业务 Skill",
                "message": f"已命中 {match.skill_name}，按 Skill 内置 workflow 执行。",
                "skill_name": match.skill_name,
                "confidence": match.confidence,
                "reason": match.reason,
                "workflow_plan": match.workflow_plan,
            }
        )
        return match

    async def _run_workflow_skill(self, state: Dict[str, Any], match) -> Dict[str, Any]:
        """Execute a matched workflow skill through the shared runtime."""

        intention_state = {
            **state,
            "intention_data": match.intention_data,
            "workflow_plan": match.workflow_plan,
            "workflow_skill": {
                "name": match.skill_name,
                "confidence": match.confidence,
                "reason": match.reason,
                "slots": match.slots,
                "workflow_plan": match.workflow_plan,
            },
        }
        return await self._run_intention_state(intention_state)

    def _is_main_agent_memory_turn(self, intention_data: Dict[str, Any]) -> bool:
        direct_action = intention_data.get("direct_action")
        if isinstance(direct_action, dict) and direct_action.get("type") == "memory":
            return True

        # Backward compatibility: older recognizers may still express a
        # memory-only internal action as a legacy memory schedule item.
        schedule = intention_data.get("agent_schedule") or []
        if not schedule or not isinstance(schedule, list):
            return False
        memory_names = {"memory", "memory_query", "preference"}
        for task in schedule:
            if not isinstance(task, dict):
                return False
            if str(task.get("agent_name", "")).strip() not in memory_names:
                return False
        return True

    async def _handle_memory_turn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle explicit memory queries/updates as a MainAgent capability."""

        state = self._inject_memory_context(state)
        intention_data = state.get("intention_data") or {}
        query = intention_data.get("rewritten_query") or state.get("user_query", "")
        intents = intention_data.get("intents") or []
        state = {
            **state,
            "context": {
                **(state.get("context") or {}),
                "rewritten_query": query,
                "intents": intents,
                "key_entities": intention_data.get("key_entities", {}),
            },
        }

        if self._is_preference_update_intent(intents, query):
            data = await self._extract_preference_update(state)
        else:
            data = self._query_memory_direct(query, state.get("context", {}))

        result = {
            "agent_name": "memory",
            "priority": 1,
            "result": {
                "status": "success",
                "agent_name": "memory",
                "data": data,
            },
        }
        final_result = {
            "status": "completed",
            "intention": {
                "intents": intention_data.get("intents", []),
                "key_entities": intention_data.get("key_entities", {}),
            },
            "agents_executed": 1,
            "results": [
                {
                    "agent_name": "memory",
                    "priority": 1,
                    "status": "success",
                    "data": data,
                }
            ],
        }
        memory_policy = self._build_memory_policy({"results": [result]})
        self._apply_memory_policy([result], memory_policy)
        final_result["memory_policy"] = memory_policy

        return {
            **state,
            "results": [result],
            "final_result": final_result,
        }

    def _is_preference_update_intent(self, intents: Any, query: str) -> bool:
        for item in intents or []:
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() in {"preference", "preference_management"}:
                return True

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
        preference_words = ("偏好", "喜欢", "不喜欢", "预算", "酒店", "交通方式", "节奏", "餐饮", "常住", "住在", "航空", "航班", "航空公司", "东航", "南航", "国航", "海航", "轻松", "紧凑", "经济型", "舒适型", "品质型")
        return any(marker in q for marker in explicit_markers) and any(word in q for word in preference_words)

    async def _extract_preference_update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from agents.preference_agent import PreferenceAgent

            extractor = PreferenceAgent(
                name="MainAgentPreferenceExtractor",
                model=self.model,
                memory_manager=self.memory_manager,
            )
            return await extractor.run(state)
        except Exception as e:
            logger.warning("MainAgent preference extraction failed: %s", e, exc_info=True)
            return {"has_preferences": False, "preferences": [], "error": str(e)}

    def _query_memory_direct(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"query": query, "answer": "当前未启用记忆系统。"}
        if hasattr(self.memory_manager, "query_memory"):
            memory_context = context.get("memory_context") if isinstance(context.get("memory_context"), dict) else context
            return self.memory_manager.query_memory(query, memory_context)
        return {"query": query, "answer": "当前记忆系统不支持直接查询。"}

    async def _supervise_business_agents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run child agents while MainAgent keeps ownership of the turn state."""

        state = self._inject_memory_context(state)
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

    def _inject_memory_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attach MainAgent-owned memory context before business scheduling."""

        memory_context = self._build_memory_context()
        if not memory_context:
            return state

        context = dict(state.get("context") or {})
        existing_memory_context = context.get("memory_context")
        if isinstance(existing_memory_context, dict):
            merged_memory_context = {**memory_context, **existing_memory_context}
        else:
            merged_memory_context = memory_context

        context["memory_context"] = merged_memory_context
        for key in ("recent_dialogue", "user_preferences", "trip_history", "behavior_feedback"):
            if key in memory_context and key not in context:
                context[key] = memory_context[key]

        return {**state, "context": context}

    def _build_memory_context(self) -> Dict[str, Any]:
        if not self.memory_manager:
            return {}

        if hasattr(self.memory_manager, "get_runtime_context"):
            try:
                return self.memory_manager.get_runtime_context(
                    recent_turns=3,
                    trip_limit=20,
                    feedback_limit=20,
                )
            except Exception:
                logger.debug("Failed to build memory runtime context", exc_info=True)

        memory_context: Dict[str, Any] = {}
        short_term = getattr(self.memory_manager, "short_term", None)
        long_term = getattr(self.memory_manager, "long_term", None)

        if short_term and hasattr(short_term, "get_recent_context"):
            try:
                memory_context["recent_dialogue"] = short_term.get_recent_context(3)
            except Exception:
                logger.debug("Failed to read short-term memory context", exc_info=True)

        if long_term and hasattr(long_term, "get_preference"):
            try:
                memory_context["user_preferences"] = long_term.get_preference()
            except Exception:
                logger.debug("Failed to read user preferences", exc_info=True)

        if long_term and hasattr(long_term, "get_trip_history"):
            try:
                memory_context["trip_history"] = long_term.get_trip_history(limit=20)
            except Exception:
                logger.debug("Failed to read trip history", exc_info=True)

        if long_term and hasattr(long_term, "get_behavior_feedback"):
            try:
                memory_context["behavior_feedback"] = long_term.get_behavior_feedback(limit=20)
            except Exception:
                logger.debug("Failed to read behavior feedback", exc_info=True)

        return {key: value for key, value in memory_context.items() if value not in (None, "")}

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
        query = graph_state.get("user_query") or graph_state.get("intention_data", {}).get("rewritten_query", "")
        if not query:
            return
        metadata: Dict[str, Any] = {"reason": reason}
        for result in reversed(graph_state.get("results", [])):
            if result.get("agent_name") not in {"clarification", "event_collection"}:
                continue
            data = result.get("result", {}).get("data", {})
            if not isinstance(data, dict):
                continue
            metadata["missing_info"] = data.get("missing_info", [])
            metadata["event_data"] = {
                key: data.get(key)
                for key in ("origin", "destination", "start_date", "duration_days", "budget_level", "pace_preference")
                if data.get(key) not in (None, "")
            }
            break
        self.memory_manager.short_term.set_pending_plan(query, metadata)

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
            self._is_explicit_preference_update_result(result)
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

    def _is_explicit_preference_update_result(self, result: Dict[str, Any]) -> bool:
        if self.memory_manager and hasattr(self.memory_manager, "is_explicit_preference_update_result"):
            return self.memory_manager.is_explicit_preference_update_result(result)
        return False
