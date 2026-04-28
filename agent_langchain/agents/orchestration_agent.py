"""MainAgent 可调用的业务智能体编排工具。

职责：根据 agent_schedule，基于 LangGraph 按 priority 分批调度业务智能体。
同 priority 并行执行，不同 priority 串行执行。
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class OrchestrationState(TypedDict):
    intention_data: Dict[str, Any]
    context: Dict[str, Any]
    batches: List[List[Dict[str, Any]]]
    batch_index: int
    results: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    search_refinement_count: int


class AgentScheduler:
    """业务智能体执行工具 - 通过 LangGraph 状态图调度多智能体。"""

    def __init__(
        self,
        name: str = "AgentScheduler",
        agent_registry: Optional[Dict[str, Any]] = None,
        memory_manager=None,
        event_callback=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.agent_registry = agent_registry or {}
        self.memory_manager = memory_manager
        self.event_callback = event_callback
        self._agent_aliases = {
            "memory_query": "memory",
            "preference": "memory",
            "information_query": "search",
            "itinerary_planning": "plan",
            "event_collection": "clarification",
        }
        self._graph = self._build_graph()

    def set_event_callback(self, callback):
        """Attach a runtime event callback used by Web streaming."""
        self.event_callback = callback

    def _emit_event(self, event: Dict[str, Any]):
        if not self.event_callback:
            return
        try:
            self.event_callback(event)
        except Exception:
            pass

    def register_agent(self, agent_name: str, agent: Any):
        self.agent_registry[agent_name] = agent
        logger.info("Registered agent: %s", agent_name)

    def unregister_agent(self, agent_name: str):
        if agent_name in self.agent_registry:
            del self.agent_registry[agent_name]
            logger.info("Unregistered agent: %s", agent_name)

    def _build_graph(self):
        graph_module = importlib.import_module("langgraph.graph")
        StateGraph = getattr(graph_module, "StateGraph")
        END = getattr(graph_module, "END")

        graph_builder = StateGraph(OrchestrationState)

        graph_builder.add_node("prepare", self._prepare_node)
        graph_builder.add_node("run_batch", self._run_batch_node)
        graph_builder.add_node("aggregate", self._aggregate_node)

        graph_builder.set_entry_point("prepare")
        graph_builder.add_conditional_edges(
            "prepare",
            self._route_after_prepare,
            {"run_batch": "run_batch", "aggregate": "aggregate"},
        )
        graph_builder.add_conditional_edges(
            "run_batch",
            self._route_after_batch,
            {"run_batch": "run_batch", "aggregate": "aggregate"},
        )
        graph_builder.add_edge("aggregate", END)

        return graph_builder.compile()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intention_data = state.get("intention_data") or {}
        if not isinstance(intention_data, dict):
            return {**state, "final_result": {"error": "Invalid intention format"}}

        initial_state: OrchestrationState = {
            "intention_data": intention_data,
            "context": state.get("context", {}),
            "batches": [],
            "batch_index": 0,
            "results": [],
            "final_result": {},
            "search_refinement_count": 0,
        }

        graph_state = await self._graph.ainvoke(initial_state)
        final_result = graph_state.get("final_result", {"status": "error", "message": "Unknown error"})

        if self.memory_manager:
            self._update_memory(intention_data, graph_state.get("results", []))

        return {
            **state,
            "context": graph_state.get("context", {}),
            "results": graph_state.get("results", []),
            "final_result": final_result,
        }

    async def _prepare_node(self, state: OrchestrationState) -> Dict[str, Any]:
        intention_data = state["intention_data"]
        agent_schedule = intention_data.get("agent_schedule", [])
        cleaned_schedule: List[Dict[str, Any]] = []
        for task in agent_schedule:
            if not isinstance(task, dict):
                continue
            raw_name = task.get("agent_name")
            agent_name = str(raw_name).strip() if raw_name is not None else ""
            agent_name = self._agent_aliases.get(agent_name, agent_name)
            if not agent_name or agent_name.lower() in {"none", "null", "n/a", "na", "unknown"}:
                continue
            if agent_name not in self.agent_registry:
                logger.warning("Skip unknown agent in schedule: %s", agent_name)
                continue
            task["agent_name"] = agent_name
            cleaned_schedule.append(task)
        agent_schedule = cleaned_schedule

        if not agent_schedule:
            direct_answer = intention_data.get("direct_answer")
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "direct_answer",
                    "title": "主智能体直接回复",
                    "message": "当前请求不需要调用业务子智能体。",
                }
            )
            return {
                "context": self._prepare_context(intention_data),
                "batches": [],
                "batch_index": 0,
                "results": [],
                "final_result": {
                    "status": "no_agents",
                    "message": direct_answer or "没有可调度的智能体",
                    "direct_answer": direct_answer,
                    "intention": {
                        "intents": intention_data.get("intents", []),
                        "key_entities": intention_data.get("key_entities", {}),
                    },
                },
            }

        sorted_schedule = sorted(agent_schedule, key=lambda item: item.get("priority", 999))

        batches: List[List[Dict[str, Any]]] = []
        current_priority = None
        for task in sorted_schedule:
            priority = task.get("priority", 999)
            if current_priority is None or priority != current_priority:
                batches.append([task])
                current_priority = priority
            else:
                batches[-1].append(task)

        logger.info("Orchestrating %d agents in %d priority batches", len(sorted_schedule), len(batches))
        self._emit_event(
            {
                "type": "chain",
                "stage": "schedule",
                "title": "调度计划",
                "message": "已生成智能体执行顺序",
                "items": [
                    {
                        "agent": task.get("agent_name"),
                        "priority": task.get("priority", 999),
                        "reason": task.get("reason", ""),
                        "expected_output": task.get("expected_output", ""),
                    }
                    for task in sorted_schedule
                ],
            }
        )

        return {
            "context": self._prepare_context(intention_data),
            "batches": batches,
            "batch_index": 0,
            "results": [],
            "search_refinement_count": state.get("search_refinement_count", 0),
        }

    async def _run_batch_node(self, state: OrchestrationState) -> Dict[str, Any]:
        batches = state.get("batches", [])
        batch_index = state.get("batch_index", 0)
        if batch_index >= len(batches):
            return {}

        tasks = batches[batch_index]
        context = state.get("context", {})
        previous_results = state.get("results", [])
        self._emit_event(
            {
                "type": "chain",
                "stage": "batch_start",
                "title": f"执行第 {batch_index + 1} 批智能体",
                "message": "、".join(str(task.get("agent_name")) for task in tasks),
            }
        )

        batch_results = await self._execute_parallel_agents(
            tasks=tasks,
            context=self._context_with_previous_results(context, previous_results),
            previous_results=previous_results,
        )

        combined_results = previous_results + batch_results
        next_batches = list(batches)
        next_context = dict(context)
        refinement_count = state.get("search_refinement_count", 0)

        plan_search_requests = self._extract_plan_search_requests(batch_results)
        if plan_search_requests and refinement_count < 1 and "search" in self.agent_registry and "plan" in self.agent_registry:
            next_context["plan_search_requests"] = plan_search_requests
            next_context["search_refinement_count"] = refinement_count + 1
            next_batches.extend(
                [
                    [
                        {
                            "agent_name": "search",
                            "priority": 90 + refinement_count * 2,
                            "reason": "根据行程规划智能体提出的信息缺口补充高德检索",
                            "expected_output": "补充 POI、天气、餐饮、路线等外部信息",
                        }
                    ],
                    [
                        {
                            "agent_name": "plan",
                            "priority": 91 + refinement_count * 2,
                            "reason": "整合补充检索结果后重新生成行程规划",
                            "expected_output": "更具体的结构化旅行计划",
                        }
                    ],
                ]
            )
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "plan_requests_search",
                    "title": "规划需要补充检索",
                    "message": "行程规划智能体发现外部信息不足，已追加一轮 search -> plan。",
                    "items": plan_search_requests,
                }
            )

        return {
            "context": next_context,
            "batches": next_batches,
            "results": combined_results,
            "batch_index": batch_index + 1,
            "search_refinement_count": next_context.get("search_refinement_count", refinement_count),
        }

    async def _aggregate_node(self, state: OrchestrationState) -> Dict[str, Any]:
        existing = state.get("final_result") or {}
        if existing.get("status") == "no_agents":
            return {"final_result": existing}

        intention_data = state.get("intention_data", {})
        results = state.get("results", [])
        return {
            "final_result": self._aggregate_results(results, intention_data),
        }

    def _route_after_prepare(self, state: OrchestrationState) -> str:
        return "run_batch" if state.get("batches") else "aggregate"

    def _route_after_batch(self, state: OrchestrationState) -> str:
        if self._has_blocking_missing_info(state):
            if self.memory_manager:
                query = state.get("intention_data", {}).get("rewritten_query", "")
                if query:
                    self.memory_manager.short_term.set_pending_plan(
                        query,
                        {"reason": "clarification_missing_info"},
                    )
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "blocked_for_clarification",
                    "title": "等待用户补充信息",
                    "message": "事项收集发现关键信息不足，已停止后续检索和规划。",
                }
            )
            return "aggregate"
        if self._has_blocking_search_failure(state):
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "blocked_for_search_failure",
                    "title": "外部信息不足",
                    "message": "信息检索未成功，已停止后续行程规划，避免生成不可靠结果。",
                }
            )
            return "aggregate"
        if self._has_blocking_memory_gaps(state):
            if self.memory_manager:
                query = state.get("intention_data", {}).get("rewritten_query", "")
                if query:
                    self.memory_manager.short_term.set_pending_plan(
                        query,
                        {"reason": "memory_preference_gaps"},
                    )
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "blocked_for_memory_gaps",
                    "title": "等待用户补充偏好",
                    "message": "记忆智能体发现预算或节奏偏好缺失，已停止后续行程规划。",
                }
            )
            return "aggregate"
        if state.get("batch_index", 0) < len(state.get("batches", [])):
            return "run_batch"
        return "aggregate"

    def _extract_plan_search_requests(self, batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for result in reversed(batch_results):
            if result.get("agent_name") not in {"plan", "itinerary_planning"}:
                continue
            wrapped = result.get("result", {})
            data = wrapped.get("data", {}) if isinstance(wrapped, dict) else {}
            if not isinstance(data, dict) or data.get("planning_complete") is not False:
                continue
            raw_requests = data.get("search_requests") or data.get("info_requests") or data.get("missing_external_info") or []
            if isinstance(raw_requests, str):
                raw_requests = [{"keywords": raw_requests, "reason": raw_requests}]
            if not isinstance(raw_requests, list):
                return []
            normalized = []
            for item in raw_requests:
                if isinstance(item, str):
                    item = {"keywords": item, "reason": item}
                if not isinstance(item, dict):
                    continue
                keywords = str(item.get("keywords") or item.get("query") or "").strip()
                if not keywords:
                    continue
                normalized.append(
                    {
                        "keywords": keywords,
                        "reason": str(item.get("reason") or "规划智能体要求补充检索"),
                        "expected_output": str(item.get("expected_output") or "补充外部旅行信息"),
                    }
                )
            return normalized[:4]
        return []

    def _has_blocking_missing_info(self, state: OrchestrationState) -> bool:
        """Stop downstream search/plan when core trip fields are missing."""
        batch_index = state.get("batch_index", 0)
        batches = state.get("batches", [])
        if batch_index >= len(batches):
            return False

        remaining_tasks = [task for batch in batches[batch_index:] for task in batch]
        if not any(task.get("agent_name") in {"search", "memory", "plan"} for task in remaining_tasks):
            return False

        critical_fields = {"destination", "start_date", "duration_days", "budget_level", "pace_preference"}
        for result in state.get("results", []):
            if result.get("agent_name") not in {"clarification", "event_collection"}:
                continue
            data = result.get("result", {}).get("data", {})
            if not isinstance(data, dict):
                continue
            missing = {
                str(item)
                for item in data.get("missing_info", [])
                if item in critical_fields or str(item) in critical_fields
            }
            if missing:
                return True
            if any(data.get(field) in (None, "") for field in critical_fields):
                return True
        return False

    def _has_blocking_memory_gaps(self, state: OrchestrationState) -> bool:
        batch_index = state.get("batch_index", 0)
        batches = state.get("batches", [])
        if batch_index >= len(batches):
            return False

        remaining_tasks = [task for batch in batches[batch_index:] for task in batch]
        if not any(task.get("agent_name") == "plan" for task in remaining_tasks):
            return False

        for result in state.get("results", []):
            if result.get("agent_name") not in {"memory", "memory_query"}:
                continue
            data = result.get("result", {}).get("data", {})
            if isinstance(data, dict) and data.get("blocking_preference_gaps"):
                return True
        return False

    def _has_blocking_search_failure(self, state: OrchestrationState) -> bool:
        batch_index = state.get("batch_index", 0)
        batches = state.get("batches", [])
        if batch_index >= len(batches):
            return False

        remaining_tasks = [task for batch in batches[batch_index:] for task in batch]
        if not any(task.get("agent_name") == "plan" for task in remaining_tasks):
            return False

        for result in state.get("results", []):
            if result.get("agent_name") not in {"search", "information_query"}:
                continue
            wrapped = result.get("result", {})
            data = wrapped.get("data", {}) if isinstance(wrapped, dict) else {}
            if wrapped.get("status") == "error":
                return True
            if isinstance(data, dict) and data.get("query_success") is False:
                return True
        return False

    def _prepare_context(self, intention_data: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "reasoning": intention_data.get("reasoning", ""),
            "intents": intention_data.get("intents", []),
            "key_entities": intention_data.get("key_entities", {}),
            "rewritten_query": intention_data.get("rewritten_query", ""),
        }

        if self.memory_manager:
            recent_context = self.memory_manager.short_term.get_recent_context(3)
            context["recent_dialogue"] = recent_context
            context["user_preferences"] = self.memory_manager.long_term.get_preference()

        return context

    def _context_with_previous_results(
        self,
        context: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        enriched = dict(context)
        if previous_results:
            enriched["previous_results"] = previous_results

        for result in previous_results:
            agent_name = result.get("agent_name")
            data = result.get("result", {}).get("data", {})
            if agent_name == "clarification" and isinstance(data, dict):
                enriched["event_data"] = data
            elif agent_name == "memory" and isinstance(data, dict):
                if data.get("preferences"):
                    preferences = data.get("preferences")
                    if isinstance(preferences, dict):
                        enriched["user_preferences"] = preferences
                    elif self.memory_manager:
                        enriched["user_preferences"] = self.memory_manager.long_term.get_preference()
            elif agent_name == "search" and isinstance(data, dict):
                enriched["search_data"] = data
        if context.get("plan_search_requests"):
            enriched["plan_search_requests"] = context.get("plan_search_requests")
        if context.get("search_refinement_count"):
            enriched["search_refinement_count"] = context.get("search_refinement_count")
        return enriched

    async def _execute_parallel_agents(
        self,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not tasks:
            return []

        if len(tasks) == 1:
            task = tasks[0]
            result = await self._execute_agent(
                agent_name=task.get("agent_name"),
                context=context,
                reason=task.get("reason", ""),
                expected_output=task.get("expected_output", ""),
                previous_results=previous_results,
            )
            return [
                {
                    "agent_name": task.get("agent_name"),
                    "priority": task.get("priority", 0),
                    "result": result,
                }
            ]

        logger.info("Executing %d agents in parallel", len(tasks))
        parallel_coroutines = []
        for task in tasks:
            agent_name = task.get("agent_name")
            parallel_coroutines.append(
                (
                    agent_name,
                    task.get("priority", 0),
                    self._execute_agent(
                        agent_name=agent_name,
                        context=context,
                        reason=task.get("reason", ""),
                        expected_output=task.get("expected_output", ""),
                        previous_results=previous_results,
                    ),
                )
            )

        execution_results = await asyncio.gather(
            *[coro for _, _, coro in parallel_coroutines],
            return_exceptions=True,
        )

        results: List[Dict[str, Any]] = []
        for (agent_name, priority, _), exec_result in zip(parallel_coroutines, execution_results):
            if isinstance(exec_result, Exception):
                logger.error("Parallel agent execution failed: %s, error: %s", agent_name, exec_result)
                result = {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": {"error": str(exec_result)},
                    "message": f"并行执行失败: {str(exec_result)}",
                }
            else:
                result = exec_result

            results.append(
                {
                    "agent_name": agent_name,
                    "priority": priority,
                    "result": result,
                }
            )

        return results

    async def _execute_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
        reason: str,
        expected_output: str,
        previous_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_registry:
            logger.warning("Agent not registered: %s", agent_name)
            return {
                "status": "error",
                "message": f"智能体未注册: {agent_name}",
            }

        agent = self.agent_registry[agent_name]
        shared_state = {
            "context": context,
            "reason": reason,
            "expected_output": expected_output,
            "previous_results": previous_results,
            "current_agent": agent_name,
        }

        try:
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "agent_start",
                    "agent": agent_name,
                    "title": f"{self._display_agent_name(agent_name)}开始执行",
                    "message": reason or expected_output or "执行任务",
                }
            )
            if not hasattr(agent, "run"):
                raise TypeError(f"Agent {agent_name} does not implement run(state)")
            result = await agent.run(shared_state)
            if isinstance(result, dict) and "agent_output" in result:
                result = result["agent_output"]

            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "未知错误")
                self._emit_event(
                    {
                        "type": "chain",
                        "stage": "agent_error",
                        "agent": agent_name,
                        "title": f"{self._display_agent_name(agent_name)}执行失败",
                        "message": error_msg,
                    }
                )
                return {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": result,
                    "message": error_msg,
                }

            self._emit_event(
                {
                    "type": "chain",
                    "stage": "agent_done",
                    "agent": agent_name,
                    "title": f"{self._display_agent_name(agent_name)}执行完成",
                    "message": self._summarize_agent_result(agent_name, result),
                }
            )
            return {
                "status": "success",
                "agent_name": agent_name,
                "data": result,
            }
        except Exception as e:
            logger.error("Agent execution failed: %s, error: %s", agent_name, e)
            self._emit_event(
                {
                    "type": "chain",
                    "stage": "agent_error",
                    "agent": agent_name,
                    "title": f"{self._display_agent_name(agent_name)}执行失败",
                    "message": str(e),
                }
            )
            return {
                "status": "error",
                "agent_name": agent_name,
                "data": {"error": str(e)},
                "message": f"智能体执行失败: {str(e)}",
            }

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        intention_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        aggregated = {
            "status": "completed",
            "intention": {
                "intents": intention_data.get("intents", []),
                "key_entities": intention_data.get("key_entities", {}),
            },
            "agents_executed": len(results),
            "results": [],
        }

        for result in results:
            aggregated["results"].append(
                {
                    "agent_name": result["agent_name"],
                    "priority": result["priority"],
                    "status": result["result"].get("status", "unknown"),
                    "data": result["result"].get("data", {}),
                }
            )

        errors = [r for r in results if r["result"].get("status") == "error"]
        if errors:
            aggregated["status"] = "partial_failure"
            aggregated["errors"] = len(errors)

        suggestions = self._collect_suggested_replies(results)
        if suggestions:
            aggregated["suggested_replies"] = suggestions
        input_requests = self._collect_input_requests(results)
        if input_requests:
            aggregated["input_requests"] = input_requests

        return aggregated

    def _collect_suggested_replies(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        suggestions: List[Dict[str, str]] = []
        seen = set()
        for result in results:
            data = result.get("result", {}).get("data", {})
            if not isinstance(data, dict):
                continue
            for item in data.get("suggested_options", []) or []:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                message = str(item.get("message", "")).strip()
                description = str(item.get("description", "")).strip()
                if not label or not message or message in seen:
                    continue
                seen.add(message)
                suggestion = {"label": label, "message": message}
                if description:
                    suggestion["description"] = description
                suggestions.append(suggestion)
        return suggestions[:6]

    def _collect_input_requests(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        seen = set()
        for result in results:
            if result.get("agent_name") not in {"clarification", "event_collection"}:
                continue
            data = result.get("result", {}).get("data", {})
            if not isinstance(data, dict):
                continue
            missing = [str(item) for item in data.get("missing_info", []) or []]
            for field in missing:
                request = self._build_input_request(field)
                if not request or request["field"] in seen:
                    continue
                seen.add(request["field"])
                requests.append(request)
        return requests

    def _build_input_request(self, field: str) -> Optional[Dict[str, Any]]:
        if field == "start_date":
            return {
                "field": "start_date",
                "label": "出发日期",
                "type": "date",
                "placeholder": "选择出发日期",
            }
        if field == "duration_days":
            return {
                "field": "duration_days",
                "label": "行程天数",
                "type": "select",
                "options": [
                    {"label": f"{day}天", "value": str(day)}
                    for day in range(1, 8)
                ],
            }
        if field == "origin":
            return {"field": "origin", "label": "出发地", "type": "text", "placeholder": "例如：成都"}
        if field == "destination":
            return {"field": "destination", "label": "目的地", "type": "text", "placeholder": "例如：北京"}
        return None

    def _display_agent_name(self, agent_name: str) -> str:
        return {
            "memory": "记忆与偏好智能体",
            "search": "信息检索智能体",
            "plan": "行程规划智能体",
            "clarification": "事项收集智能体",
        }.get(agent_name, agent_name or "智能体")

    def _summarize_agent_result(self, agent_name: str, result: Any) -> str:
        if not isinstance(result, dict):
            return "已返回执行结果"

        if agent_name == "clarification":
            origin = result.get("origin") or "未知出发地"
            destination = result.get("destination") or "未知目的地"
            days = result.get("duration_days")
            missing = result.get("missing_info") or []
            parts = [f"{origin} -> {destination}"]
            if days:
                parts.append(f"{days}天")
            if missing:
                parts.append(f"待补充：{'、'.join(str(item) for item in missing[:4])}")
            return "；".join(parts)

        if agent_name == "search":
            data = result.get("results") if isinstance(result.get("results"), dict) else result
            pois = data.get("pois") if isinstance(data, dict) else None
            weather = data.get("weather") if isinstance(data, dict) else None
            summary = data.get("summary") if isinstance(data, dict) else ""
            parts = []
            if pois:
                parts.append(f"POI {len(pois)} 条")
            if weather:
                parts.append("天气已获取")
            if summary and not parts:
                parts.append(str(summary)[:80])
            return "；".join(parts) or "外部信息已获取"

        if agent_name == "plan":
            itinerary = result.get("itinerary") or {}
            title = itinerary.get("title")
            daily_plans = itinerary.get("daily_plans") or []
            return f"{title or '行程规划'}；{len(daily_plans)} 天安排" if itinerary else "已生成规划结果"

        if agent_name == "memory":
            if result.get("answer"):
                return str(result.get("answer"))[:120]
            prefs = result.get("preferences")
            if prefs:
                return "已读取或更新用户偏好"
            return "已查询长期记忆"

        return "已返回执行结果"

    def _update_memory(self, intention_data: Dict[str, Any], results: List[Dict[str, Any]]):
        if not self.memory_manager:
            return

        for result in results:
            agent_name = result["agent_name"]
            data = result["result"].get("data", {})

            if agent_name in {"memory", "preference"} and isinstance(data, dict):
                preferences_data = data.get("preferences", {})
                if isinstance(preferences_data, list):
                    for pref_item in preferences_data:
                        if not isinstance(pref_item, dict):
                            continue

                        pref_type = pref_item.get("type")
                        pref_value = pref_item.get("value")
                        pref_action = pref_item.get("action", "replace")

                        if not pref_type or not pref_value:
                            continue

                        if pref_action == "append":
                            current_prefs = self.memory_manager.long_term.get_preference()
                            existing_value = current_prefs.get(pref_type)
                            if isinstance(existing_value, list):
                                if pref_value not in existing_value:
                                    existing_value.append(pref_value)
                                self.memory_manager.long_term.save_preference(pref_type, existing_value)
                            else:
                                new_list = [existing_value, pref_value] if existing_value else [pref_value]
                                self.memory_manager.long_term.save_preference(pref_type, new_list)
                        else:
                            self.memory_manager.long_term.save_preference(pref_type, pref_value)
                elif isinstance(preferences_data, dict):
                    for pref_type, value in preferences_data.items():
                        if value and pref_type != "has_preferences" and pref_type != "error":
                            self.memory_manager.long_term.save_preference(pref_type, value)

            if agent_name in {"plan", "itinerary_planning"} and isinstance(data, dict):
                itinerary = data.get("itinerary", {})
                if itinerary:
                    event_data = {}
                    for r in results:
                        if r["agent_name"] in {"clarification", "event_collection"}:
                            event_data = r["result"].get("data", {})
                            break

                    destination = event_data.get("destination")
                    if destination:
                        self.memory_manager.long_term.save_trip_history(
                            {
                                "origin": event_data.get("origin"),
                                "destination": destination,
                                "start_date": event_data.get("start_date"),
                                "end_date": event_data.get("end_date"),
                                "purpose": event_data.get("trip_purpose", "旅游"),
                            }
                        )
                        self.memory_manager.short_term.clear_pending_plan()

        logger.info("Memory updated after orchestration")


# Backward compatibility: older code/tests may still import OrchestrationAgent.
OrchestrationAgent = AgentScheduler
