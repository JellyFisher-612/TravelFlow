#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TravelFlow 旅游出行助手 - CLI 交互界面
使用 Rich 库实现美观的终端交互
"""
import asyncio
import sys
import os
import io
from typing import Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from config import LLM_CONFIG, RESILIENCE_CONFIG
from context.memory_manager import MemoryManager
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.langchain_runtime import build_chat_model
from utils.langsmith_setup import setup_langsmith_tracing
from utils.llm_resilience import run_health_check as check_llm_health
from query_orchestrator import QueryOrchestrator, should_skip_long_term_summary
from agents.agent_scheduler import AgentScheduler
from agents.intent_recognition import IntentRecognition
from agents.main_agent import MainAgent
# 移除其他智能体的导入，改用懒加载


class TravelFlowCLI:
    """TravelFlow 旅游出行助手 CLI"""

    def __init__(self):
        """初始化 CLI"""
        self.console = Console()
        self.user_id = None
        self.session_id = None
        self.memory_manager = None
        self.main_agent = None
        self.agent_scheduler = None
        self.intent_recognition = None
        self.orchestrator = None  # backward-compatible alias
        self.intention_agent = None  # backward-compatible alias
        self.model = None
        self._agent_cache = {}  # 智能体缓存
        self.circuit_breaker = None  # 在 initialize_system 中从 RESILIENCE_CONFIG 初始化
        self.query_orchestrator = QueryOrchestrator(self)
        self._web_trace_events = []
        self._runtime_event_callback = None

    def set_runtime_event_callback(self, callback):
        """设置运行时事件回调，用于 Web 流式展示。"""
        self._runtime_event_callback = callback
        if self.main_agent and hasattr(self.main_agent, "set_event_callback"):
            self.main_agent.set_event_callback(self._emit_runtime_event if callback else None)
        if self.agent_scheduler and hasattr(self.agent_scheduler, "set_event_callback"):
            self.agent_scheduler.set_event_callback(self._emit_runtime_event if callback else None)

    def _emit_runtime_event(self, message):
        """向外发送运行时事件（如果设置了回调）。"""
        if self._runtime_event_callback and message:
            try:
                self._runtime_event_callback(message)
            except Exception:
                pass

    def _on_agent_registry_event(self, message):
        """接收 LazyAgentRegistry 的加载事件，供 Web 端展示。"""
        if message:
            self._web_trace_events.append(message)
            self._emit_runtime_event(message)

    def print_banner(self):
        """打印欢迎横幅"""
        self.console.print("\n[bold cyan]🌏 TravelFlow 旅游出行助手[/bold cyan] - 让旅行规划更简单\n", style="bold")

    def print_help(self):
        """打印帮助信息"""
        table = Table(title="命令列表", show_header=True, header_style="bold magenta")
        table.add_column("命令", style="cyan", width=20)
        table.add_column("说明", style="white")

        table.add_row("help", "显示此帮助信息")
        table.add_row("status", "查看当前状态和记忆")
        table.add_row("health", "检查 LLM 服务是否可用")
        table.add_row("clear", "清空当前任务（保留长期记忆）")
        table.add_row("history", "查看历史行程")
        table.add_row("preferences", "查看用户偏好")
        table.add_row("exit", "退出程序")
        table.add_row("", "")
        table.add_row("[自然语言]", "直接输入您的需求，如：")
        table.add_row("", "  - 我要从上海去北京玩三天")
        table.add_row("", "  - 帮我查杭州明天的天气")
        table.add_row("", "  - 我喜欢轻松一点的行程")

        self.console.print(table)

    async def initialize_system(
        self,
        user_id: Optional[str] = None,
        interactive: bool = True,
        session_id: Optional[str] = None,
    ):
        """初始化系统 - 使用懒加载优化启动速度"""
        # 初始化 LangSmith tracing（若未开启或未配置 key，则自动跳过）
        setup_langsmith_tracing()

        # 获取用户信息
        if user_id:
            self.user_id = user_id
        elif interactive:
            self.user_id = Prompt.ask(
                "用户ID",
                default="default_user"
            )
        else:
            self.user_id = "default_user"

        # 生成或复用会话ID
        if session_id:
            self.session_id = session_id
        else:
            import uuid
            self.session_id = str(uuid.uuid4())[:8]

        with self.console.status("初始化中...", spinner="dots"):
            # 初始化 LangChain ChatOpenAI 模型
            self.model = build_chat_model()

            # 初始化记忆管理器（传入LLM模型用于总结）
            self.memory_manager = MemoryManager(
                user_id=self.user_id,
                session_id=self.session_id,
                llm_model=self.model
            )

            # 初始化主智能体的意图识别能力（必须预加载）
            self.intent_recognition = IntentRecognition(
                name="IntentRecognition",
                model=self.model
            )
            self.intention_agent = self.intent_recognition

            # 使用懒加载注册器（智能体在首次使用时才加载）
            from agents.lazy_agent_registry import LazyAgentRegistry
            self._agent_cache = {}
            lazy_registry = LazyAgentRegistry(
                model=self.model, 
                cache=self._agent_cache,
                memory_manager=self.memory_manager,
                event_callback=self._on_agent_registry_event,
            )

            # 预先加载关键智能体（可选，利用 preload）
            # lazy_registry.preload("memory", "search")

            # 初始化主智能体可调用的业务智能体编排工具
            self.agent_scheduler = AgentScheduler(
                name="AgentScheduler",
                agent_registry=lazy_registry,
                memory_manager=self.memory_manager,
                event_callback=self._emit_runtime_event if self._runtime_event_callback else None,
            )
            self.orchestrator = self.agent_scheduler

            # 初始化主智能体：负责用户沟通、意图识别和按需委托业务智能体。
            self.main_agent = MainAgent(
                name="MainAgent",
                model=self.model,
                intent_recognition=self.intent_recognition,
                agent_scheduler=self.agent_scheduler,
                memory_manager=self.memory_manager,
                event_callback=self._emit_runtime_event if self._runtime_event_callback else None,
                intention_callback=self._emit_intention_chain,
            )

            # 熔断器（连接与可用性）
            rc = RESILIENCE_CONFIG
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=rc.get("circuit_failure_threshold", 5),
                recovery_timeout_sec=rc.get("circuit_recovery_timeout_sec", 60.0),
                half_open_successes=rc.get("circuit_half_open_successes", 2),
            )

        if interactive:
            self.console.print(f"✓ 就绪 (用户: {self.user_id}) - 输入 help 查看帮助\n", style="green")

    async def _execute_query(self, user_input: str) -> tuple[Optional[dict], Optional[str]]:
        """执行用户查询并返回结构化结果，不负责终端输出。"""
        return await self.query_orchestrator.execute_query(user_input)

    def _should_skip_long_term_summary(self, user_input: str) -> bool:
        """Compatibility wrapper for older tests/callers."""
        return should_skip_long_term_summary(user_input)

    def _emit_intention_chain(self, intention_data: dict):
        """Emit user-facing orchestration reasoning without exposing hidden chain-of-thought."""
        if not isinstance(intention_data, dict):
            return

        intents = intention_data.get("intents") or []
        intent_items = []
        for item in intents:
            if not isinstance(item, dict):
                continue
            intent_items.append(
                {
                    "type": item.get("type", ""),
                    "confidence": item.get("confidence", ""),
                    "description": item.get("description", ""),
                    "reason": item.get("reason", ""),
                }
            )

        self._emit_runtime_event(
            {
                "type": "chain",
                "stage": "intent",
                "title": "意图识别完成",
                "message": intention_data.get("reasoning", "") or "已识别用户需求并生成调度策略",
                "rewritten_query": intention_data.get("rewritten_query", ""),
                "entities": intention_data.get("key_entities", {}),
                "items": intent_items,
            }
        )

    def render_result_text(self, result_data: dict, include_agent_trace: bool = False) -> str:
        """将结构化结果渲染为纯文本，便于 Web 返回。"""
        temp_console = Console(record=True, force_terminal=False, file=io.StringIO())
        original_console = self.console
        try:
            self.console = temp_console
            if include_agent_trace:
                self._display_agents_called(result_data)
                self.console.print()
            self._display_results(result_data)
            return temp_console.export_text(clear=True).strip()
        finally:
            self.console = original_console

    async def process_query_for_web(self, user_input: str) -> tuple[str, Optional[dict], Optional[str], list[str]]:
        """Web 调用入口：返回可展示文本、原始结果、错误信息和加载轨迹。"""
        result_data, error_message = await self._execute_query(user_input)
        trace = list(self._web_trace_events)
        if error_message:
            return f"❌ {error_message}", None, error_message, trace
        if not result_data:
            fallback = "未能获取有效结果，请重新描述您的需求。"
            return fallback, None, fallback, trace
        return self.render_result_text(result_data), result_data, None, trace

    async def process_query(self, user_input: str):
        """
        处理用户查询（原逻辑保留；仅在入口加熔断检查、对 LLM 调用加重试）
        """
        with self.console.status("思考中...", spinner="dots"):
            result_data, error_message = await self._execute_query(user_input)

        if error_message:
            self.console.print(f"❌ {error_message}", style="bold red")
            return

        if not result_data:
            self.console.print("未能获取有效结果，请重新描述您的需求。", style="yellow")
            return

        self._display_agents_called(result_data)
        self.console.print()
        self._display_results(result_data)

    def _display_agents_called(self, result_data: dict):
        """显示调用的智能体列表"""
        results = result_data.get("results", [])
        if not results:
            return

        # 收集所有调用的智能体
        agents_called = []
        for result in results:
            agent_name = result.get("agent_name", "")
            status = result.get("status", "")

            display_name = self._get_agent_display_name(agent_name)

            # 根据状态添加标记
            if status == "success":
                agents_called.append(f"{display_name} ✓")
            elif status == "error":
                agents_called.append(f"{display_name} ✗")
            else:
                agents_called.append(f"{display_name} ?")

        if agents_called:
            self.console.print()
            self.console.print(f"🤖 调用智能体: {', '.join(agents_called)}", style="dim")

    def _display_results(self, result_data: dict):
        """显示执行结果 - 确保永远有回复"""
        self.console.print()

        # 获取结果列表
        results = result_data.get("results", [])

        if not results:
            # 情况1: 没有任何智能体被调用
            status = result_data.get("status", "unknown")
            if status == "no_agents":
                direct_answer = result_data.get("direct_answer") or result_data.get("message")
                if direct_answer and direct_answer != "没有可调度的智能体":
                    self.console.print(direct_answer)
                else:
                    self.console.print("✓ 好的，我已记录下来。", style="green")
                    self.console.print("\n💡 您可以继续补充信息，或者尝试：", style="dim")
                    self.console.print("  • 规划行程：「帮我规划去北京的行程」", style="dim")
                    self.console.print("  • 查询信息：「北京的天气怎么样」", style="dim")
                    self.console.print("  • 查信息：「杭州明天天气怎么样」", style="dim")
            else:
                self.console.print("未能获取有效结果，请重新描述您的需求。", style="yellow")
        else:
            # 情况2: 有智能体被调用，生成人性化回复
            has_output = self._generate_human_response(results)

            # 情况3: 智能体执行了但没有显示内容（兜底）
            if not has_output:
                self.console.print("✓ 已处理您的请求。", style="green")

        self.console.print()

    async def _get_long_term_summary(self, user_input: str = "") -> str:
        """
        生成长期记忆摘要，用于传递给 MainAgent 的 IntentRecognition
        只使用结构化长期记忆；旧聊天记录仅作为日志，不在用户轮次中读取或总结。

        Args:
            user_input: 用户输入，用于筛选相关历史行程

        Returns:
            格式化的长期记忆摘要
        """
        summary_parts = []

        # 1. 用户偏好信息（始终加载）
        prefs = self.memory_manager.long_term.get_preference()
        if prefs:
            pref_lines = ["【用户背景信息】（来自长期记忆，可用于推断缺失信息）"]

            # 遍历所有偏好，全部加载
            for pref_key, pref_value in prefs.items():
                if pref_value:  # 只添加有值的偏好
                    # 如果是列表，用逗号连接
                    if isinstance(pref_value, list):
                        pref_lines.append(f"• {pref_key}: {', '.join(pref_value)}")
                    else:
                        pref_lines.append(f"• {pref_key}: {pref_value}")

            # 只有在有具体偏好内容时才添加
            if len(pref_lines) > 1:
                summary_parts.extend(pref_lines)

        # 2. 智能筛选相关历史行程
        all_trips = self.memory_manager.long_term.get_trip_history(limit=None)
        if all_trips:
            # 筛选相关的行程（地点匹配）
            relevant_trips = []
            other_trips = []

            for trip in all_trips:
                origin = trip.get("origin", "") or ""
                destination = trip.get("destination", "") or ""

                # 如果用户输入提到了这个行程的地点，标记为相关
                if (origin and origin in user_input) or (destination and destination in user_input):
                    relevant_trips.append(trip)
                else:
                    other_trips.append(trip)

            # 优先显示相关的，再补充最近的
            trips_to_show = relevant_trips[:2] + other_trips[:1]  # 2条相关 + 1条最近

            if trips_to_show:
                summary_parts.append("\n【历史行程】")
                for i, trip in enumerate(trips_to_show[:3], 1):
                    origin = trip.get("origin", "未知")
                    destination = trip.get("destination", "未知")
                    start_date = trip.get("start_date", "")
                    purpose = trip.get("purpose", "")

                    # 标记相关性
                    relevance_mark = "✦ " if trip in relevant_trips else ""
                    summary_parts.append(
                        f"{i}. {relevance_mark}{origin} → {destination} ({start_date}) - {purpose}"
                    )

        # 3. 行为反馈是结构化长期记忆，可作为规划约束或质量参考。
        feedback = self.memory_manager.long_term.get_behavior_feedback(limit=3)
        if feedback:
            summary_parts.append("\n【用户历史反馈】")
            for item in feedback[-3:]:
                if isinstance(item, dict):
                    value = item.get("feedback", item)
                else:
                    value = item
                if value:
                    summary_parts.append(f"• {value}")

        return "\n".join(summary_parts) if summary_parts else ""

    def _generate_human_response(self, results: list) -> bool:
        """
        根据结果生成人性化的回复
        """
        return self._format_agent_results(results)

    def _format_agent_results(self, results: list, all_results: Optional[list] = None) -> bool:
        """格式化各业务智能体执行结果。"""
        has_output = False
        all_results = all_results if all_results is not None else results
        has_itinerary = any(r.get("agent_name") in {"plan", "itinerary_planning"} for r in all_results)
        has_completed_plan = any(
            r.get("agent_name") in {"plan", "itinerary_planning"}
            and isinstance(r.get("data"), dict)
            and r.get("data", {}).get("planning_complete") is True
            for r in all_results
        )

        for result in results:
            agent_name = result.get("agent_name", "")
            status = result.get("status", "")
            data = result.get("data", {})

            # 处理失败的智能体
            if status == "error":
                has_output = self._format_error_section([result]) or has_output
                continue

            # 只处理成功的智能体。
            if status != "success":
                continue

            current_agent_shown = False
            if agent_name in {"plan", "itinerary_planning"}:
                if data.get("planning_complete") is False and has_completed_plan:
                    continue
                current_agent_shown = self._format_travel_plan(data)
            elif agent_name in {"intent", "intent_recognition", "intention", "intention_agent"}:
                current_agent_shown = self._format_intent_section(data)
            elif agent_name in {"memory", "preference"} and data.get("preferences") is not None:
                current_agent_shown = self._format_preference_update(data, has_itinerary)
            elif agent_name in {"clarification", "event_collection"}:
                current_agent_shown = self._format_clarification_section(data, has_itinerary)
            elif agent_name in {"search", "information_query"}:
                current_agent_shown = self._format_search_section(data)
            elif agent_name in {"memory", "memory_query"}:
                current_agent_shown = self._format_memory_section(data, has_itinerary)

            if not current_agent_shown:
                current_agent_shown = self._format_generic_agent_result(agent_name, data)

            if current_agent_shown:
                has_output = True

        return has_output

    def _format_intent_section(self, intent_data: dict) -> bool:
        """格式化意图识别结果。"""
        if not isinstance(intent_data, dict):
            return False

        direct_answer = intent_data.get("direct_answer") or intent_data.get("message")
        if direct_answer and direct_answer != "没有可调度的智能体":
            self.console.print(f"\n{direct_answer}")
            return True
        return False

    def _format_travel_plan(self, plan_data: dict) -> bool:
        """格式化行程规划输出。"""
        itinerary = plan_data.get("itinerary")
        if not itinerary and "data" in plan_data and isinstance(plan_data["data"], dict):
            itinerary = plan_data["data"].get("itinerary")

        if not itinerary:
            return False

        title = itinerary.get('title', '行程规划')
        self.console.print(f"\n✈️  [bold cyan]{title}[/bold cyan]")
        self.console.print(f"时长: {itinerary.get('duration', '未知')}\n")
        self._format_hard_constraints(itinerary.get("hard_constraints") or [])
        self._format_budget_estimate(itinerary.get("budget_estimate") or {})
        self._format_daily_plans(itinerary.get("daily_plans", []))
        self._format_fallback_options(itinerary.get("fallback_options") or [])
        self._format_plan_notes(itinerary.get("notes", []))
        return True

    def _format_hard_constraints(self, hard_constraints: list) -> None:
        if not hard_constraints:
            return

        self.console.print("[bold]必须先核验的硬约束[/bold]")
        for item in hard_constraints:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "核验项")
            status = item.get("status", "needs_official_check")
            action = item.get("action", "")
            self.console.print(f"  • {name} [{status}]")
            if action:
                self.console.print(f"    {action}", style="dim")
        self.console.print()

    def _format_budget_estimate(self, budget: dict) -> None:
        if not isinstance(budget, dict) or not budget.get("items"):
            return

        self.console.print("[bold]预算粗估[/bold]")
        for item in budget.get("items", []):
            if isinstance(item, dict):
                self.console.print(f"  • {item.get('name', '')}: {item.get('range', '')}")
        if budget.get("note"):
            self.console.print(f"  {budget['note']}", style="dim")
        self.console.print()

    def _format_daily_plans(self, daily_plans: list) -> None:
        for day_plan in daily_plans:
            day_num = day_plan.get("day", 1)
            self.console.print(f"[bold yellow]第 {day_num} 天[/bold yellow]")

            activities = day_plan.get("activities") or day_plan.get("time_slots") or []
            for slot in activities:
                time = slot.get("time", "")
                activity = slot.get("activity") or slot.get("location") or ""
                description = slot.get("description", "")
                transport = slot.get("transport", "")

                self.console.print(f"  {time} - {activity}")
                if description:
                    self.console.print(f"    {description}", style="dim")
                if transport:
                    self.console.print(f"    🚇 {transport}", style="dim")

            meals = day_plan.get("meals", {})
            if meals:
                self.console.print()
                if meals.get("lunch"):
                    self.console.print(f"  🍜 {meals['lunch']}", style="dim")
                if meals.get("dinner"):
                    self.console.print(f"  🍽️  {meals['dinner']}", style="dim")
            self.console.print()

    def _format_fallback_options(self, fallbacks: list) -> None:
        if not fallbacks:
            return

        self.console.print("[bold]备选方案[/bold]")
        for item in fallbacks:
            if isinstance(item, dict):
                self.console.print(f"  • {item.get('scenario', '情况')}: {item.get('option', '')}")
        self.console.print()

    def _format_plan_notes(self, notes: list) -> None:
        if not notes:
            return

        self.console.print("[bold]📌 注意事项[/bold]")
        for note in notes:
            self.console.print(f"  • {note}")

    def _format_memory_section(self, memory_data: dict, has_itinerary: bool = False) -> bool:
        """格式化记忆查询结果。"""
        query_result = memory_data.get("answer") or memory_data.get("result") or memory_data.get("content")
        if not query_result and "data" in memory_data and isinstance(memory_data["data"], dict):
            inner = memory_data["data"]
            query_result = inner.get("answer") or inner.get("result") or inner.get("content")

        if query_result and not has_itinerary:
            self.console.print(f"\n{query_result}")
            return True
        if memory_data.get("follow_up_question") and not has_itinerary:
            self.console.print(f"\n{memory_data['follow_up_question']}", style="dim")
            return True
        return False

    def _format_preference_update(self, memory_data: dict, has_itinerary: bool = False) -> bool:
        """格式化偏好写入结果。"""
        raw_prefs = memory_data.get("preferences")
        if not raw_prefs and "data" in memory_data and isinstance(memory_data["data"], dict):
            raw_prefs = memory_data["data"].get("preferences")

        if isinstance(raw_prefs, dict):
            prefs_list = raw_prefs.get("preferences", [])
        else:
            prefs_list = raw_prefs if isinstance(raw_prefs, list) else []

        if not prefs_list:
            err = memory_data.get("error", "")
            if err:
                self.console.print(f"偏好未保存: {err}", style="yellow")
                return True
            return False

        self.console.print("✓ [bold green]已更新您的偏好设置[/bold green]")
        type_names = {
            "home_location": "常驻地",
            "transportation_preference": "交通偏好",
            "hotel_brands": "酒店偏好",
            "airlines": "航空公司偏好",
            "seat_preference": "座位偏好",
            "meal_preference": "餐食偏好",
            "budget_level": "预算等级"
        }
        for pref in prefs_list:
            pref_type = pref.get("type", "")
            pref_value = pref.get("value", "")
            action = pref.get("action", "replace")
            display_type = type_names.get(pref_type, pref_type)
            action_text = "追加" if action == "append" else "设置为"
            self.console.print(f"  • {display_type} {action_text} [cyan]{pref_value}[/cyan]")
        if not has_itinerary:
            self.console.print("\n💡 下次规划行程时会参考这些偏好。", style="dim")
        return True

    def _format_clarification_section(self, data: dict, has_itinerary: bool) -> bool:
        origin = data.get("origin") or data.get("data", {}).get("origin")
        destination = data.get("destination") or data.get("data", {}).get("destination")
        start_date = data.get("start_date") or data.get("data", {}).get("start_date")
        end_date = data.get("end_date") or data.get("data", {}).get("end_date")
        budget_level = data.get("budget_level") or data.get("data", {}).get("budget_level")
        pace_preference = data.get("pace_preference") or data.get("data", {}).get("pace_preference")
        missing_info = data.get("missing_info") or data.get("data", {}).get("missing_info") or []
        summary = data.get("summary") or data.get("data", {}).get("summary")

        info_shown = False
        if not has_itinerary:
            known_parts = []
            if origin:
                known_parts.append(f"从{origin}出发")
            if destination:
                known_parts.append(f"去{destination}")
            if start_date:
                known_parts.append(f"{start_date}出发")
            if end_date:
                known_parts.append(f"{end_date}返程")
            if budget_level:
                known_parts.append(f"{budget_level}预算")
            if pace_preference:
                known_parts.append(f"{pace_preference}节奏")

            if known_parts:
                self.console.print(f"\n我先确认一下：你想{('，'.join(known_parts))}。")
            elif summary:
                self.console.print(f"\n{summary}")
            else:
                self.console.print("\n我可以继续帮你规划，不过还需要先确认几个关键信息。")
            info_shown = True

        if missing_info:
            readable_missing = [self._format_missing_field(item) for item in missing_info]
            self.console.print(f"\n为了把行程安排具体，我还需要你补充：{'、'.join(readable_missing)}。", style="yellow")
            example = self._build_missing_info_example(missing_info)
            if example:
                self.console.print(f"你可以直接在下面的表单补充，也可以回复：{example}。")
            else:
                self.console.print("你可以直接在下面的表单补充，我会继续规划。")
            if any(item in {"budget_level", "pace_preference"} for item in missing_info):
                self.console.print("也可以先点下面的预算或节奏选项，我会继续追问剩下的信息。")
            info_shown = True

        return info_shown

    def _format_search_section(self, data: dict) -> bool:
        query_results = data.get("results")
        if not query_results and "data" in data and isinstance(data["data"], dict):
            query_results = data["data"].get("results")
        if not query_results:
            query_results = data

        if not isinstance(query_results, dict):
            query_results = {}

        summary = query_results.get("summary", "")
        sources = query_results.get("sources", []) or []
        message = query_results.get("message", "")
        error = query_results.get("error", "")
        shown = False

        if summary:
            self.console.print(f"\n{summary}")
            shown = True
        elif message:
            self.console.print(f"\n{message}", style="dim")
            shown = True
        elif error:
            self.console.print(f"\n{error}", style="yellow")
            shown = True

        if sources:
            self.console.print("\n[bold]参考来源[/bold]")
            for i, source in enumerate(sources[:3], 1):
                if isinstance(source, dict):
                    label = source.get("url") or source.get("title") or str(source)
                else:
                    label = str(source)
                self.console.print(f"  {i}. {label}", style="dim")
            shown = True

        return shown

    def _format_error_section(self, errors: list) -> bool:
        """格式化智能体错误信息。"""
        shown = False
        for error in errors:
            if isinstance(error, dict):
                agent_name = error.get("agent_name", "")
                data = error.get("data", {})
                error_msg = data.get("error", "未知错误") if isinstance(data, dict) else str(data)
            else:
                agent_name = ""
                error_msg = str(error)

            agent_display_name = self._get_agent_display_name(agent_name)
            self.console.print(f"❌ {agent_display_name}执行失败: {error_msg}", style="red")
            shown = True
        return shown

    def _format_generic_agent_result(self, agent_name: str, data: dict) -> bool:
        common_keys = ["answer", "content", "result", "message", "summary", "text", "description"]
        fallback_content = ""

        for k in common_keys:
            if k in data and isinstance(data[k], str) and data[k].strip():
                fallback_content = data[k]
                break

        if not fallback_content and "data" in data and isinstance(data["data"], dict):
            for k in common_keys:
                if k in data["data"] and isinstance(data["data"][k], str) and data["data"][k].strip():
                    fallback_content = data["data"][k]
                    break

        if fallback_content:
            self.console.print(f"\n{fallback_content}")
            return True

        agent_display_name = self._get_agent_display_name(agent_name)
        self.console.print(f"✓ {agent_display_name}已完成", style="green")
        return True

    def _get_agent_display_name(self, agent_name: str) -> str:
        """获取智能体的显示名称"""
        agent_display_names = {
            "orchestrator": "编排工具",
            "memory": "记忆与偏好",
            "search": "信息检索",
            "plan": "行程规划",
            "clarification": "事项收集",
            "event_collection": "事项收集",
            "preference": "偏好管理",
            "itinerary_planning": "行程规划",
            "information_query": "信息查询",
            "memory_query": "记忆查询",
        }
        return agent_display_names.get(agent_name, agent_name)

    def _format_missing_field(self, field: str) -> str:
        field_names = {
            "origin": "出发地",
            "destination": "目的地",
            "start_date": "出发日期",
            "end_date": "返程日期",
            "duration_days": "行程天数",
            "budget_level": "预算偏好",
            "pace_preference": "行程节奏",
            "return_location": "返程地",
            "trip_purpose": "出行目的",
            "budget": "预算",
            "companions": "同行人",
        }
        return field_names.get(str(field), str(field))

    def _build_missing_info_example(self, missing_info: list) -> str:
        examples = {
            "origin": "从成都出发",
            "destination": "去北京",
            "start_date": "5月1日出发",
            "duration_days": "玩3天",
            "budget_level": "舒适型预算",
            "pace_preference": "轻松节奏",
        }
        parts = [examples[item] for item in missing_info if item in examples]
        return "，".join(parts)

    def show_status(self):
        """显示当前状态"""
        # 记忆统计
        full_context = self.memory_manager.get_full_context()
        short_term_stats = full_context["short_term"]["statistics"]
        long_term_stats = full_context["long_term"]["statistics"]

        memory_table = Table(title="记忆状态", show_header=True, header_style="bold magenta")
        memory_table.add_column("类型", style="cyan")
        memory_table.add_column("状态", style="white")

        memory_table.add_row(
            "短期记忆",
            f"{short_term_stats['total_messages']} 条消息"
        )
        memory_table.add_row(
            "长期记忆",
            f"{long_term_stats['total_trips']} 次行程"
        )
        memory_table.add_row(
            "已加载智能体",
            f"{len(self._agent_cache)} 个"
        )

        self.console.print(memory_table)
        self.console.print()

        # 历史对话
        recent_messages = self.memory_manager.short_term.get_recent_context(n_turns=5)
        if recent_messages:
            dialogue_table = Table(title="最近对话 (最多5轮)", show_header=True, header_style="bold cyan")
            dialogue_table.add_column("角色", style="cyan", width=8)
            dialogue_table.add_column("内容", style="white", width=60)
            dialogue_table.add_column("时间", style="dim", width=12)

            for msg in recent_messages:
                role_name = "👤 用户" if msg["role"] == "user" else "🤖 助手"
                content = msg["content"]

                # 截断过长的内容
                if len(content) > 100:
                    content = content[:100] + "..."

                # 格式化时间
                timestamp = msg.get("timestamp", "")
                if timestamp:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except Exception:
                        time_str = ""
                else:
                    time_str = ""

                dialogue_table.add_row(role_name, content, time_str)

            self.console.print(dialogue_table)
            self.console.print()

    async def run_health_check(self):
        """在会话内执行健康检查并显示熔断器状态"""
        if self.circuit_breaker:
            status = self.circuit_breaker.get_status()
            self.console.print(f"[bold]熔断器[/bold]: {status['state']}", style="cyan")
        ok, msg = await check_llm_health(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
            model_name=LLM_CONFIG["model_name"],
            timeout_sec=RESILIENCE_CONFIG.get("health_check_timeout_sec", 10.0),
        )
        if ok:
            self.console.print("LLM 服务: [green]正常[/green]", style="bold")
        else:
            self.console.print(f"LLM 服务: [red]不可用[/red] - {msg}", style="bold")
        self.console.print()

    def show_history(self):
        """显示历史行程"""
        history = self.memory_manager.long_term.get_trip_history(10)

        if not history:
            self.console.print("暂无历史行程", style="yellow")
            return

        table = Table(title="历史行程", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan")
        table.add_column("出发地", style="white")
        table.add_column("目的地", style="white")
        table.add_column("日期", style="white")
        table.add_column("目的", style="white")

        for trip in history:
            table.add_row(
                trip.get("trip_id", ""),
                trip.get("origin", ""),
                trip.get("destination", ""),
                trip.get("start_date", ""),
                trip.get("purpose", "")
            )

        self.console.print(table)

    def show_preferences(self):
        """显示用户偏好"""
        prefs = self.memory_manager.long_term.get_preference()

        table = Table(title="用户偏好", show_header=True, header_style="bold magenta")
        table.add_column("类型", style="cyan")
        table.add_column("值", style="white")

        for key, value in prefs.items():
            if value:
                table.add_row(key, str(value))

        self.console.print(table)

    async def run(self):
        """运行 CLI"""
        # 打印横幅
        self.print_banner()

        # 初始化系统
        await self.initialize_system()

        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = Prompt.ask("\n[cyan]>[/cyan]")

                if not user_input.strip():
                    continue

                # 处理命令
                command = user_input.strip().lower()

                if command == "exit":
                    self.memory_manager.end_session()
                    self.console.print("再见！", style="cyan")
                    break
                elif command == "help":
                    self.print_help()
                elif command == "status":
                    self.show_status()
                elif command == "health":
                    await self.run_health_check()
                elif command == "clear":
                    self.memory_manager.short_term.clear()
                    self.console.print("✓ 已清空短期记忆", style="green")
                elif command == "history":
                    self.show_history()
                elif command == "preferences":
                    self.show_preferences()
                else:
                    # 处理自然语言查询
                    await self.process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\n使用 'exit' 退出", style="dim")
            except CircuitOpenError:
                self.console.print("\n[bold yellow]⚠ 服务暂时不可用，请稍后再试。[/bold yellow]", style="dim")
            except Exception as e:
                self.console.print(f"\n错误: {e}", style="red")


# Backward compatibility for older tests/imports.
AligoCLI = TravelFlowCLI


def run_health_check_standalone() -> int:
    """
    独立执行健康检查（用于 `python cli.py health`）。
    不进入交互式 CLI，只检测 LLM 是否可达。
    Returns:
        0 成功，1 失败（便于脚本/监控）
    """
    import asyncio
    ok, msg = asyncio.run(check_llm_health(
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
        model_name=LLM_CONFIG["model_name"],
        timeout_sec=RESILIENCE_CONFIG.get("health_check_timeout_sec", 10.0),
    ))
    if ok:
        print("OK")
        return 0
    print(f"FAIL: {msg}")
    return 1


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "health":
        exit(run_health_check_standalone())
    cli = TravelFlowCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
