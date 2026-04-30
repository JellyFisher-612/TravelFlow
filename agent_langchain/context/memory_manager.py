"""
记忆管理器 (Memory Manager)
统一管理两层记忆，提供简单的API
"""
from typing import Dict, Any, List, Optional
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from utils.langchain_runtime import ainvoke_text
import logging
import json

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器：统一管理两层记忆
    - 短期记忆：最近对话（会话级）
    - 长期记忆：用户偏好和历史（跨会话）
    """

    def __init__(self, user_id: str, session_id: str, storage_path: str = "data/memory", llm_model=None):
        """
        初始化记忆管理器

        Args:
            user_id: 用户ID
            session_id: 会话ID
            storage_path: 长期记忆存储路径
            llm_model: LLM模型实例（用于总结长期记忆）
        """
        self.user_id = user_id
        self.session_id = session_id
        self.llm_model = llm_model

        # 初始化两层记忆
        self.short_term = ShortTermMemory(user_id=user_id, session_id=session_id, max_turns=10)
        self.long_term = LongTermMemory(user_id, storage_path)

        logger.info(f"Memory manager initialized for user {user_id}, session {session_id}")

    # ========== 短期记忆操作 ==========

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到短期记忆和长期记忆

        Args:
            role: 角色 (user/assistant)
            content: 消息内容
            metadata: 元数据
        """
        # 添加到短期记忆（当前会话）
        self.short_term.add_message(role, content, metadata)

        # 同时添加到长期记忆（跨会话持久化）
        self.long_term.add_chat_message(role, content, self.session_id, metadata)

    def apply_agent_results(self, results: List[Dict[str, Any]], policy: Dict[str, Any] = None):
        """
        Apply structured memory updates approved by the supervisor.

        Memory-capable agents may extract preferences or detect conflicts, but
        the caller supplies policy so long-term writes remain under global
        conversation control.
        """
        policy = policy or {}
        allow_preference_writes = bool(policy.get("allow_preference_writes", True))
        allow_trip_history_writes = bool(policy.get("allow_trip_history_writes", True))
        clear_pending_plan_on_trip = bool(policy.get("clear_pending_plan_on_trip", True))

        if not results:
            return

        for result in results:
            if not isinstance(result, dict):
                continue

            agent_name = result.get("agent_name")
            data = self._extract_result_data(result)

            if allow_preference_writes and agent_name in {"memory", "preference"} and isinstance(data, dict):
                self._apply_preferences(data.get("preferences", {}), policy=policy, result_data=data)

            if allow_trip_history_writes and agent_name in {"plan", "itinerary_planning"} and isinstance(data, dict):
                self._apply_trip_history(
                    data,
                    results,
                    clear_pending_plan=clear_pending_plan_on_trip,
                )

    def _extract_result_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        wrapped = result.get("result", {})
        if isinstance(wrapped, dict) and isinstance(wrapped.get("data"), dict):
            return wrapped["data"]
        if isinstance(result.get("data"), dict):
            return result["data"]
        return {}

    def _apply_preferences(
        self,
        preferences_data: Any,
        policy: Dict[str, Any] = None,
        result_data: Dict[str, Any] = None,
    ):
        policy = policy or {}
        result_data = result_data or {}
        if isinstance(preferences_data, list):
            for pref_item in preferences_data:
                if not isinstance(pref_item, dict):
                    continue
                pref_type = pref_item.get("type")
                pref_value = pref_item.get("value")
                pref_action = pref_item.get("action", "replace")
                if not pref_type or not pref_value:
                    continue
                self._save_preference_item(
                    pref_type,
                    pref_value,
                    pref_action,
                    metadata=self._build_preference_metadata(pref_item, policy, result_data),
                )
            return

        if isinstance(preferences_data, dict):
            for pref_type, value in preferences_data.items():
                if value and pref_type not in {"has_preferences", "error"}:
                    self._save_preference_record(
                        pref_type,
                        value,
                        metadata=self._build_preference_metadata({}, policy, result_data),
                    )

    def _build_preference_metadata(
        self,
        pref_item: Dict[str, Any],
        policy: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        source = pref_item.get("source") if isinstance(pref_item.get("source"), dict) else {}
        if not source:
            source = {
                "session_id": getattr(self, "session_id", None),
                "owner": policy.get("owner"),
                "reason": policy.get("reason"),
            }
            if pref_item.get("quote"):
                source["quote"] = pref_item.get("quote")
            elif result_data.get("summary"):
                source["quote"] = result_data.get("summary")
            source = {key: value for key, value in source.items() if value is not None}

        return {
            "source": source,
            "confidence": pref_item.get("confidence", result_data.get("confidence", 1.0)),
            "scope": pref_item.get("scope", "global"),
            "status": pref_item.get("status", "active"),
            "owner": policy.get("owner"),
            "reason": policy.get("reason"),
            "session_id": getattr(self, "session_id", None),
        }

    def _save_preference_item(
        self,
        pref_type: str,
        pref_value: Any,
        pref_action: str,
        metadata: Dict[str, Any] = None,
    ):
        if pref_action == "append":
            current_prefs = self.long_term.get_preference()
            existing_value = current_prefs.get(pref_type) if isinstance(current_prefs, dict) else None
            if isinstance(existing_value, list):
                if pref_value not in existing_value:
                    existing_value.append(pref_value)
                self._save_preference_record(pref_type, existing_value, metadata=metadata, action=pref_action)
                return

            new_list = [existing_value, pref_value] if existing_value else [pref_value]
            self._save_preference_record(pref_type, new_list, metadata=metadata, action=pref_action)
            return

        self._save_preference_record(pref_type, pref_value, metadata=metadata, action=pref_action)

    def _save_preference_record(
        self,
        pref_type: str,
        value: Any,
        metadata: Dict[str, Any] = None,
        action: str = "replace",
    ):
        try:
            self.long_term.save_preference(pref_type, value, metadata=metadata, action=action)
        except TypeError:
            # Older test doubles or integrations may still expose the legacy
            # two-argument API. Keep the manager backward-compatible.
            self.long_term.save_preference(pref_type, value)

    def _apply_trip_history(
        self,
        plan_data: Dict[str, Any],
        results: List[Dict[str, Any]],
        clear_pending_plan: bool = True,
    ):
        if not plan_data.get("itinerary"):
            return

        event_data = {}
        for result in results:
            if result.get("agent_name") in {"clarification", "event_collection"}:
                candidate = self._extract_result_data(result)
                if isinstance(candidate, dict):
                    event_data = candidate
                break

        destination = event_data.get("destination")
        if not destination:
            return

        self.long_term.save_trip_history(
            {
                "origin": event_data.get("origin"),
                "destination": destination,
                "start_date": event_data.get("start_date"),
                "end_date": event_data.get("end_date"),
                "purpose": event_data.get("trip_purpose", "旅游"),
            }
        )
        if clear_pending_plan:
            self.short_term.clear_pending_plan()

    # ========== 长期记忆操作 ==========
    # 注意：大部分方法直接使用 self.short_term 和 self.long_term 即可，无需封装

    # ========== 综合查询 ==========

    def get_full_context(self) -> Dict[str, Any]:
        """
        获取完整上下文（两层记忆）

        Returns:
            完整上下文字典
        """
        return {
            "short_term": {
                "recent_dialogue": self.short_term.get_recent_context(5),
                "context_string": self.short_term.get_context_string(5),
                "statistics": self.short_term.get_statistics()
            },
            "long_term": {
                "preferences": self.long_term.get_preference(),
                "chat_history": self.long_term.get_chat_history(10),
                "trip_history": self.long_term.get_trip_history(5),
                "frequent_destinations": self.long_term.get_frequent_destinations(3),
                "statistics": self.long_term.get_statistics()
            }
        }

    def get_runtime_context(
        self,
        recent_turns: int = 3,
        trip_limit: int = 20,
        feedback_limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Build a bounded memory snapshot for MainAgent-driven orchestration.

        MainAgent decides when memory is needed; MemoryManager owns what memory
        is exposed and how it is shaped for downstream agents.
        """
        context: Dict[str, Any] = {}

        try:
            context["recent_dialogue"] = self.short_term.get_recent_context(recent_turns)
        except Exception:
            logger.debug("Failed to read short-term memory context", exc_info=True)

        try:
            if hasattr(self.short_term, "get_working_state"):
                context["working_state"] = self.short_term.get_working_state().get("state", {})
        except Exception:
            logger.debug("Failed to read working memory state", exc_info=True)

        try:
            context["user_preferences"] = self.long_term.get_preference()
            if hasattr(self.long_term, "get_preference_records"):
                context["user_profile_records"] = self.long_term.get_preference_records()
        except Exception:
            logger.debug("Failed to read user preferences", exc_info=True)

        try:
            context["trip_history"] = self.long_term.get_trip_history(limit=trip_limit)
        except Exception:
            logger.debug("Failed to read trip history", exc_info=True)

        try:
            context["behavior_feedback"] = self.long_term.get_behavior_feedback(limit=feedback_limit)
        except Exception:
            logger.debug("Failed to read behavior feedback", exc_info=True)

        return {key: value for key, value in context.items() if value not in (None, "")}

    def update_working_state(self, patch: Dict[str, Any]):
        if hasattr(self.short_term, "update_working_state"):
            self.short_term.update_working_state(patch)

    def get_working_state(self) -> Dict[str, Any]:
        if hasattr(self.short_term, "get_working_state"):
            payload = self.short_term.get_working_state()
            return payload.get("state", {}) if isinstance(payload.get("state"), dict) else {}
        return {}

    def clear_working_state(self):
        if hasattr(self.short_term, "clear_working_state"):
            self.short_term.clear_working_state()

    def query_memory(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Answer an explicit personal-memory query without giving MainAgent
        direct ownership of memory state or formatting rules.
        """
        context = context or self.get_runtime_context()
        prefs = context.get("user_preferences") or {}
        trips = context.get("trip_history") or []
        feedback = context.get("behavior_feedback") or []

        if any(word in (query or "") for word in ("我是谁", "知道我是谁", "认识我")):
            known = []
            if isinstance(prefs, dict) and (prefs.get("name") or prefs.get("nickname")):
                known.append(f"你的称呼是{prefs.get('name') or prefs.get('nickname')}")
            else:
                known.append("我目前不知道你的姓名或真实身份")
            if isinstance(prefs, dict) and prefs.get("home_location"):
                known.append(f"你常从{prefs['home_location']}出发")
            destinations = self._recent_destinations(trips)
            if destinations:
                known.append(f"你近期关注或规划过这些目的地：{'、'.join(destinations[:5])}")
            return {
                "query": query,
                "answer": "；".join(known) + "。如果你愿意，可以告诉我你的名字或称呼，我会记住。",
                "preferences": prefs,
                "trip_history": trips[:5] if isinstance(trips, list) else [],
                "behavior_feedback": feedback[:5] if isinstance(feedback, list) else [],
            }

        answer_parts = []
        if isinstance(prefs, dict) and any(prefs.values()):
            pref_lines = [f"{key}: {value}" for key, value in prefs.items() if value]
            if pref_lines:
                answer_parts.append("已记录偏好：" + "；".join(pref_lines))
        if isinstance(trips, list) and trips:
            trip_lines = []
            for trip in trips[:5]:
                if not isinstance(trip, dict):
                    continue
                origin = trip.get("origin") or "未知出发地"
                destination = trip.get("destination") or "未知目的地"
                start = trip.get("start_date") or ""
                trip_lines.append(f"{origin}到{destination}{f'（{start}）' if start else ''}")
            if trip_lines:
                answer_parts.append("近期行程：" + "；".join(trip_lines))
        if isinstance(feedback, list) and feedback:
            feedback_lines = [
                str(item.get("feedback", item))
                for item in feedback[:5]
                if isinstance(item, dict)
            ]
            if feedback_lines:
                answer_parts.append("行为反馈：" + "；".join(feedback_lines))

        return {
            "query": query,
            "answer": "；".join(answer_parts) if answer_parts else "目前没有找到相关的长期记忆。",
            "preferences": prefs,
            "trip_history": trips[:5] if isinstance(trips, list) else [],
            "behavior_feedback": feedback[:5] if isinstance(feedback, list) else [],
        }

    def is_explicit_preference_update_result(self, result: Dict[str, Any]) -> bool:
        """
        Decide whether an agent result is a preference-update candidate.

        Query responses may include a preferences snapshot; those must not be
        treated as writes.
        """
        if not isinstance(result, dict) or result.get("agent_name") not in {"memory", "preference"}:
            return False
        data = self._extract_result_data(result)
        if not isinstance(data, dict):
            return False
        preferences = data.get("preferences")
        if not preferences:
            return False
        if data.get("has_preferences") is True:
            return True
        return isinstance(preferences, list)

    def _recent_destinations(self, trips: Any) -> List[str]:
        if not isinstance(trips, list):
            return []
        destinations = []
        for trip in trips[-5:]:
            if not isinstance(trip, dict):
                continue
            dest = trip.get("destination")
            if dest and dest not in destinations:
                destinations.append(dest)
        return destinations

    def get_context_for_agent(self, long_term_summary: str = None) -> str:
        """
        获取用于Agent的上下文字符串

        Args:
            long_term_summary: 长期记忆总结（可选，需提前调用 get_long_term_summary_async）

        Returns:
            格式化的上下文字符串
        """
        lines = []

        # 长期记忆总结（历史会话）
        if long_term_summary:
            lines.append("【历史会话总结】")
            lines.append(long_term_summary)
            lines.append("")

        # 用户偏好
        prefs = self.long_term.get_preference()
        has_prefs = any(v for v in prefs.values() if v)
        if has_prefs:
            lines.append("【用户偏好】")
            for key, value in prefs.items():
                if value:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        # 短期记忆（当前会话）
        context_str = self.short_term.get_context_string(3)
        if context_str != "无历史对话":
            lines.append("【当前会话对话】")
            lines.append(context_str)
            lines.append("")

        return "\n".join(lines) if lines else "无上下文信息"

    # ========== 会话管理 ==========

    def end_session(self):
        """结束会话"""
        self.short_term.clear()
        logger.info(f"Session ended: {self.session_id}")

    async def get_long_term_summary_async(self, max_messages: int = 50) -> str:
        """
        使用LLM总结长期聊天历史（异步版本）

        Args:
            max_messages: 最多总结的消息数量

        Returns:
            总结后的文本
        """
        if not self.llm_model:
            return ""

        # 获取长期聊天历史（排除当前会话）
        all_history = self.long_term.get_chat_history(limit=max_messages)
        history_from_other_sessions = [
            msg for msg in all_history
            if msg.get("session_id") != self.session_id
        ]

        # 获取行程历史
        trip_history = self.long_term.get_trip_history(limit=20)

        # 如果既没有聊天记录也没有行程记录，直接返回
        if not history_from_other_sessions and not trip_history:
            return ""

        # 构建聊天记录文本
        history_text = []
        for msg in history_from_other_sessions[-max_messages:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            history_text.append(f"[{timestamp}] {role}: {content}")

        history_str = "\n".join(history_text) if history_text else "（无聊天记录）"

        # 构建行程历史文本
        trip_text = []
        for trip in trip_history:
            origin = trip.get("origin", "未知")
            destination = trip.get("destination", "未知")
            start_date = trip.get("start_date", "")
            end_date = trip.get("end_date", "")
            purpose = trip.get("purpose", "旅游")
            timestamp = trip.get("timestamp", "")

            if start_date and end_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date} 至 {end_date}) - {purpose}")
            elif start_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date}) - {purpose}")
            else:
                trip_text.append(f"[{timestamp}] {origin} → {destination} - {purpose}")

        trip_str = "\n".join(trip_text) if trip_text else "（无行程记录）"

        # 使用LLM总结
        summarization_prompt = f"""请总结以下历史信息中的关键内容，包括：
1. 用户的旅行偏好和习惯
2. 用户询问过的重要问题
3. 用户的出行历史和目的地
4. 其他重要的上下文信息

【历史聊天记录】
{history_str}

【历史行程记录】
{trip_str}

请用简洁的语言总结（不超过200字）："""

        try:
            summary = await ainvoke_text(self.llm_model, [{"role": "user", "content": summarization_prompt}])

            logger.info(f"Generated long-term memory summary ({len(summary)} chars)")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate long-term summary: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def get_long_term_summary(self, max_messages: int = 50) -> str:
        """
        使用LLM总结长期聊天历史（同步版本）

        Args:
            max_messages: 最多总结的消息数量

        Returns:
            总结后的文本
        """
        import asyncio

        # 检查是否在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 已经在事件循环中，不能使用 asyncio.run
            logger.warning("get_long_term_summary called from async context, please use get_long_term_summary_async instead")
            return ""
        except RuntimeError:
            # 没有运行的事件循环，可以使用 asyncio.run
            return asyncio.run(self.get_long_term_summary_async(max_messages))
