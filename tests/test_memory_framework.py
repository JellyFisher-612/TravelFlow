from __future__ import annotations

import tempfile
import unittest

from config import MEMORY_CONFIG
from context.long_term_memory import LongTermMemory
from context.memory_manager import MemoryManager
from context.short_term_memory import ShortTermMemory


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _RecordingSummaryLLM:
    def __init__(self, content: str = "本会话确认了杭州三日游，并强调轻松节奏。"):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return _FakeLLMResponse(self.content)


class MemoryFrameworkTests(unittest.TestCase):
    def test_preference_records_preserve_simple_reads_and_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(user_id="profile_user", storage_path=tmpdir)

            memory.save_preference(
                "pace_preference",
                "轻松",
                metadata={
                    "source": {"session_id": "s1", "quote": "以后都安排轻松一点"},
                    "confidence": 0.9,
                },
            )

            self.assertEqual("轻松", memory.get_preference("pace_preference"))
            record = memory.get_preference_records()["pace_preference"]
            self.assertEqual("轻松", record["value"])
            self.assertEqual("active", record["status"])
            self.assertEqual(1, record["version"])
            self.assertEqual("s1", record["source"]["session_id"])
            self.assertEqual(0.9, record["confidence"])

            memory.save_preference(
                "pace_preference",
                "均衡",
                metadata={"source": {"session_id": "s2", "quote": "以后改成均衡节奏"}},
            )

            updated = memory.get_preference_records()["pace_preference"]
            self.assertEqual("均衡", memory.get_preference("pace_preference"))
            self.assertEqual(2, updated["version"])
            self.assertEqual("轻松", updated["history"][0]["value"])

    def test_memory_manager_commits_profile_records_with_policy_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(
                user_id="manager_user",
                session_id="session-1",
                storage_path=tmpdir,
            )

            manager.apply_agent_results(
                [
                    {
                        "agent_name": "memory",
                        "result": {
                            "data": {
                                "has_preferences": True,
                                "summary": "用户希望以后按经济型预算规划。",
                                "preferences": [
                                    {
                                        "type": "budget_level",
                                        "value": "经济型",
                                        "action": "replace",
                                        "confidence": 0.95,
                                    }
                                ],
                            }
                        },
                    }
                ],
                policy={
                    "owner": "MainAgent",
                    "reason": "turn_completed",
                    "allow_preference_writes": True,
                    "allow_trip_history_writes": False,
                },
            )

            records = manager.long_term.get_preference_records()
            record = records["budget_level"]
            self.assertEqual("经济型", manager.long_term.get_preference("budget_level"))
            self.assertEqual("MainAgent", record["source"]["owner"])
            self.assertEqual("session-1", record["source"]["session_id"])
            self.assertEqual(0.95, record["confidence"])

            runtime_context = manager.get_runtime_context()
            self.assertEqual("经济型", runtime_context["user_preferences"]["budget_level"])
            self.assertEqual("经济型", runtime_context["user_profile_records"]["budget_level"]["value"])

    def test_working_state_generalizes_pending_plan(self):
        memory = ShortTermMemory(user_id="working_user", session_id="thread-1")
        memory._redis = None

        memory.update_working_state({"destination": "北京", "missing_info": ["start_date"]})
        state = memory.get_working_state()["state"]
        self.assertEqual("北京", state["destination"])
        self.assertEqual(["start_date"], state["missing_info"])

        memory.set_pending_plan("我要去北京旅游", {"reason": "clarification_missing_info"})
        working = memory.get_working_state()["state"]
        self.assertEqual("北京", working["destination"])
        self.assertEqual("我要去北京旅游", working["pending_plan"]["query"])

        memory.clear_pending_plan()
        self.assertNotIn("pending_plan", memory.get_working_state()["state"])

    def test_json_fallback_prunes_chat_history_by_configured_limit(self):
        original = dict(MEMORY_CONFIG)
        try:
            MEMORY_CONFIG["json_max_chat_messages"] = 3
            MEMORY_CONFIG["json_chat_ttl_days"] = 0
            MEMORY_CONFIG["json_max_bytes"] = 0

            with tempfile.TemporaryDirectory() as tmpdir:
                memory = LongTermMemory(user_id="prune_user", storage_path=tmpdir)
                for idx in range(5):
                    memory.add_chat_message("user", f"message-{idx}", session_id="s1")

                history = memory.get_chat_history(limit=None)
                self.assertEqual(["message-2", "message-3", "message-4"], [msg["content"] for msg in history])
                self.assertEqual(3, memory.get_statistics()["total_messages"])
        finally:
            MEMORY_CONFIG.clear()
            MEMORY_CONFIG.update(original)


class MemorySummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_summary_is_persisted_without_runtime_resummarization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _RecordingSummaryLLM()
            manager = MemoryManager(
                user_id="summary_user",
                session_id="session-1",
                storage_path=tmpdir,
                llm_model=llm,
            )

            for idx in range(4):
                manager.add_message("user", f"第{idx}轮：我要去杭州")
                manager.add_message("assistant", "已记录杭州行程。", metadata={"display": "已记录杭州行程。"})

            summary = await manager.summarize_current_session_async(min_messages=2)
            self.assertEqual("本会话确认了杭州三日游，并强调轻松节奏。", summary)
            self.assertEqual(1, len(llm.calls))

            meta = manager.long_term.get_session_summary("session-1")
            self.assertEqual(summary, meta["summary"])
            self.assertEqual(8, meta["summary_message_count"])

            cached = await manager.get_long_term_summary_async()
            self.assertEqual("", cached)
            self.assertEqual(1, len(llm.calls))

    async def test_long_term_summary_reads_other_persisted_session_summaries_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _RecordingSummaryLLM()
            manager = MemoryManager(
                user_id="summary_user",
                session_id="session-1",
                storage_path=tmpdir,
                llm_model=llm,
            )
            manager.long_term.update_session_summary("old-session", "旧会话摘要", 12)

            summary = await manager.get_long_term_summary_async()
            self.assertEqual("- 旧会话摘要", summary)
            self.assertEqual([], llm.calls)

    async def test_memory_query_uses_chat_logs_only_for_explicit_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(
                user_id="summary_user",
                session_id="session-1",
                storage_path=tmpdir,
            )
            manager.long_term.update_session_summary("old-session", "旧对话里用户比较过杭州和苏州。", 10)

            normal = manager.query_memory("我过去去过哪里", {"trip_history": [], "user_preferences": {}})
            self.assertFalse(normal["used_chat_logs"])

            explicit = manager.query_memory("我们之前聊过什么旅游方案", {"trip_history": [], "user_preferences": {}})
            self.assertTrue(explicit["used_chat_logs"])
            self.assertIn("杭州和苏州", explicit["answer"])


if __name__ == "__main__":
    unittest.main()
