"""Integration tests for the memory subsystem.

Tests MemoryManager + LongTermMemory + ShortTermMemory working together
with real file-system storage (tempdir) and mocked LLM.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class _RecordingSummaryLLM:
    """LLM double that records summary requests and returns deterministic text."""

    def __init__(self, summary: str = "测试总结内容。") -> None:
        self.summary = summary
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(("ainvoke", messages))
        return type("FakeResponse", (), {"content": self.summary})()


def _make_memory_manager(tmpdir: str, llm=None):
    """Create a MemoryManager with real file storage and fake LLM."""
    import sys
    # Patch config to avoid real config dependency
    if "config" not in sys.modules:
        sys.modules["config"] = type(sys)("config")
        sys.modules["config"].MEMORY_CONFIG = {
            "max_chat_history": 50,
            "summary_trigger_count": 10,
        }
        sys.modules["config"].SYSTEM_CONFIG = {}

    from context.memory_manager import MemoryManager
    from context.long_term_memory import LongTermMemory
    from context.short_term_memory import ShortTermMemory

    ltm = LongTermMemory(storage_path=tmpdir)
    stm = ShortTermMemory()
    summary_llm = llm or _RecordingSummaryLLM()
    return MemoryManager(
        user_id="test_user",
        session_id="test_session",
        long_term_memory=ltm,
        short_term_memory=stm,
        summary_llm=summary_llm,
    ), ltm, stm, summary_llm


class IntegrationMemoryTests(unittest.IsolatedAsyncioTestCase):
    """Memory subsystem integration: preferences, chat history, working state."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    async def test_preference_append_then_query(self):
        """添加偏好 → 查询确认 → 再追加 → 查询确认列表增长"""
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir)

        # Save first preference
        await mm.save_preference("hotel_brands", "汉庭", action="append")
        prefs = await ltm.get_preference("hotel_brands")
        self.assertEqual(["汉庭"], prefs)

        # Append second
        await mm.save_preference("hotel_brands", "如家", action="append")
        prefs = await ltm.get_preference("hotel_brands")
        self.assertIn("汉庭", prefs)
        self.assertIn("如家", prefs)
        self.assertEqual(2, len(prefs))

    async def test_chat_history_persistence(self):
        """多轮对话 → 重新创建 MemoryManager → 历史仍在"""
        mm1, ltm1, _, _ = _make_memory_manager(self.tmpdir)

        # Add chat messages
        await mm1.add_message("user", "我想去杭州旅游")
        await mm1.add_message("assistant", "好的，杭州是个好地方")
        await mm1.add_message("user", "帮我规划三天行程")

        # Create new MemoryManager with same storage path
        mm2, ltm2, _, _ = _make_memory_manager(self.tmpdir)
        history = await ltm2.get_chat_history()

        # History should be persisted (at least the user messages)
        roles = [msg["role"] for msg in history]
        self.assertIn("user", roles)

    async def test_working_state_lifecycle(self):
        """设置 pending_plan → 查询确认 → 标记完成 → 查询确认清除"""
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir)

        # Set pending plan in working state
        stm.update_working_state("pending_plan", {"destination": "杭州", "days": 3})
        state = stm.get_working_state()
        self.assertEqual("杭州", state["pending_plan"]["destination"])

        # Clear pending plan
        stm.update_working_state("pending_plan", None)
        state = stm.get_working_state()
        self.assertIsNone(state.get("pending_plan"))

    async def test_session_summary_generation(self):
        """多轮对话后 → 触发总结 → 总结内容包含关键信息"""
        summary_text = "用户计划从北京去杭州出差三天，预算舒适型。"
        llm = _RecordingSummaryLLM(summary=summary_text)
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir, llm=llm)

        # Add enough messages to trigger summary
        for i in range(12):
            await mm.add_message("user", f"消息 {i}")
            await mm.add_message("assistant", f"回复 {i}")

        # Trigger summary
        summary = await mm.summarize_session()

        self.assertIn("杭州", summary)
        self.assertGreater(len(llm.calls), 0)

    async def test_preference_overwrite(self):
        """设置偏好 → 覆盖模式更新 → 查询确认新值"""
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir)

        # Set initial preference
        await mm.save_preference("home_location", "北京", action="replace")
        loc = await ltm.get_preference("home_location")
        self.assertEqual("北京", loc)

        # Overwrite
        await mm.save_preference("home_location", "上海", action="replace")
        loc = await ltm.get_preference("home_location")
        self.assertEqual("上海", loc)

    async def test_cross_session_memory(self):
        """会话A添加偏好 → 会话B查询 → 偏好可见"""
        # Session A
        mm_a, ltm_a, _, _ = _make_memory_manager(self.tmpdir)
        await mm_a.save_preference("airlines", "东航", action="append")

        # Session B (same user, same storage path, different session)
        from context.memory_manager import MemoryManager
        from context.long_term_memory import LongTermMemory
        from context.short_term_memory import ShortTermMemory

        ltm_b = LongTermMemory(storage_path=self.tmpdir)
        stm_b = ShortTermMemory()
        mm_b = MemoryManager(
            user_id="test_user",
            session_id="session_b",
            long_term_memory=ltm_b,
            short_term_memory=stm_b,
            summary_llm=_RecordingSummaryLLM(),
        )

        prefs = await ltm_b.get_preference("airlines")
        self.assertEqual(["东航"], prefs)

    async def test_trip_history_save_and_query(self):
        """保存行程记录 → 查询确认存在"""
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir)

        await mm.save_trip_history({
            "origin": "北京",
            "destination": "杭州",
            "start_date": "2026-06-01",
            "duration_days": 3,
            "purpose": "出差",
        })

        trips = await ltm.get_trip_history()
        self.assertGreater(len(trips), 0)
        self.assertEqual("杭州", trips[-1]["destination"])

    async def test_add_message_writes_both_memories(self):
        """add_message 同时写入短期和长期记忆"""
        mm, ltm, stm, _ = _make_memory_manager(self.tmpdir)

        await mm.add_message("user", "你好", metadata={"test": True})

        # Short-term
        recent = stm.get_recent_context(n_turns=5)
        self.assertGreater(len(recent), 0)
        self.assertEqual("你好", recent[-1]["content"])

        # Long-term (chat history)
        history = await ltm.get_chat_history()
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
