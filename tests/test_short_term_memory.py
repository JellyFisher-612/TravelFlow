from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "agent_langchain" / "context" / "short_term_memory.py"


def load_short_term_memory_class():
    sys.modules["config"] = types.SimpleNamespace(MEMORY_CONFIG={"redis_url": "", "cache_ttl_sec": 3600})
    spec = importlib.util.spec_from_file_location("short_term_memory_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ShortTermMemory


class ShortTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self.ShortTermMemory = load_short_term_memory_class()

    def make_memory(self, max_turns: int = 10):
        memory = self.ShortTermMemory(user_id="user-1", session_id="session-1", max_turns=max_turns)
        memory._redis = None
        return memory

    def test_empty_state_returns_empty_list(self):
        memory = self.make_memory()

        self.assertEqual([], memory.get_recent_context())
        self.assertEqual(0, memory.get_statistics()["total_messages"])

    def test_add_message_stores_messages_correctly(self):
        memory = self.make_memory()

        memory.add_message("user", "帮我规划杭州三日游", {"source": "test"})

        messages = memory.get_recent_context()
        self.assertEqual(1, len(messages))
        self.assertEqual("user", messages[0]["role"])
        self.assertEqual("帮我规划杭州三日游", messages[0]["content"])
        self.assertEqual({"source": "test"}, messages[0]["metadata"])
        self.assertIn("timestamp", messages[0])

    def test_get_recent_context_returns_messages_in_order(self):
        memory = self.make_memory()

        memory.add_message("user", "第一轮")
        memory.add_message("assistant", "第一轮回复")
        memory.add_message("user", "第二轮")

        self.assertEqual(["第一轮", "第一轮回复", "第二轮"], [msg["content"] for msg in memory.get_recent_context()])

    def test_get_recent_context_respects_sliding_window_limit(self):
        memory = self.make_memory(max_turns=2)

        for index in range(6):
            memory.add_message("user" if index % 2 == 0 else "assistant", f"message-{index}")

        self.assertEqual(["message-2", "message-3", "message-4", "message-5"], [msg["content"] for msg in memory.get_recent_context()])
        self.assertEqual(4, memory.get_statistics()["total_messages"])

    def test_get_recent_context_can_return_recent_n_turns(self):
        memory = self.make_memory(max_turns=5)
        for index in range(6):
            memory.add_message("user" if index % 2 == 0 else "assistant", f"message-{index}")

        self.assertEqual(["message-2", "message-3", "message-4", "message-5"], [msg["content"] for msg in memory.get_recent_context(n_turns=2)])

    def test_clear_removes_all_messages(self):
        memory = self.make_memory()
        memory.add_message("user", "出发地上海")
        memory.add_message("assistant", "已记录")

        memory.clear()

        self.assertEqual([], memory.get_recent_context())
        self.assertEqual(0, memory.get_statistics()["total_messages"])

    def test_message_count_tracking_works_correctly(self):
        memory = self.make_memory()

        for index in range(3):
            memory.add_message("user", f"message-{index}")

        stats = memory.get_statistics()
        self.assertEqual(3, stats["total_messages"])
        self.assertEqual(10, stats["max_turns"])
        self.assertIsNotNone(stats["oldest_message_time"])
        self.assertIsNotNone(stats["newest_message_time"])


if __name__ == "__main__":
    unittest.main()
