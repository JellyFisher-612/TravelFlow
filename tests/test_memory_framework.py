from __future__ import annotations

import tempfile
import unittest

from context.long_term_memory import LongTermMemory
from context.memory_manager import MemoryManager
from context.short_term_memory import ShortTermMemory


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


if __name__ == "__main__":
    unittest.main()
