from __future__ import annotations

import unittest

from agents.main_agent import MainAgent
from context.memory_manager import MemoryManager


class FakeLongTermStore:
    def __init__(self):
        self.preferences = {}
        self.saved_preferences = []
        self.saved_trips = []

    def get_preference(self):
        return dict(self.preferences)

    def save_preference(self, pref_type, value):
        self.preferences[pref_type] = value
        self.saved_preferences.append((pref_type, value))

    def save_trip_history(self, trip_info):
        self.saved_trips.append(trip_info)


class FakeShortTermStore:
    def __init__(self):
        self.pending_plan_cleared = False

    def clear_pending_plan(self):
        self.pending_plan_cleared = True


class PolicyRecordingMemoryManager:
    def __init__(self):
        self.policies = []
        self.results = []

    def apply_agent_results(self, results, policy=None):
        self.results.append(list(results))
        self.policies.append(policy or {})


class FakeIntentRecognition:
    async def run(self, state):
        return {
            **state,
            "intention_data": {
                "intents": [],
                "key_entities": {},
                "agent_schedule": [],
                "direct_answer": "你好，我是 TravelFlow。",
            },
        }


class FakeScheduler:
    def __init__(self):
        self.calls = []

    def create_orchestration_state(self, state):
        self.calls.append(("create", state))
        return {
            "intention_data": state["intention_data"],
            "context": {},
            "batches": [],
            "batch_index": 0,
            "results": [],
            "final_result": {},
            "search_refinement_count": 0,
        }

    async def prepare(self, state):
        return {
            "context": {},
            "batches": [],
            "batch_index": 0,
            "results": [],
            "final_result": {
                "status": "no_agents",
                "direct_answer": "你好，我是 TravelFlow。",
            },
        }

    def has_runnable_batches(self, state):
        return False

    async def aggregate(self, state):
        return {"final_result": state["final_result"]}

    def apply_memory_results(self, results):
        self.calls.append(("memory", results))

    async def run(self, state):
        raise AssertionError("MainAgent should not delegate the full turn to scheduler.run()")


class SupervisableFakeScheduler:
    def __init__(self):
        self.executed_batches = []
        self.memory_results = None

    def create_orchestration_state(self, state):
        return {
            "intention_data": state["intention_data"],
            "context": {},
            "batches": [],
            "batch_index": 0,
            "results": [],
            "final_result": {},
            "search_refinement_count": 0,
        }

    async def prepare(self, state):
        return {
            "context": {},
            "batches": [[{"agent_name": "clarification"}], [{"agent_name": "plan"}]],
            "batch_index": 0,
            "results": [],
        }

    def has_runnable_batches(self, state):
        return state["batch_index"] < len(state["batches"])

    async def run_next_batch(self, state):
        batch = state["batches"][state["batch_index"]]
        agent_name = batch[0]["agent_name"]
        self.executed_batches.append(agent_name)
        return {
            "batch_index": state["batch_index"] + 1,
            "results": state["results"]
            + [
                {
                    "agent_name": agent_name,
                    "priority": state["batch_index"] + 1,
                    "result": {"status": "success", "data": {"agent": agent_name}},
                }
            ],
        }

    def get_blocking_reason(self, state):
        return None

    def handle_blocking_reason(self, reason, state):
        raise AssertionError("No blocking expected")

    def truncate_remaining_batches(self, state):
        raise AssertionError("No truncation expected")

    async def aggregate(self, state):
        return {
            "final_result": {
                "status": "completed",
                "agents_executed": len(state["results"]),
                "results": state["results"],
            }
        }

    def apply_memory_results(self, results):
        self.memory_results = list(results)

    async def run(self, state):
        raise AssertionError("MainAgent should supervise scheduler steps instead of delegating run()")


class BlockingFakeScheduler(SupervisableFakeScheduler):
    def __init__(self):
        super().__init__()
        self.truncated = False

    async def prepare(self, state):
        return {
            "context": {},
            "batches": [[{"agent_name": "clarification"}], [{"agent_name": "plan"}]],
            "batch_index": 0,
            "results": [],
        }

    async def run_next_batch(self, state):
        self.executed_batches.append("clarification")
        return {
            "batch_index": 1,
            "results": [
                {
                    "agent_name": "clarification",
                    "priority": 1,
                    "result": {
                        "status": "success",
                        "data": {
                            "destination": "北京",
                            "start_date": "",
                            "duration_days": 3,
                            "budget_level": "",
                            "pace_preference": "",
                            "missing_info": ["start_date", "budget_level", "pace_preference"],
                        },
                    },
                }
            ],
        }

    def get_blocking_reason(self, state):
        return "clarification_missing_info"

    def handle_blocking_reason(self, reason, state):
        raise AssertionError("MainAgent should own dialogue decisions")

    def truncate_remaining_batches(self, state):
        self.truncated = True
        state["batches"] = state["batches"][: state["batch_index"]]


class ExplodingMemoryManager:
    def add_message(self, *args, **kwargs):
        raise AssertionError("MainAgent should not persist chat messages directly")


class RecordingShortTermStore:
    def __init__(self):
        self.pending_plan = None

    def set_pending_plan(self, query, metadata=None):
        self.pending_plan = (query, metadata or {})


class SupervisorMemoryManager:
    def __init__(self):
        self.short_term = RecordingShortTermStore()
        self.applied = []

    def apply_agent_results(self, results, policy=None):
        self.applied.append((list(results), policy or {}))


class DirectMemoryManager:
    def __init__(self):
        self.short_term = self
        self.long_term = self
        self.applied = []

    def get_recent_context(self, n_turns=3):
        return []

    def get_preference(self):
        return {"nickname": "小李", "pace_preference": "轻松"}

    def get_trip_history(self, limit=20):
        return [{"origin": "上海", "destination": "北京", "start_date": "2026-05-05"}]

    def get_behavior_feedback(self, limit=20):
        return []

    def get_runtime_context(self, recent_turns=3, trip_limit=20, feedback_limit=20):
        return {
            "recent_dialogue": [],
            "user_preferences": self.get_preference(),
            "trip_history": self.get_trip_history(trip_limit),
            "behavior_feedback": self.get_behavior_feedback(feedback_limit),
        }

    def query_memory(self, query, context=None):
        context = context or self.get_runtime_context()
        prefs = context.get("user_preferences") or {}
        return {
            "query": query,
            "answer": f"你的称呼是{prefs.get('nickname')}。",
            "preferences": prefs,
            "trip_history": context.get("trip_history", []),
            "behavior_feedback": context.get("behavior_feedback", []),
        }

    def is_explicit_preference_update_result(self, result):
        return False

    def apply_agent_results(self, results, policy=None):
        self.applied.append((list(results), policy or {}))


class ExplodingScheduler:
    def create_orchestration_state(self, state):
        raise AssertionError("Memory-only turns should be handled by MainAgent directly")

    def set_event_callback(self, callback):
        pass


class MemoryResponsibilityTests(unittest.IsolatedAsyncioTestCase):
    def build_memory_manager(self) -> tuple[MemoryManager, FakeLongTermStore, FakeShortTermStore]:
        manager = MemoryManager.__new__(MemoryManager)
        long_term = FakeLongTermStore()
        short_term = FakeShortTermStore()
        manager.long_term = long_term
        manager.short_term = short_term
        return manager, long_term, short_term

    def test_apply_agent_results_saves_preferences_and_trip_history(self):
        manager, long_term, short_term = self.build_memory_manager()

        manager.apply_agent_results(
            [
                {
                    "agent_name": "clarification",
                    "result": {
                        "data": {
                            "origin": "上海",
                            "destination": "北京",
                            "start_date": "2026-05-05",
                            "end_date": "2026-05-07",
                            "trip_purpose": "亲子游",
                        }
                    },
                },
                {
                    "agent_name": "memory",
                    "result": {
                        "data": {
                            "preferences": [
                                {"type": "hotel_brands", "value": "汉庭", "action": "append"},
                                {"type": "pace_preference", "value": "轻松", "action": "replace"},
                            ]
                        }
                    },
                },
                {
                    "agent_name": "plan",
                    "result": {"data": {"itinerary": {"title": "北京三日游"}}},
                },
            ]
        )

        self.assertEqual([("hotel_brands", ["汉庭"]), ("pace_preference", "轻松")], long_term.saved_preferences)
        self.assertEqual(
            [
                {
                    "origin": "上海",
                    "destination": "北京",
                    "start_date": "2026-05-05",
                    "end_date": "2026-05-07",
                    "purpose": "亲子游",
                }
            ],
            long_term.saved_trips,
        )
        self.assertTrue(short_term.pending_plan_cleared)

    async def test_main_agent_does_not_persist_chat_messages_directly(self):
        scheduler = FakeScheduler()
        agent = MainAgent(
            intent_recognition=FakeIntentRecognition(),
            agent_scheduler=scheduler,
            memory_manager=ExplodingMemoryManager(),
        )

        state = await agent.run({"user_query": "你好"})

        self.assertEqual("no_agents", state["final_result"]["status"])
        self.assertEqual("create", scheduler.calls[0][0])

    async def test_main_agent_handles_memory_query_without_scheduler_subagent(self):
        memory_manager = DirectMemoryManager()

        class MemoryIntentRecognition:
            async def run(self, state):
                return {
                    **state,
                    "intention_data": {
                        "intents": [{"type": "memory"}],
                        "key_entities": {},
                        "rewritten_query": "我是谁",
                        "direct_action": {"type": "memory", "operation": "query"},
                        "agent_schedule": [],
                    },
                }

        agent = MainAgent(
            intent_recognition=MemoryIntentRecognition(),
            agent_scheduler=ExplodingScheduler(),
            memory_manager=memory_manager,
        )

        state = await agent.run({"user_query": "我是谁"})

        self.assertEqual("completed", state["final_result"]["status"])
        self.assertIn("小李", state["final_result"]["results"][0]["data"]["answer"])
        self.assertEqual(1, len(memory_manager.applied))
        self.assertFalse(memory_manager.applied[0][1]["allow_preference_writes"])

    async def test_main_agent_keeps_legacy_memory_schedule_compatibility(self):
        memory_manager = DirectMemoryManager()

        class LegacyMemoryIntentRecognition:
            async def run(self, state):
                return {
                    **state,
                    "intention_data": {
                        "intents": [{"type": "memory"}],
                        "key_entities": {},
                        "rewritten_query": "我是谁",
                        "agent_schedule": [{"agent_name": "memory", "priority": 1}],
                    },
                }

        agent = MainAgent(
            intent_recognition=LegacyMemoryIntentRecognition(),
            agent_scheduler=ExplodingScheduler(),
            memory_manager=memory_manager,
        )

        state = await agent.run({"user_query": "我是谁"})

        self.assertEqual("completed", state["final_result"]["status"])
        self.assertIn("小李", state["final_result"]["results"][0]["data"]["answer"])

    async def test_main_agent_supervises_child_agent_batches(self):
        scheduler = SupervisableFakeScheduler()
        memory_manager = PolicyRecordingMemoryManager()

        class ScheduledIntentRecognition:
            async def run(self, state):
                return {
                    **state,
                    "intention_data": {
                        "intents": [{"type": "plan"}],
                        "key_entities": {},
                        "agent_schedule": [
                            {"agent_name": "clarification", "priority": 1},
                            {"agent_name": "plan", "priority": 2},
                        ],
                    },
                }

        agent = MainAgent(
            intent_recognition=ScheduledIntentRecognition(),
            agent_scheduler=scheduler,
            memory_manager=memory_manager,
        )

        state = await agent.run({"user_query": "帮我规划去北京"})

        self.assertEqual(["clarification", "plan"], scheduler.executed_batches)
        self.assertEqual("completed", state["final_result"]["status"])
        self.assertEqual(2, state["final_result"]["agents_executed"])
        self.assertEqual(1, len(memory_manager.policies))
        self.assertFalse(memory_manager.policies[0]["allow_trip_history_writes"])
        self.assertEqual("turn_completed", state["final_result"]["memory_policy"]["reason"])

    async def test_main_agent_owns_blocking_dialogue_decision(self):
        events = []
        scheduler = BlockingFakeScheduler()
        memory_manager = SupervisorMemoryManager()

        class ScheduledIntentRecognition:
            async def run(self, state):
                return {
                    **state,
                    "intention_data": {
                        "intents": [{"type": "plan"}],
                        "key_entities": {},
                        "rewritten_query": "帮我规划去北京玩三天",
                        "agent_schedule": [
                            {"agent_name": "clarification", "priority": 1},
                            {"agent_name": "plan", "priority": 2},
                        ],
                    },
                }

        agent = MainAgent(
            intent_recognition=ScheduledIntentRecognition(),
            agent_scheduler=scheduler,
            memory_manager=memory_manager,
            event_callback=events.append,
        )

        state = await agent.run({"user_query": "帮我规划去北京玩三天"})

        self.assertEqual(["clarification"], scheduler.executed_batches)
        self.assertTrue(scheduler.truncated)
        self.assertEqual("帮我规划去北京玩三天", memory_manager.short_term.pending_plan[0])
        self.assertEqual("clarification_missing_info", memory_manager.short_term.pending_plan[1]["reason"])
        self.assertEqual(
            ["start_date", "budget_level", "pace_preference"],
            memory_manager.short_term.pending_plan[1]["missing_info"],
        )
        self.assertEqual(
            {"destination": "北京", "duration_days": 3},
            memory_manager.short_term.pending_plan[1]["event_data"],
        )
        self.assertEqual(
            "clarification_missing_info",
            state["final_result"]["supervisor_decision"]["reason"],
        )
        self.assertEqual(1, len(memory_manager.applied))
        self.assertFalse(memory_manager.applied[0][1]["allow_trip_history_writes"])
        self.assertEqual("clarification_missing_info", memory_manager.applied[0][1]["reason"])
        self.assertTrue(
            any(
                isinstance(event, dict) and event.get("stage") == "supervisor_decision"
                for event in events
            )
        )

    async def test_main_agent_allows_trip_history_write_only_after_completed_plan(self):
        scheduler = SupervisableFakeScheduler()
        memory_manager = PolicyRecordingMemoryManager()

        async def completed_plan_batch(state):
            scheduler.executed_batches.append("plan")
            return {
                "batch_index": 1,
                "results": [
                    {
                        "agent_name": "plan",
                        "priority": 1,
                        "result": {
                            "status": "success",
                            "data": {"itinerary": {"title": "北京三日游"}},
                        },
                    }
                ],
            }

        scheduler.prepare = lambda state: _async_value(
            {
                "context": {},
                "batches": [[{"agent_name": "plan"}]],
                "batch_index": 0,
                "results": [],
            }
        )
        scheduler.run_next_batch = completed_plan_batch

        class ScheduledIntentRecognition:
            async def run(self, state):
                return {
                    **state,
                    "intention_data": {
                        "intents": [{"type": "plan"}],
                        "key_entities": {},
                        "agent_schedule": [{"agent_name": "plan", "priority": 1}],
                    },
                }

        agent = MainAgent(
            intent_recognition=ScheduledIntentRecognition(),
            agent_scheduler=scheduler,
            memory_manager=memory_manager,
        )

        state = await agent.run({"user_query": "帮我规划去北京"})

        self.assertEqual("completed", state["final_result"]["status"])
        self.assertEqual(1, len(memory_manager.policies))
        self.assertTrue(memory_manager.policies[0]["allow_trip_history_writes"])
        self.assertTrue(state["final_result"]["memory_policy"]["allow_trip_history_writes"])


async def _async_value(value):
    return value


if __name__ == "__main__":
    unittest.main()
