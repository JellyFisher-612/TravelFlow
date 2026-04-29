from __future__ import annotations

import unittest

from agents.agent_scheduler import AgentScheduler
from agents.intent_recognition import IntentRecognition
from agents.main_agent import MainAgent

from tests.fakes import ExplodingLLM, RecordingAgent


class PlanningDispatchTests(unittest.IsolatedAsyncioTestCase):
    def build_main_agent(self, registry: dict, events: list | None = None) -> MainAgent:
        callback = events.append if events is not None else None
        scheduler = AgentScheduler(agent_registry=registry, event_callback=callback)
        return MainAgent(
            intent_recognition=IntentRecognition(model=ExplodingLLM()),
            agent_scheduler=scheduler,
            event_callback=callback,
        )

    async def test_complete_planning_request_runs_layered_agents_and_passes_context(self):
        clarification = RecordingAgent(
            "clarification",
            {
                "origin": "上海",
                "destination": "北京",
                "start_date": "2026-05-05",
                "duration_days": 3,
                "budget_level": "舒适型",
                "pace_preference": "轻松",
                "missing_info": [],
            },
        )
        search = RecordingAgent(
            "search",
            {
                "query_success": True,
                "results": {"pois": [{"name": "故宫博物院"}]},
                "quality": {"hard_constraints_verified": False},
            },
        )
        memory = RecordingAgent(
            "memory",
            {"preferences": {"budget_level": "舒适型", "pace_preference": "轻松"}},
        )
        plan = RecordingAgent(
            "plan",
            {
                "planning_complete": True,
                "itinerary": {
                    "title": "北京三日游",
                    "daily_plans": [{"day": 1, "activities": ["故宫博物院"]}],
                },
            },
        )
        main_agent = self.build_main_agent(
            {
                "clarification": clarification,
                "search": search,
                "memory": memory,
                "plan": plan,
            }
        )

        state = await main_agent.run({"user_query": "帮我规划下周从上海去北京玩三天"})

        self.assertEqual(
            [
                ("clarification", 1),
                ("search", 2),
                ("memory", 2),
                ("plan", 3),
            ],
            [(item["agent_name"], item["priority"]) for item in state["intention_data"]["agent_schedule"]],
        )
        self.assertEqual(
            [
                ("clarification", 1),
                ("search", 2),
                ("memory", 2),
                ("plan", 3),
            ],
            [(item["agent_name"], item["priority"]) for item in state["results"]],
        )
        self.assertEqual("completed", state["final_result"]["status"])
        self.assertEqual(1, len(clarification.calls))
        self.assertEqual(1, len(search.calls))
        self.assertEqual(1, len(memory.calls))
        self.assertEqual(1, len(plan.calls))

        plan_context = plan.calls[0]["context"]
        self.assertEqual("北京", plan_context["event_data"]["destination"])
        self.assertTrue(plan_context["search_data"]["query_success"])
        self.assertEqual("轻松", plan_context["user_preferences"]["pace_preference"])

    async def test_missing_critical_fields_block_search_memory_plan_and_request_inputs(self):
        events = []
        clarification = RecordingAgent(
            "clarification",
            {
                "origin": "上海",
                "destination": "北京",
                "start_date": "2026-05-05",
                "duration_days": 3,
                "budget_level": "",
                "pace_preference": "",
                "missing_info": ["budget_level", "pace_preference"],
            },
        )
        search = RecordingAgent("search", {"query_success": True})
        memory = RecordingAgent("memory", {"preferences": {}})
        plan = RecordingAgent("plan", {"planning_complete": True})
        main_agent = self.build_main_agent(
            {
                "clarification": clarification,
                "search": search,
                "memory": memory,
                "plan": plan,
            },
            events=events,
        )

        state = await main_agent.run({"user_query": "帮我规划下周从上海去北京玩三天"})

        self.assertEqual(1, len(clarification.calls))
        self.assertEqual([], search.calls)
        self.assertEqual([], memory.calls)
        self.assertEqual([], plan.calls)
        self.assertEqual(["clarification"], [item["agent_name"] for item in state["results"]])
        self.assertEqual("completed", state["final_result"]["status"])
        self.assertEqual(1, state["final_result"]["agents_executed"])
        self.assertTrue(
            any(
                isinstance(event, dict) and event.get("stage") == "supervisor_decision"
                for event in events
            )
        )
        self.assertEqual(
            "clarification_missing_info",
            state["final_result"]["supervisor_decision"]["reason"],
        )
        self.assertEqual(
            ["budget_level", "pace_preference"],
            [item["field"] for item in state["final_result"]["input_requests"]],
        )


if __name__ == "__main__":
    unittest.main()
