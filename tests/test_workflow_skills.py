from __future__ import annotations

import unittest

from agents.agent_scheduler import AgentScheduler
from agents.main_agent import MainAgent
from agents.workflow_skills.router import SkillRouter

from tests.fakes import RecordingAgent


class ExplodingIntentRecognition:
    async def run(self, state):
        raise AssertionError("Workflow skill match should bypass LLM intent recognition")


class WorkflowSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = SkillRouter()

    def test_travel_planning_skill_matches_common_trip_request(self):
        match = self.router.match({"user_query": "我要去北京旅游"})

        self.assertIsNotNone(match)
        self.assertEqual("travel_planning", match.skill_name)
        self.assertEqual(
            [("clarification", 1), ("search", 2), ("plan", 3)],
            [(item["agent_name"], item["priority"]) for item in match.intention_data["agent_schedule"]],
        )
        self.assertEqual("scheduled_agents", match.workflow_plan["execution_mode"])
        self.assertEqual(["clarification", "search", "plan"], [step["agent_name"] for step in match.workflow_plan["steps"]])
        self.assertEqual(["clarification", "search", "plan"], [agent["agent_name"] for agent in match.workflow_plan["child_agents"]])
        self.assertEqual(["AmapService", "TrainService", "DDGS"], [tool["name"] for tool in match.workflow_plan["tools"]])
        self.assertEqual(["MemoryManager", "ShortTermMemory"], [item["name"] for item in match.workflow_plan["memory_dependencies"]])
        self.assertEqual("ask_user", match.workflow_plan["blocking_policy"]["missing_required_fields"])
        self.assertEqual("北京", match.intention_data["key_entities"]["destination"])

    def test_weather_skill_matches_weather_query(self):
        match = self.router.match({"user_query": "杭州明天天气怎么样"})

        self.assertIsNotNone(match)
        self.assertEqual("weather_query", match.skill_name)
        self.assertEqual([("search", 1)], [(item["agent_name"], item["priority"]) for item in match.intention_data["agent_schedule"]])
        self.assertEqual("weather", match.workflow_plan["steps"][0]["search_profile"])
        self.assertEqual(["search"], [agent["agent_name"] for agent in match.workflow_plan["child_agents"]])
        self.assertEqual(["AmapService"], [tool["name"] for tool in match.workflow_plan["tools"]])
        self.assertEqual(["maps_weather"], match.workflow_plan["tools"][0]["operations"])

    def test_memory_skill_uses_direct_action(self):
        match = self.router.match({"user_query": "我的偏好是什么"})

        self.assertIsNotNone(match)
        self.assertEqual("memory_profile", match.skill_name)
        self.assertEqual([], match.intention_data["agent_schedule"])
        self.assertEqual("main_agent_direct_action", match.workflow_plan["execution_mode"])
        self.assertEqual([], match.workflow_plan["child_agents"])
        self.assertEqual(["MemoryManager", "PreferenceAgent"], [tool["name"] for tool in match.workflow_plan["tools"]])
        self.assertEqual(["LongTermMemory", "ShortTermMemory"], [item["name"] for item in match.workflow_plan["memory_dependencies"]])
        self.assertEqual({"type": "memory", "operation": "query", "reason": "读取或更新用户长期偏好、历史行程和行为反馈"}, match.intention_data["direct_action"])

    def test_information_skill_matches_generic_travel_info_query(self):
        match = self.router.match({"user_query": "北京有什么好玩的"})

        self.assertIsNotNone(match)
        self.assertEqual("information_query", match.skill_name)
        self.assertEqual("search", match.intention_data["agent_schedule"][0]["agent_name"])
        self.assertEqual("travel_info", match.workflow_plan["steps"][0]["search_profile"])
        self.assertEqual(["AmapService", "DDGS"], [tool["name"] for tool in match.workflow_plan["tools"]])

    def test_unmatched_query_falls_back_to_intent_recognition(self):
        match = self.router.match({"user_query": "随便聊聊哲学"})

        self.assertIsNone(match)

    async def test_main_agent_runs_matched_workflow_without_intent_recognition(self):
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
        plan = RecordingAgent(
            "plan",
            {
                "planning_complete": True,
                "itinerary": {"title": "北京三日游", "daily_plans": [{"day": 1, "activities": ["故宫博物院"]}]},
            },
        )
        scheduler = AgentScheduler(agent_registry={"clarification": clarification, "search": search, "plan": plan})
        main_agent = MainAgent(intent_recognition=ExplodingIntentRecognition(), agent_scheduler=scheduler)

        state = await main_agent.run({"user_query": "我要去北京旅游"})

        self.assertEqual("travel_planning", state["workflow_skill"]["name"])
        self.assertEqual("scheduled_agents", state["workflow_plan"]["execution_mode"])
        self.assertEqual("completed", state["final_result"]["status"])
        self.assertEqual(["clarification", "search", "plan"], [item["agent_name"] for item in state["results"]])


if __name__ == "__main__":
    unittest.main()
