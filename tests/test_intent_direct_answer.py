from __future__ import annotations

import unittest

from agents.intent_recognition import IntentRecognition

from tests.fakes import ExplodingLLM, FailingTextLLM


class IntentDirectAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def recognize(self, query: str, model=None) -> dict:
        result = await IntentRecognition(model=model).run({"user_query": query})
        return result["intention_data"]

    async def test_direct_answer_queries_do_not_schedule_business_agents_or_call_llm(self):
        cases = [
            ("你好！", "TravelFlow"),
            ("hello", "TravelFlow"),
            ("在吗", "我在"),
            ("你是谁？", "TravelFlow"),
            ("你能做什么？", "旅行规划"),
            ("谢谢", "不客气"),
            ("好的", "继续"),
        ]

        for query, expected_text in cases:
            with self.subTest(query=query):
                model = ExplodingLLM()
                data = await self.recognize(query, model=model)

                self.assertEqual([], data["agent_schedule"])
                self.assertIn("conversation", {item["type"] for item in data["intents"]})
                self.assertIn(expected_text, data["direct_answer"])
                self.assertNotIn("参考来源", data["direct_answer"])
                self.assertEqual([], model.calls)

    async def test_llm_failure_uses_conservative_rule_fallback_without_crashing(self):
        model = FailingTextLLM()

        data = await self.recognize("推荐一个周末去处", model=model)

        self.assertEqual(
            [
                {
                    "agent_name": "search",
                    "priority": 1,
                    "reason": "保守信息查询兜底",
                    "expected_output": "查询结果",
                }
            ],
            data["agent_schedule"],
        )
        self.assertEqual("search", data["intents"][0]["type"])
        self.assertEqual(1, len(model.calls))

    async def test_current_trip_budget_constraints_use_planning_fallback_without_llm(self):
        model = ExplodingLLM()

        data = await self.recognize(
            "本次行程预算选择经济型，住宿每晚300元以内，餐饮和交通尽量节省。",
            model=model,
        )

        self.assertEqual("plan", data["intents"][0]["type"])
        self.assertEqual("经济型", data["key_entities"]["budget_level"])
        self.assertEqual(300, data["key_entities"]["lodging_budget_per_night"])
        self.assertEqual("节省", data["key_entities"]["meal_budget_preference"])
        self.assertEqual("节省", data["key_entities"]["transport_budget_preference"])
        self.assertEqual(
            [
                ("clarification", 1),
                ("search", 2),
                ("plan", 3),
            ],
            [(item["agent_name"], item["priority"]) for item in data["agent_schedule"]],
        )
        self.assertEqual([], model.calls)

    async def test_current_trip_budget_constraints_take_precedence_over_memory_markers(self):
        model = ExplodingLLM()

        data = await self.recognize("我的本次行程预算选经济型，住宿每晚300元以内", model=model)

        self.assertEqual("plan", data["intents"][0]["type"])
        self.assertEqual("current", data["key_entities"]["trip_scope"])
        self.assertEqual("clarification", data["agent_schedule"][0]["agent_name"])
        self.assertEqual([], model.calls)

    async def test_current_trip_budget_range_extracts_min_and_max(self):
        model = ExplodingLLM()

        data = await self.recognize("本次行程预算选择舒适型，住宿每晚300到600元，兼顾体验和性价比。", model=model)

        self.assertEqual("plan", data["intents"][0]["type"])
        self.assertEqual("舒适型", data["key_entities"]["budget_level"])
        self.assertEqual(300, data["key_entities"]["lodging_budget_per_night_min"])
        self.assertEqual(600, data["key_entities"]["lodging_budget_per_night_max"])
        self.assertNotIn("lodging_budget_per_night", data["key_entities"])
        self.assertEqual([], model.calls)

    async def test_memory_queries_use_direct_action_not_business_schedule(self):
        model = ExplodingLLM()

        data = await self.recognize("我的偏好是什么？", model=model)

        self.assertEqual("memory", data["intents"][0]["type"])
        self.assertEqual([], data["agent_schedule"])
        self.assertEqual({"type": "memory", "operation": "query", "reason": "读取或更新用户长期偏好、历史行程和行为反馈"}, data["direct_action"])
        self.assertEqual([], model.calls)
