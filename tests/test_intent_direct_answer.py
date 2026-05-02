from __future__ import annotations

import unittest

from agents.intent_recognition import IntentRecognition

from tests.fakes import ExplodingLLM, FailingTextLLM


class JSONTextLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return type("FakeResponse", (), {"content": self.content})()


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
            ("你是做什么的", "旅行规划"),
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

    async def test_llm_direct_answer_is_not_filled_with_default_search(self):
        model = JSONTextLLM(
            """
            {
              "reasoning": "用户询问助手功能，应直接回复。",
              "intents": [
                {
                  "type": "system_function_inquiry",
                  "confidence": 1.0,
                  "description": "系统功能咨询",
                  "reason": "用户询问助手的功能和角色。"
                }
              ],
              "key_entities": {},
              "rewritten_query": "用户询问 TravelFlow 旅游出行助手的功能和角色。",
              "direct_answer": "我是 TravelFlow 旅游出行助手，可以帮你规划行程。",
              "agent_schedule": []
            }
            """
        )

        data = await self.recognize("请说明当前系统的职责范围", model=model)

        self.assertEqual([], data["agent_schedule"])
        self.assertIn("TravelFlow", data["direct_answer"])
        self.assertEqual(1, len(model.calls))

    async def test_minimal_destination_request_does_not_inherit_trip_history_fields(self):
        model = JSONTextLLM(
            """
            {
              "reasoning": "历史中有南京行程，因此补全为历史行程。",
              "intents": [{"type": "plan", "confidence": 0.9, "description": "规划", "reason": "用户想去南京"}],
              "key_entities": {
                "origin": "为我个",
                "destination": "南京",
                "start_date": "2026-05-03",
                "duration_days": 4,
                "budget_level": "经济型",
                "pace_preference": "轻松"
              },
              "rewritten_query": "用户想从为我个出发，2026-05-03去南京玩4天。",
              "agent_schedule": [
                {"agent_name": "clarification", "priority": 1, "reason": "提取字段", "expected_output": "结构化行程字段"},
                {"agent_name": "search", "priority": 2, "reason": "检索", "expected_output": "外部数据"},
                {"agent_name": "plan", "priority": 3, "reason": "规划", "expected_output": "行程"}
              ]
            }
            """
        )

        data = await self.recognize("我想去南京", model=model)

        self.assertEqual("我想去南京", data["rewritten_query"])
        self.assertEqual({"destination": "南京"}, data["key_entities"])
        self.assertNotIn("为我个", data["rewritten_query"])
