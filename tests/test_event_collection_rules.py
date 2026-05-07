from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from agents.clarification_agent import EventCollectionAgent
from tests.fakes import ExplodingLLM, RecordingLLM


def load_event_collection_agent_class():
    return EventCollectionAgent


class EventCollectionRuleTests(unittest.IsolatedAsyncioTestCase):
    async def extract(self, query: str, model=None) -> dict:
        agent = EventCollectionAgent(model=model or ExplodingLLM())
        return await agent.run({"context": {"rewritten_query": query}})

    async def test_budget_range_input_is_extracted_without_calling_llm(self):
        agent_cls = load_event_collection_agent_class()
        model = ExplodingLLM()
        agent = agent_cls(model=model)

        result = await agent.run(
            {
                "context": {
                    "rewritten_query": "本次行程预算选择舒适型，住宿每晚300到600元，兼顾体验和性价比。"
                }
            }
        )

        self.assertEqual("舒适型", result["budget_level"])
        self.assertEqual(300, result["lodging_budget_per_night_min"])
        self.assertEqual(600, result["lodging_budget_per_night_max"])
        self.assertIsNone(result["lodging_budget_per_night"])
        self.assertIn("origin", result["missing_info"])
        self.assertIn("destination", result["missing_info"])
        self.assertNotIn("budget_level", result["missing_info"])
        self.assertEqual([], model.calls)

    async def test_minimal_destination_does_not_extract_desire_as_origin(self):
        agent_cls = load_event_collection_agent_class()
        model = ExplodingLLM()
        agent = agent_cls(model=model)

        result = await agent.run({"context": {"rewritten_query": "我想去南京"}})

        self.assertIsNone(result["origin"])
        self.assertEqual("南京", result["destination"])
        self.assertIn("origin", result["missing_info"])
        self.assertIn("start_date", result["missing_info"])
        self.assertIn("duration_days", result["missing_info"])
        self.assertEqual([], model.calls)

    async def test_extract_dates_with_chinese_relative(self):
        """明天、后天、下周一等相对日期"""
        today = datetime.now().date()
        cases = [
            ("明天去杭州三天", today + timedelta(days=1)),
            ("后天去杭州三天", today + timedelta(days=2)),
            ("下周一去杭州三天", today + timedelta(days=(7 - today.weekday()) % 7 or 7)),
        ]

        for query, expected_start in cases:
            with self.subTest(query=query):
                model = ExplodingLLM()
                result = await self.extract(query, model=model)

                self.assertEqual(expected_start.isoformat(), result["start_date"])
                self.assertEqual((expected_start + timedelta(days=2)).isoformat(), result["end_date"])
                self.assertEqual(3, result["duration_days"])
                self.assertEqual([], model.calls)

    async def test_extract_dates_with_absolute_format(self):
        """3月11日、2026-03-11等绝对日期"""
        current_year = datetime.now().year
        cases = [
            ("3月11日去杭州三天", f"{current_year}-03-11"),
            ("2026-03-11去杭州三天", "2026-03-11"),
        ]

        for query, expected_start in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_start, result["start_date"])
                self.assertEqual(3, result["duration_days"])

    async def test_extract_dates_with_range(self):
        """3月11日到18日、下周一周五"""
        current_year = datetime.now().year
        today = datetime.now().date()
        next_monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
        next_friday = next_monday + timedelta(days=4)

        cases = [
            ("3月11日到18日去杭州", f"{current_year}-03-11", f"{current_year}-03-18", 8),
            ("下周一周五去杭州", next_monday.isoformat(), next_friday.isoformat(), 5),
        ]

        for query, expected_start, expected_end, expected_duration in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_start, result["start_date"])
                self.assertEqual(expected_end, result["end_date"])
                self.assertEqual(expected_duration, result["duration_days"])

    async def test_extract_duration_days(self):
        """三天、一周、5天"""
        cases = [("去杭州三天", 3), ("去杭州一周", 7), ("去杭州5天", 5)]

        for query, expected_duration in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_duration, result["duration_days"])

    async def test_extract_duration_weeks(self):
        """两周、一个礼拜"""
        cases = [("去杭州两周", 14), ("去杭州一个礼拜", 7)]

        for query, expected_duration in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_duration, result["duration_days"])

    async def test_extract_destination_chinese_cities(self):
        """北京、上海、杭州、成都"""
        for city in ["北京", "上海", "杭州", "成都"]:
            with self.subTest(city=city):
                result = await self.extract(f"我想去{city}旅游")

                self.assertEqual(city, result["destination"])
                self.assertIsNone(result["origin"])

    async def test_extract_destination_with_direction(self):
        """去杭州、到北京、飞上海"""
        cases = [("去杭州", "杭州"), ("到北京", "北京"), ("飞上海", "上海")]

        for query, expected_destination in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_destination, result["destination"])

    async def test_extract_destination_does_not_grab_origin(self):
        """从北京去杭州 → 只提取杭州，不提取北京"""
        result = await self.extract("从北京去杭州")

        self.assertEqual("北京", result["origin"])
        self.assertEqual("杭州", result["destination"])

    async def test_extract_origin_patterns(self):
        """从北京出发、从上海去、我人在深圳"""
        cases = [
            ("从北京出发", "北京", None),
            ("从上海去杭州", "上海", "杭州"),
            ("我人在深圳", "深圳", None),
        ]

        for query, expected_origin, expected_destination in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_origin, result["origin"])
                self.assertEqual(expected_destination, result["destination"])

    async def test_extract_purpose_business(self):
        """出差、商务、开会"""
        cases = [("去杭州出差", "出差"), ("去杭州商务拜访", "商务"), ("去杭州开会", "商务")]

        for query, expected_purpose in cases:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual(expected_purpose, result["trip_purpose"])

    async def test_extract_purpose_leisure(self):
        """旅游、度假、玩"""
        for query in ["去杭州旅游", "去杭州度假", "去杭州玩"]:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual("旅游", result["trip_purpose"])

    async def test_extract_pace_relaxed(self):
        """轻松、休闲、不要太赶"""
        for query in ["去杭州轻松一点", "去杭州休闲一点", "去杭州不要太赶"]:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual("轻松", result["pace_preference"])

    async def test_extract_pace_intensive(self):
        """紧凑、多去几个地方、赶一点"""
        for query in ["去杭州节奏紧凑", "去杭州多去几个地方", "去杭州赶一点"]:
            with self.subTest(query=query):
                result = await self.extract(query)

                self.assertEqual("紧凑", result["pace_preference"])

    async def test_extract_all_entities_at_once(self):
        """下周三从北京去杭州出差三天，预算600以内"""
        today = datetime.now().date()
        days_ahead = (2 - today.weekday()) % 7 or 7
        expected_start = today + timedelta(days=days_ahead)

        result = await self.extract("下周三从北京去杭州出差三天，预算600以内")

        self.assertEqual("北京", result["origin"])
        self.assertEqual("杭州", result["destination"])
        self.assertEqual(expected_start.isoformat(), result["start_date"])
        self.assertEqual(3, result["duration_days"])
        self.assertEqual("舒适型", result["budget_level"])
        self.assertEqual("出差", result["trip_purpose"])

    async def test_empty_input(self):
        """空字符串"""
        model = RecordingLLM(
            {
                "origin": None,
                "destination": None,
                "missing_info": ["所有信息"],
                "extracted_count": 0,
            }
        )

        result = await self.extract("", model=model)

        self.assertIsNone(result.get("origin"))
        self.assertIsNone(result.get("destination"))
        self.assertEqual(0, result["extracted_count"])
        self.assertGreaterEqual(len(model.calls), 1)

    async def test_no_entities_found(self):
        """今天天气真好"""
        model = RecordingLLM({"has_trip_facts": False, "missing_info": ["所有信息"], "extracted_count": 0})

        result = await self.extract("今天天气真好", model=model)

        self.assertEqual(0, result["extracted_count"])
        self.assertIn("所有信息", result["missing_info"])
        self.assertGreaterEqual(len(model.calls), 1)

    async def test_ambiguous_conflicting_entities(self):
        """从北京去北京 (origin == destination)"""
        model = ExplodingLLM()

        result = await self.extract("从北京去北京", model=model)

        self.assertEqual("北京", result["origin"])
        self.assertEqual("北京", result["destination"])
        self.assertEqual([], model.calls)

    async def test_rule_miss_falls_back_to_llm(self):
        """规则无法提取时应该调用 LLM（用 mock 验证）"""
        model = RecordingLLM(
            {
                "origin": "上海",
                "destination": "莫干山",
                "duration_days": 2,
                "missing_info": ["start_date", "budget_level", "pace_preference"],
                "extracted_count": 3,
                "summary": "已通过模型补充识别目的地。",
            }
        )

        result = await self.extract("找个江浙沪山里安静的地方住两晚", model=model)

        self.assertEqual("莫干山", result["destination"])
        self.assertEqual(2, result["duration_days"])
        self.assertEqual("with_structured_output", model.calls[0][0])
        self.assertTrue(any(call[0] == "structured_ainvoke" for call in model.calls))


if __name__ == "__main__":
    unittest.main()
