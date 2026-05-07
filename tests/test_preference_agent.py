from __future__ import annotations

import unittest

from agents.preference_agent import PreferenceAgent
from tests.fakes import ExplodingLLM, RecordingLLM


class PreferenceAgentTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, query: str, model=None) -> dict:
        return await PreferenceAgent(model=model or ExplodingLLM()).run({"context": {"rewritten_query": query}})

    def preferences_by_type(self, result: dict, pref_type: str) -> list[dict]:
        return [item for item in result.get("preferences", []) if item.get("type") == pref_type]

    def first_preference(self, result: dict, pref_type: str) -> dict:
        matches = self.preferences_by_type(result, pref_type)
        self.assertTrue(matches, f"missing preference type {pref_type}")
        return matches[0]

    async def test_append_mode_with_keyword_hai(self):
        """'我还喜欢汉庭' → append 模式"""
        model = ExplodingLLM()

        result = await self.run_agent("我还喜欢汉庭", model=model)

        preference = self.first_preference(result, "hotel_brands")
        self.assertEqual("汉庭", preference["value"])
        self.assertEqual("append", preference["action"])
        self.assertEqual([], model.calls)

    async def test_append_mode_with_keyword_ye(self):
        """'我也喜欢如家' → append 模式"""
        model = ExplodingLLM()

        result = await self.run_agent("我也喜欢如家", model=model)

        preference = self.first_preference(result, "hotel_brands")
        self.assertEqual("如家", preference["value"])
        self.assertEqual("append", preference["action"])
        self.assertEqual([], model.calls)

    async def test_replace_mode_with_keyword_ban(self):
        """'我搬家到上海了' → replace 模式"""
        model = ExplodingLLM()

        result = await self.run_agent("我搬家到上海了", model=model)

        preference = self.first_preference(result, "home_location")
        self.assertEqual("上海", preference["value"])
        self.assertEqual("replace", preference["action"])
        self.assertEqual([], model.calls)

    async def test_replace_mode_with_keyword_gai(self):
        """'改成东航' → replace 模式"""
        model = ExplodingLLM()

        result = await self.run_agent("改成东航", model=model)

        preference = self.first_preference(result, "airlines")
        self.assertEqual("东航", preference["value"])
        self.assertEqual("replace", preference["action"])
        self.assertEqual([], model.calls)

    async def test_extract_hotel_brand(self):
        """'我喜欢住汉庭' → hotel_brands: ['汉庭']"""
        result = await self.run_agent("我喜欢住汉庭")

        self.assertEqual("汉庭", self.first_preference(result, "hotel_brands")["value"])

    async def test_extract_airline(self):
        """'我常坐东航' → airlines: ['东航']"""
        result = await self.run_agent("我常坐东航")

        self.assertEqual("东航", self.first_preference(result, "airlines")["value"])

    async def test_extract_seat_preference(self):
        """'我要靠窗的座位' → seat_preference: 'window'"""
        result = await self.run_agent("我要靠窗的座位")

        self.assertEqual("window", self.first_preference(result, "seat_preference")["value"])

    async def test_extract_budget_preference(self):
        """'我一般住300到500的酒店' → lodging_budget range"""
        result = await self.run_agent("我一般住300到500的酒店")

        preference = self.first_preference(result, "lodging_budget_per_night")
        self.assertEqual({"min": 300, "max": 500}, preference["value"])
        self.assertEqual("replace", preference["action"])

    async def test_extract_multiple_preferences(self):
        """'我喜欢汉庭，常坐东航，要靠窗座位' → 3条偏好"""
        result = await self.run_agent("我喜欢汉庭，常坐东航，要靠窗座位")

        self.assertEqual("汉庭", self.first_preference(result, "hotel_brands")["value"])
        self.assertEqual("东航", self.first_preference(result, "airlines")["value"])
        self.assertEqual("window", self.first_preference(result, "seat_preference")["value"])

    async def test_no_preference_detected(self):
        """'今天天气怎么样' → 无偏好"""
        model = RecordingLLM({"has_preferences": False, "preferences": [], "summary": "未识别到长期偏好。"})

        result = await self.run_agent("今天天气怎么样", model=model)

        self.assertFalse(result["has_preferences"])
        self.assertEqual([], result["preferences"])
        self.assertGreaterEqual(len(model.calls), 1)

    async def test_empty_input(self):
        """空字符串 → 无偏好"""
        model = RecordingLLM({"has_preferences": False, "preferences": [], "summary": "输入为空。"})

        result = await self.run_agent("", model=model)

        self.assertFalse(result["has_preferences"])
        self.assertEqual([], result["preferences"])
        self.assertGreaterEqual(len(model.calls), 1)

    async def test_implicit_preference_from_context(self):
        """'出差一般住好一点的' → 隐含高预算偏好"""
        result = await self.run_agent("出差一般住好一点的")

        preference = self.first_preference(result, "budget_level")
        self.assertEqual("品质型", preference["value"])
        self.assertEqual("replace", preference["action"])


if __name__ == "__main__":
    unittest.main()
