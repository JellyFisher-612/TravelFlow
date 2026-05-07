from __future__ import annotations

import unittest

from agents.plan_agent import ItineraryPlanningAgent
from tests.fakes import FailingTextLLM, RecordingLLM


def realistic_itinerary_response(budget_note: str = "舒适型预算，住宿优先地铁便利") -> dict:
    return {
        "planning_complete": True,
        "summary": "已生成杭州三日行程。",
        "itinerary": {
            "title": "杭州三日商务旅行计划",
            "duration": "3天",
            "budget_strategy": budget_note,
            "transportation": {
                "arrival": "从北京到杭州的大交通需在官方渠道核验。",
                "local": "市内优先地铁，跨区使用打车。",
            },
            "hotel_recommendations": [
                {"name": "地铁站周边经济型连锁酒店", "area": "武林广场", "status": "needs_official_check"}
            ],
            "food_recommendations": [
                {"name": "湖滨银泰周边杭帮菜", "area": "湖滨", "status": "unverified"}
            ],
            "daily_plans": [
                {
                    "day": 1,
                    "activities": [
                        {"time": "09:00 上午", "activity": "西湖", "transport": "地铁到龙翔桥后步行"},
                        {"time": "14:00 下午", "activity": "灵隐寺", "transport": "打车或公交接驳"},
                        {"time": "19:00 晚上", "activity": "湖滨晚餐", "transport": "地铁返回酒店"},
                    ],
                },
                {
                    "day": 2,
                    "activities": [
                        {"time": "09:30 上午", "activity": "西溪湿地", "transport": "地铁换乘公交"},
                        {"time": "15:00 下午", "activity": "商务会面", "transport": "打车控制时间"},
                    ],
                },
                {
                    "day": 3,
                    "activities": [
                        {"time": "10:00 上午", "activity": "中国茶叶博物馆", "transport": "打车"},
                        {"time": "15:00 下午", "activity": "返程预留", "transport": "前往杭州东站需核验车次"},
                    ],
                },
            ],
        },
        "verification_summary": {"status": "needs_official_check"},
    }


class PlanAgentTests(unittest.IsolatedAsyncioTestCase):
    def build_event(self, **overrides) -> dict:
        event = {
            "origin": "北京",
            "destination": "杭州",
            "start_date": "2026-06-01",
            "duration_days": 3,
            "budget_level": "舒适型",
            "pace_preference": "轻松",
            "missing_info": [],
        }
        event.update(overrides)
        return event

    def build_search_results(self) -> dict:
        pois = [
            {"name": "西湖", "adname": "西湖区", "type": "风景名胜", "source_keywords": "景点"},
            {"name": "灵隐寺", "adname": "西湖区", "type": "风景名胜", "source_keywords": "景点"},
            {"name": "西溪湿地", "adname": "西湖区", "type": "公园", "source_keywords": "景点"},
            {"name": "中国茶叶博物馆", "adname": "西湖区", "type": "博物馆", "source_keywords": "景点"},
            {"name": "湖滨银泰", "adname": "上城区", "type": "商圈", "source_keywords": "美食 餐厅"},
            {"name": "武林广场酒店圈", "adname": "拱墅区", "type": "酒店", "source_keywords": "经济型酒店"},
        ]
        return {
            "destination": "杭州",
            "pois": pois,
            "pois_by_category": {
                "美食 餐厅": [pois[4]],
                "经济型酒店": [pois[5]],
            },
            "weather": {"summary": "多云，22到28度"},
            "routes": [{"from": "西湖", "to": "灵隐寺", "duration": "35分钟"}],
            "supplemental_search": [
                {"verified": True, "source": "12306", "content": "需到官方渠道核验具体车次。"}
            ],
        }

    def build_state(self, event: dict | None = None, search_results: dict | None = None, context: dict | None = None) -> dict:
        event = event or self.build_event()
        context_info = {"rewritten_query": "帮我规划从北京去杭州出差三天", "user_preferences": {}}
        if context:
            context_info.update(context)
        return {
            "context": context_info,
            "previous_results": [
                {"agent_name": "clarification", "result": {"data": event}},
                {"agent_name": "search", "result": {"data": {"results": search_results or self.build_search_results()}}},
            ],
        }

    async def run_agent(self, model, state: dict | None = None) -> dict:
        return await ItineraryPlanningAgent(model=model).run(state or self.build_state())

    async def test_run_with_complete_context(self):
        """完整上下文（目的地、日期、偏好）生成行程"""
        result = await self.run_agent(RecordingLLM(realistic_itinerary_response()))

        self.assertTrue(result["planning_complete"])
        self.assertIn("杭州", result["itinerary"]["title"])
        self.assertEqual(3, len(result["itinerary"]["daily_plans"]))

    async def test_run_with_minimal_context(self):
        """最少信息也能生成合理建议"""
        event = self.build_event(origin=None, budget_level=None, pace_preference=None)
        state = self.build_state(event=event, search_results={}, context={"search_refinement_count": 1})

        result = await self.run_agent(RecordingLLM(realistic_itinerary_response("基础预算待补充")), state)

        self.assertTrue(result["planning_complete"])
        self.assertIn("daily_plans", result["itinerary"])

    async def test_run_includes_daily_schedule(self):
        """输出包含每日时间安排"""
        result = await self.run_agent(RecordingLLM(realistic_itinerary_response()))

        times = [
            activity["time"]
            for day in result["itinerary"]["daily_plans"]
            for activity in day.get("activities", [])
        ]
        self.assertTrue(any("09:00" in time or "上午" in time for time in times))
        self.assertTrue(any("下午" in time for time in times))

    async def test_run_includes_transport_info(self):
        """输出包含交通信息"""
        result = await self.run_agent(RecordingLLM(realistic_itinerary_response()))

        self.assertIn("transportation", result["itinerary"])
        self.assertIn("local", result["itinerary"]["transportation"])

    async def test_run_includes_hotel_recommendation(self):
        """输出包含住宿建议"""
        result = await self.run_agent(RecordingLLM(realistic_itinerary_response()))

        self.assertEqual("needs_official_check", result["itinerary"]["hotel_recommendations"][0]["status"])

    async def test_run_includes_food_recommendation(self):
        """输出包含餐饮建议"""
        result = await self.run_agent(RecordingLLM(realistic_itinerary_response()))

        self.assertIn("杭帮菜", result["itinerary"]["food_recommendations"][0]["name"])

    async def test_run_respects_budget_level(self):
        """经济型预算 vs 豪华型预算输出不同"""
        def response_for_prompt(prompt: str) -> dict:
            if '"budget_level": "品质型"' in prompt:
                return realistic_itinerary_response("品质型预算，优先高便利酒店和舒适交通")
            return realistic_itinerary_response("经济型预算，优先地铁和连锁酒店")

        economy_result = await self.run_agent(
            RecordingLLM(response_for_prompt),
            self.build_state(event=self.build_event(budget_level="经济型")),
        )
        luxury_result = await self.run_agent(
            RecordingLLM(response_for_prompt),
            self.build_state(event=self.build_event(budget_level="品质型")),
        )

        self.assertNotEqual(
            economy_result["itinerary"]["budget_strategy"],
            luxury_result["itinerary"]["budget_strategy"],
        )
        self.assertIn("经济型", economy_result["itinerary"]["budget_strategy"])
        self.assertIn("品质型", luxury_result["itinerary"]["budget_strategy"])

    async def test_run_handles_llm_failure(self):
        """LLM 失败时返回降级提示而非崩溃"""
        result = await self.run_agent(FailingTextLLM())

        self.assertTrue(result["planning_complete"])
        self.assertIn("模型结构化输出不可用", result["summary"])
        self.assertIn("daily_plans", result["itinerary"])

    async def test_run_handles_empty_search_results(self):
        """搜索结果为空时仍能生成基础行程"""
        state = self.build_state(search_results={}, context={"search_refinement_count": 1})

        result = await self.run_agent(FailingTextLLM(), state)

        self.assertTrue(result["planning_complete"])
        self.assertEqual("杭州3日旅行计划", result["itinerary"]["title"])
        self.assertGreaterEqual(len(result["itinerary"]["daily_plans"]), 1)


if __name__ == "__main__":
    unittest.main()
