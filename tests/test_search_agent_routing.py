from __future__ import annotations

import unittest
from unittest.mock import patch

import agents.search_agent as query_info
from tests.fakes import FakeAmapService, FakeTrainService


def load_query_info_module():
    return query_info


query_info = load_query_info_module()


class ExplodingDDGS:
    def __init__(self):
        raise AssertionError("DDGS should not be used by this search routing test")


class SearchAgentRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_weather_query_uses_amap_weather_without_web_search(self):
        amap = FakeAmapService(
            weather={
                "lives": [
                    {
                        "weather": "多云",
                        "temperature": "24",
                        "humidity": "61",
                        "winddirection": "东风",
                    }
                ]
            }
        )
        train = FakeTrainService()

        with (
            patch.object(query_info, "AmapService", new=lambda: amap),
            patch.object(query_info, "TrainService", new=lambda: train),
            patch.object(query_info, "DDGS", new=ExplodingDDGS),
            patch.object(query_info, "DDGS_AVAILABLE", new=True),
        ):
            agent = query_info.InformationQueryAgent(model=None)
            result = await agent.run(
                {
                    "user_query": "杭州明天天气怎么样？",
                    "context": {"rewritten_query": "杭州明天天气怎么样？"},
                }
            )

        self.assertEqual("天气查询", result["query_type"])
        self.assertTrue(result["query_success"])
        self.assertEqual("杭州", result["results"]["location_name"])
        self.assertIn("多云", result["results"]["summary"])
        self.assertEqual([("maps_weather", {"city": "杭州"})], amap.calls)
        self.assertEqual([], train.calls)

    async def test_train_query_uses_12306_and_marks_official_transport_source(self):
        amap = FakeAmapService()
        train = FakeTrainService(
            tickets=[
                {
                    "train_code": "G7311",
                    "from_station": "上海",
                    "to_station": "杭州",
                    "start_time": "08:30",
                    "arrive_time": "09:30",
                    "duration": "01:00",
                    "second_class_seat": "有票 73元",
                }
            ]
        )

        with (
            patch.object(query_info, "AmapService", new=lambda: amap),
            patch.object(query_info, "TrainService", new=lambda: train),
            patch.object(query_info, "DDGS", new=ExplodingDDGS),
            patch.object(query_info, "DDGS_AVAILABLE", new=True),
        ):
            agent = query_info.InformationQueryAgent(model=None)
            result = await agent.run(
                {
                    "user_query": "2026-05-01上海到杭州的高铁有哪些？",
                    "context": {
                        "rewritten_query": "2026-05-01上海到杭州的高铁有哪些？"
                    },
                }
            )

        self.assertEqual("火车车次查询", result["query_type"])
        self.assertTrue(result["query_success"])
        self.assertTrue(result["verified"])
        self.assertTrue(result["requires_official_source"])
        self.assertEqual("high", result["trust_level"])
        self.assertEqual("G7311", result["results"]["tickets"][0]["train_code"])
        self.assertEqual(
            {
                "date": "2026-05-01",
                "from_station": "上海",
                "to_station": "杭州",
                "train_filter_flags": "G",
                "earliest_start_time": None,
                "latest_start_time": None,
                "sort_flag": "",
                "sort_reverse": False,
                "limited_num": 10,
                "response_format": "json",
            },
            train.calls[0],
        )
        self.assertEqual([], amap.calls)

    async def test_trip_search_bundle_contains_demand_profile_and_quality_policy(self):
        amap = FakeAmapService()
        train = FakeTrainService(
            tickets=[
                {
                    "start_train_code": "G7311",
                    "from_station": "上海",
                    "to_station": "杭州",
                    "start_time": "08:30",
                    "arrive_time": "09:30",
                    "lishi": "01:00",
                    "prices": [
                        {"seat_name": "二等座", "num": "有", "price": 73},
                        {"seat_name": "一等座", "num": "5", "price": 117},
                    ],
                },
                {
                    "start_train_code": "G7312",
                    "from_station": "杭州",
                    "to_station": "上海",
                    "start_time": "17:30",
                    "arrive_time": "18:30",
                    "lishi": "01:00",
                    "prices": [
                        {"seat_name": "二等座", "num": "8", "price": 73},
                        {"seat_name": "商务座", "num": "无", "price": 220},
                    ],
                },
            ]
        )

        with (
            patch.object(query_info, "AmapService", new=lambda: amap),
            patch.object(query_info, "TrainService", new=lambda: train),
            patch.object(query_info, "DDGS", new=ExplodingDDGS),
            patch.object(query_info, "DDGS_AVAILABLE", new=True),
        ):
            agent = query_info.InformationQueryAgent(model=None)

            async def fake_web_search(query: str):
                return {
                    "query_type": "网络搜索",
                    "query_success": True,
                    "verified": False,
                    "requires_official_source": True,
                    "trust_level": "medium",
                    "results": {
                        "summary": f"{query} 未找到官方来源",
                        "sources": [],
                        "official_sources": [],
                    },
                }

            agent._web_search = fake_web_search
            result = await agent.run(
                {
                    "user_query": "我想带孩子去杭州玩三天，想要小众、人少、好吃一点，门票怎么预约？",
                    "context": {
                        "rewritten_query": "我想带孩子去杭州玩三天，想要小众、人少、好吃一点，门票怎么预约？",
                        "event_data": {
                            "origin": "上海",
                            "destination": "杭州",
                            "start_date": "2026-05-01",
                            "duration_days": 3,
                            "budget_level": "舒适型",
                            "pace_preference": "轻松",
                            "transportation_preference": "高铁",
                        },
                    },
                }
            )

        self.assertEqual("行程相关信息查询", result["query_type"])
        self.assertTrue(result["query_success"])
        results = result["results"]
        bundle = results["search_bundle"]
        profile = bundle["planning"]["demand_profile"]
        quality = bundle["quality"]

        self.assertTrue({"family", "food", "hidden_gems"}.issubset(profile["focus_tags"]))
        self.assertIn("official_attraction_policy", bundle["planning"]["must_verify"])
        self.assertIn("official_attraction_policy", quality["unverified_must_verify"])
        self.assertIn("outbound_train_options", quality["verified_fields"])
        self.assertIn("return_train_options", quality["verified_fields"])
        self.assertTrue(quality["hard_constraints_require_official_sources"])
        self.assertTrue(
            results["search_trust_policy"][
                "unverified_hard_constraints_must_not_be_written_as_confirmed"
            ]
        )
        self.assertIn("亲子 景点 儿童乐园 科技馆", results["search_keywords"])
        self.assertGreaterEqual(len(train.calls), 2)
        self.assertEqual("G7311", bundle["transport"]["outbound_trains"][0]["train_code"])
        self.assertEqual("有票 73元", bundle["transport"]["outbound_trains"][0]["second_class_seat"])
        self.assertEqual("剩余5张票 117元", bundle["transport"]["outbound_trains"][0]["first_class_seat"])
        self.assertEqual("G7312", bundle["transport"]["return_trains"][0]["train_code"])
        self.assertEqual("剩余8张票 73元", bundle["transport"]["return_trains"][0]["second_class_seat"])
        self.assertEqual("无票 220元", bundle["transport"]["return_trains"][0]["business_seat"])
        self.assertEqual(
            {
                "date": "2026-05-01",
                "from_station": "上海",
                "to_station": "杭州",
                "train_filter_flags": "G",
                "earliest_start_time": 7,
                "latest_start_time": 14,
                "sort_flag": "startTime",
                "sort_reverse": False,
                "limited_num": 80,
                "response_format": "json",
            },
            train.calls[0],
        )
        self.assertEqual(
            {
                "date": "2026-05-03",
                "from_station": "杭州",
                "to_station": "上海",
                "train_filter_flags": "G",
                "earliest_start_time": 14,
                "latest_start_time": 22,
                "sort_flag": "startTime",
                "sort_reverse": True,
                "limited_num": 80,
                "response_format": "json",
            },
            train.calls[1],
        )
        self.assertTrue(any(call[0] == "maps_weather" for call in amap.calls))
        self.assertTrue(any(call[0] == "maps_text_search" for call in amap.calls))


if __name__ == "__main__":
    unittest.main()
