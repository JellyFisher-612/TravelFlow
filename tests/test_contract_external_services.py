"""Contract tests for external service clients.

Tests the HTTP/subprocess client logic with mocked responses.
Does NOT call real APIs — verifies parsing, error handling, and degradation.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class FakeMCPResult:
    """Simulates an MCP tool call result."""

    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]


class AmapServiceContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract tests for AmapService MCP client logic."""

    def _make_service(self):
        with patch.dict("os.environ", {"AMAP_MAPS_API_KEY": "test-key"}):
            from utils.amap_service import AmapService
            return AmapService(api_key="test-key")

    async def _mock_call_tool(self, service, tool_name: str, response_data: Any):
        """Patch _call_tool to return deterministic data."""
        async def fake_call(tool, args):
            return response_data
        service._call_tool = fake_call

    async def test_weather_parses_valid_response(self):
        service = self._make_service()
        weather_data = {"lives": [{"weather": "晴", "temperature": "22", "humidity": "50"}]}
        await self._mock_call_tool(service, "maps_weather", weather_data)

        result = await service.maps_weather("杭州")

        self.assertEqual("晴", result["lives"][0]["weather"])
        self.assertEqual("22", result["lives"][0]["temperature"])

    async def test_geocode_parses_valid_response(self):
        service = self._make_service()
        geocode_data = {"geocodes": [{"location": "120.1551,30.2741"}]}
        await self._mock_call_tool(service, "maps_geo", geocode_data)

        result = await service.maps_geo("西湖", city="杭州")

        self.assertEqual("120.1551,30.2741", result["geocodes"][0]["location"])

    async def test_geocode_handles_no_results(self):
        service = self._make_service()
        await self._mock_call_tool(service, "maps_geo", {"geocodes": []})

        result = await service.maps_geo("不存在的地方")

        self.assertEqual([], result["geocodes"])

    async def test_poi_search_parses_results(self):
        service = self._make_service()
        pois = [
            {"name": "西湖", "address": "杭州市西湖区", "type": "风景名胜"},
            {"name": "灵隐寺", "address": "杭州市西湖区灵隐路", "type": "风景名胜"},
        ]
        await self._mock_call_tool(service, "maps_text_search", pois)

        result = await service.maps_text_search("杭州", "景点")

        self.assertEqual(2, len(result))
        self.assertEqual("西湖", result[0]["name"])

    async def test_directions_parse_walking_route(self):
        service = self._make_service()
        route_data = {"route": {"paths": [{"distance": "1200", "duration": "900"}]}}
        await self._mock_call_tool(service, "maps_direction_walking", route_data)

        result = await service.maps_direction_walking("120.1,30.2", "120.2,30.3")

        self.assertIn("route", result)

    async def test_handles_malformed_json(self):
        service = self._make_service()
        # Simulate _call_tool returning raw text that's not valid JSON
        async def bad_call(tool, args):
            return "not json"
        service._call_tool = bad_call

        result = await service.maps_weather("杭州")

        # Should return the raw string, not crash
        self.assertEqual("not json", result)

    async def test_handles_empty_response(self):
        service = self._make_service()
        async def empty_call(tool, args):
            return ""
        service._call_tool = empty_call

        result = await service.maps_weather("杭州")

        self.assertEqual("", result)


class TrainServiceContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract tests for TrainService (12306 MCP) client logic."""

    def _make_service(self):
        from utils.train_service import TrainService
        return TrainService(command="echo", args=["test"])

    async def _mock_call_tool(self, service, response_data: Any):
        async def fake_call(tool, args):
            return response_data
        service._call_tool = fake_call

    async def test_query_trains_parses_valid_response(self):
        service = self._make_service()
        tickets = [
            {
                "train_code": "G100",
                "from_station": "上海",
                "to_station": "杭州",
                "start_time": "09:00",
                "arrive_time": "10:00",
                "duration": "01:00",
                "second_class_seat": "有票 73元",
            }
        ]
        await self._mock_call_tool(service, tickets)

        result = await service.get_tickets("2026-06-01", "上海", "杭州")

        self.assertEqual(1, len(result))
        self.assertEqual("G100", result[0]["train_code"])
        self.assertIn("73元", result[0]["second_class_seat"])

    async def test_query_trains_handles_no_results(self):
        service = self._make_service()
        await self._mock_call_tool(service, [])

        result = await service.get_tickets("2026-06-01", "上海", "杭州")

        self.assertEqual([], result)

    async def test_query_trains_handles_multiple_classes(self):
        service = self._make_service()
        tickets = [
            {
                "train_code": "G100",
                "second_class_seat": "有票 73元",
                "first_class_seat": "有票 117元",
                "business_seat": "有票 219元",
            }
        ]
        await self._mock_call_tool(service, tickets)

        result = await service.get_tickets("2026-06-01", "上海", "杭州")

        self.assertIn("73元", result[0]["second_class_seat"])
        self.assertIn("117元", result[0]["first_class_seat"])
        self.assertIn("219元", result[0]["business_seat"])

    async def test_decode_tool_result_with_text_content(self):
        """_decode_tool_result parses JSON text in MCP content."""
        from utils.train_service import TrainService
        service = self._make_service()

        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = json.dumps([{"train_code": "G200"}], ensure_ascii=False)
        mock_result.content = [mock_item]

        decoded = service._decode_tool_result(mock_result)

        self.assertEqual("G200", decoded[0]["train_code"])

    async def test_decode_tool_result_with_plain_text(self):
        """_decode_tool_result returns plain text when not valid JSON."""
        from utils.train_service import TrainService
        service = self._make_service()

        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = "No trains found"
        mock_result.content = [mock_item]

        decoded = service._decode_tool_result(mock_result)

        self.assertEqual("No trains found", decoded)

    async def test_decode_tool_result_with_none_content(self):
        """_decode_tool_result returns raw result when content is None."""
        from utils.train_service import TrainService
        service = self._make_service()

        mock_result = MagicMock()
        mock_result.content = None

        decoded = service._decode_tool_result(mock_result)

        self.assertEqual(mock_result, decoded)


# Needed for type annotation in test helper
from typing import Any


if __name__ == "__main__":
    unittest.main()
