"""Reusable test doubles for TravelFlow tests.

These fakes keep automated tests away from real LLMs and external services.
"""

from __future__ import annotations


class ExplodingLLM:
    """LLM double that fails the test if intent routing calls it."""

    def __init__(self) -> None:
        self.calls = []

    def with_structured_output(self, schema):
        self.calls.append(("with_structured_output", schema))
        raise AssertionError("Intent test attempted to call structured LLM output")

    async def ainvoke(self, messages):
        self.calls.append(("ainvoke", messages))
        raise AssertionError("Intent test attempted to call an LLM")


class FailingTextLLM:
    """LLM double that simulates provider outage for fallback tests."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error or RuntimeError("LLM unavailable")

    async def ainvoke(self, messages):
        self.calls.append(("ainvoke", messages))
        raise self.error


class FakeAgent:
    """Simple async business-agent fake for scheduler tests."""

    def __init__(self, name: str, response: dict | None = None) -> None:
        self.name = name
        self.response = response or {"success": True}
        self.calls = []

    async def run(self, state):
        self.calls.append(state)
        return {
            "agent_name": self.name,
            "success": True,
            "data": self.response,
        }


class RecordingAgent:
    """Business-agent fake that returns raw scheduler data and records calls."""

    def __init__(self, name: str, response: dict | None = None) -> None:
        self.name = name
        self.response = response or {}
        self.calls = []

    async def run(self, state):
        self.calls.append(state)
        return dict(self.response)


class FakeSearchService:
    """Deterministic search/MCP double that records all requested tools."""

    def __init__(self, responses: dict | None = None) -> None:
        self.responses = responses or {}
        self.calls = []

    async def call(self, tool_name: str, payload: dict):
        self.calls.append((tool_name, payload))
        return self.responses.get(tool_name, {"success": True, "data": []})


class FakeAmapService:
    """Amap MCP fake that returns deterministic weather, POIs, and routes."""

    def __init__(
        self,
        weather: dict | None = None,
        geocode: dict | None = None,
        pois_by_keyword: dict | None = None,
    ) -> None:
        self.weather = weather or {
            "lives": [{"weather": "晴", "temperature": "22", "humidity": "50"}]
        }
        self.geocode = geocode or {"geocodes": [{"location": "120.1551,30.2741"}]}
        self.pois_by_keyword = pois_by_keyword or {}
        self.calls = []

    async def maps_weather(self, city: str):
        self.calls.append(("maps_weather", {"city": city}))
        return self.weather

    async def maps_geo(self, address: str, city: str = ""):
        self.calls.append(("maps_geo", {"address": address, "city": city}))
        return self.geocode

    async def maps_text_search(self, city: str, keywords: str):
        self.calls.append(("maps_text_search", {"city": city, "keywords": keywords}))
        if keywords in self.pois_by_keyword:
            return self.pois_by_keyword[keywords]
        return [
            {
                "id": f"{city}-{keywords}",
                "name": f"{city}{keywords}候选",
                "address": f"{city}{keywords}地址",
                "type": "travel",
                "location": {"longitude": 120.1, "latitude": 30.2},
            }
        ]

    async def maps_direction_walking(self, origin: str, destination: str):
        self.calls.append(
            ("maps_direction_walking", {"origin": origin, "destination": destination})
        )
        return {"route": "walking", "distance": "1000"}

    async def maps_direction_driving(self, origin: str, destination: str):
        self.calls.append(
            ("maps_direction_driving", {"origin": origin, "destination": destination})
        )
        return {"route": "driving", "distance": "3000"}

    async def maps_distance(self, origins: str, destination: str, type: str = "1"):
        self.calls.append(
            (
                "maps_distance",
                {"origins": origins, "destination": destination, "type": type},
            )
        )
        return {"results": [{"distance": "3000", "duration": "600"}]}


class FakeTrainService:
    """12306 MCP fake that records ticket lookups and returns fixed trains."""

    def __init__(self, tickets: list | None = None) -> None:
        self.tickets = tickets or [
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
        self.calls = []

    async def get_tickets(
        self,
        date: str,
        from_station: str,
        to_station: str,
        train_filter_flags: str = "",
        earliest_start_time: int | None = None,
        latest_start_time: int | None = None,
        sort_flag: str = "",
        sort_reverse: bool = False,
        limited_num: int = 10,
        response_format: str = "json",
    ):
        self.calls.append(
            {
                "date": date,
                "from_station": from_station,
                "to_station": to_station,
                "train_filter_flags": train_filter_flags,
                "earliest_start_time": earliest_start_time,
                "latest_start_time": latest_start_time,
                "sort_flag": sort_flag,
                "sort_reverse": sort_reverse,
                "limited_num": limited_num,
                "response_format": response_format,
            }
        )
        return self.tickets


class FakeMemory:
    """In-memory preference/history double."""

    def __init__(self, values: dict | None = None) -> None:
        self.values = values or {}
        self.calls = []

    async def get(self, key: str, default=None):
        self.calls.append(("get", key))
        return self.values.get(key, default)

    async def set(self, key: str, value):
        self.calls.append(("set", key, value))
        self.values[key] = value
        return True
