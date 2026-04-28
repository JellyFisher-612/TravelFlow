"""Amap MCP client.

This module calls the official Amap MCP server through the standard MCP
stdio transport:

    npx -y @amap/amap-maps-mcp-server

The public methods intentionally match the common Amap MCP tool names.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from config import AMAP_CONFIG


class AmapService:
    """High-level wrapper around Amap MCP tools."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        command: str = "npx",
        args: Optional[List[str]] = None,
    ) -> None:
        cfg = AMAP_CONFIG or {}
        self.api_key = api_key or os.getenv("AMAP_MAPS_API_KEY") or os.getenv("AMAP_API_KEY") or cfg.get("api_key")
        self.command = command
        self.args = args or ["-y", "@amap/amap-maps-mcp-server"]
        if not self.api_key:
            raise ValueError("AMAP_MAPS_API_KEY/AMAP_API_KEY 未配置，且 config.AMAP_CONFIG['api_key'] 为空")

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise RuntimeError("缺少 MCP Python SDK，请先安装 requirements.txt 中的 mcp 依赖") from exc

        clean_args = {key: value for key, value in arguments.items() if value not in (None, "")}
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={
                **os.environ,
                "AMAP_MAPS_API_KEY": self.api_key,
            },
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, clean_args)
                return self._decode_tool_result(result)

    def _decode_tool_result(self, result: Any) -> Any:
        content = getattr(result, "content", None)
        if content is None:
            return result

        decoded_items: List[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None:
                decoded_items.append(item)
                continue
            decoded_items.append(self._parse_text_payload(text))

        if len(decoded_items) == 1:
            return decoded_items[0]
        return decoded_items

    @staticmethod
    def _parse_text_payload(text: str) -> Any:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return stripped

    # ------------------------- MCP tool methods -------------------------

    async def maps_geo(self, address: str, city: str = "") -> Any:
        return await self._call_tool("maps_geo", {"address": address, "city": city})

    async def maps_regeocode(self, location: str) -> Any:
        return await self._call_tool("maps_regeocode", {"location": location})

    async def maps_ip_location(self, ip: str = "") -> Any:
        return await self._call_tool("maps_ip_location", {"ip": ip})

    async def maps_weather(self, city: str) -> Any:
        return await self._call_tool("maps_weather", {"city": city})

    async def maps_direction_driving(self, origin: str, destination: str) -> Any:
        return await self._call_tool(
            "maps_direction_driving",
            {"origin": origin, "destination": destination},
        )

    async def maps_direction_walking(self, origin: str, destination: str) -> Any:
        return await self._call_tool(
            "maps_direction_walking",
            {"origin": origin, "destination": destination},
        )

    async def maps_bicycling(self, origin: str, destination: str) -> Any:
        return await self._call_tool(
            "maps_bicycling",
            {"origin": origin, "destination": destination},
        )

    async def maps_distance(self, origins: str, destination: str, type: str = "1") -> Any:
        return await self._call_tool(
            "maps_distance",
            {"origins": origins, "destination": destination, "type": type},
        )

    async def maps_text_search(self, keywords: str, city: str = "") -> List[Dict[str, Any]]:
        raw = await self._call_tool("maps_text_search", {"keywords": keywords, "city": city})
        return self._extract_pois(raw)

    async def maps_around_search(self, location: str, radius: int = 1000) -> List[Dict[str, Any]]:
        raw = await self._call_tool("maps_around_search", {"location": location, "radius": radius})
        return self._extract_pois(raw)

    async def maps_search_detail(self, id: str) -> Any:
        return await self._call_tool("maps_search_detail", {"id": id})

    # ------------------------- Backward-compatible aliases -------------------------

    async def geocode(self, address: str, city: str = "") -> Any:
        return await self.maps_geo(address=address, city=city)

    async def reverse_geocode(self, location: str) -> Any:
        return await self.maps_regeocode(location=location)

    async def search_poi(self, city: str, keywords: str, **_: Any) -> List[Dict[str, Any]]:
        return await self.maps_text_search(keywords=keywords, city=city)

    async def get_weather(self, city: str, **_: Any) -> Any:
        return await self.maps_weather(city=city)

    async def route_plan(self, origin: str, destination: str, mode: str = "walking", **_: Any) -> Any:
        if mode == "driving":
            return await self.maps_direction_driving(origin=origin, destination=destination)
        if mode == "bicycling":
            return await self.maps_bicycling(origin=origin, destination=destination)
        return await self.maps_direction_walking(origin=origin, destination=destination)

    # ------------------------- Normalization helpers -------------------------

    def _extract_pois(self, raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            if raw and all(isinstance(item, dict) for item in raw):
                return [self._normalize_poi(item) for item in raw]
            return []

        if not isinstance(raw, dict):
            return []

        candidates = raw.get("pois") or raw.get("data") or raw.get("results") or []
        if isinstance(candidates, dict):
            candidates = candidates.get("pois") or candidates.get("list") or []
        if not isinstance(candidates, list):
            return []
        return [self._normalize_poi(item) for item in candidates if isinstance(item, dict)]

    def _normalize_poi(self, poi: Dict[str, Any]) -> Dict[str, Any]:
        location = poi.get("location")
        if isinstance(location, str):
            lng, lat = self._parse_location(location)
            location_value: Any = {"longitude": lng, "latitude": lat}
        else:
            location_value = location or {}

        return {
            "id": poi.get("id") or poi.get("uid"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "cityname": poi.get("cityname") or poi.get("city"),
            "adname": poi.get("adname") or poi.get("district"),
            "type": poi.get("type"),
            "location": location_value,
            "tel": poi.get("tel"),
            "website": poi.get("website"),
            "business_area": poi.get("business_area"),
            "raw": poi,
        }

    @staticmethod
    def _parse_location(loc_str: str) -> tuple[Optional[float], Optional[float]]:
        if loc_str and "," in loc_str:
            try:
                lng_str, lat_str = loc_str.split(",", 1)
                return float(lng_str), float(lat_str)
            except Exception:
                return None, None
        return None, None
