from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List

from utils.json_parser import robust_json_parse
from utils.langchain_runtime import ainvoke_text
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

from .common import (
    SearchPlanOutput,
    SummaryOutput,
    classify_source,
    is_suspicious_url,
    requires_official_source,
)

logger = logging.getLogger(__name__)

class TripSearchExecutionMixin:
    async def _trip_info_query(
        self,
        event_data: Dict[str, Any],
        user_query: str,
        refinement_requests: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """针对行程规划场景生成检索计划，并调用垂直工具返回结构化素材包。"""

        destination = (event_data.get("destination") or "").strip()
        if not destination:
            raise ValueError("event_data 中缺少 destination 字段，无法进行行程信息查询")

        amap = self._new_amap_service()
        search_plan = await self._build_trip_search_plan(event_data, user_query, refinement_requests or [])

        geocodes_task = asyncio.create_task(amap.maps_geo(address=destination))
        weather_task = asyncio.create_task(amap.maps_weather(city=destination))
        poi_tasks = search_plan.get("tasks_by_type", {}).get("poi_search", [])

        async def query_keyword(keyword: str) -> tuple[str, List[Dict[str, Any]]]:
            try:
                return keyword, await amap.maps_text_search(city=destination, keywords=keyword)
            except Exception as e:
                logger.warning("Amap POI query failed for %s/%s: %s", destination, keyword, e)
                return keyword, []

        poi_query_results = await asyncio.gather(
            *[query_keyword(str(task.get("keywords", ""))) for task in poi_tasks[:10]]
        )

        pois: List[Dict[str, Any]] = []
        pois_by_category: Dict[str, List[Dict[str, Any]]] = {}
        seen_poi_ids = set()
        for keywords, queried in poi_query_results:
            category_items: List[Dict[str, Any]] = []
            for poi in queried:
                poi_key = poi.get("id") or poi.get("name")
                if not poi_key:
                    continue
                poi["source_keywords"] = keywords
                category_items.append(poi)
                if poi_key in seen_poi_ids:
                    continue
                seen_poi_ids.add(poi_key)
                pois.append(poi)
            pois_by_category[keywords] = category_items[:10]

        try:
            geocodes = await geocodes_task
        except Exception as e:
            logger.warning("Amap geocode query failed for %s: %s", destination, e)
            geocodes = {"error": str(e)}

        try:
            weather = await weather_task
        except Exception as e:
            logger.warning("Amap weather query failed for %s: %s", destination, e)
            weather = {"error": str(e)}

        transport = await self._execute_transport_tasks(search_plan, user_query)
        selected_scenic = self._select_scenic_pois(pois_by_category, pois)
        nearby = await self._query_nearby_trip_support(amap, destination, selected_scenic)
        routes, distances = await self._query_trip_routes(amap, pois_by_category, pois)

        supplemental_search: List[Dict[str, Any]] = []
        for request in search_plan.get("tasks_by_type", {}).get("web_search", [])[:4]:
            keywords = str(request.get("keywords") or request.get("query") or "").strip()
            if not keywords or keywords in {"景点", "天气"}:
                continue
            query = keywords if destination in keywords else f"{destination} {keywords}"
            try:
                web_result = await self._web_search(query)
            except Exception as e:
                web_result = {
                    "query_type": "补充检索",
                    "query_success": False,
                    "results": {"error": str(e)},
                }
            supplemental_search.append(
                {
                    "request": request,
                    "query": query,
                    "result": web_result,
                    "verified": bool(web_result.get("verified")) if isinstance(web_result, dict) else False,
                    "trust_level": web_result.get("trust_level", "unknown") if isinstance(web_result, dict) else "unknown",
                }
            )

        search_bundle = {
            "planning": {
                "search_strategy": search_plan.get("search_strategy", ""),
                "demand_profile": search_plan.get("demand_profile", {}),
                "must_verify": search_plan.get("must_verify", []),
                "blocking_missing_fields": search_plan.get("blocking_missing_fields", []),
                "non_blocking_gaps": search_plan.get("non_blocking_gaps", []),
            },
            "transport": transport,
            "destination": {
                "name": destination,
                "geocodes": geocodes,
                "weather": weather,
                "pois": pois,
                "pois_by_category": pois_by_category,
                "selected_scenic_pois": selected_scenic,
                "nearby": nearby,
                "routes": routes,
                "distances": distances,
            },
            "quality": self._build_search_quality_report(
                event_data,
                transport,
                pois,
                weather,
                nearby,
                routes,
                search_plan,
                supplemental_search,
            ),
            "sources": [
                {"title": "Amap MCP", "source_type": "official_map", "trust_level": "high", "official": True},
                {"title": "12306 MCP", "source_type": "official_transport", "trust_level": "high", "official": True},
            ],
        }

        summary = self._format_trip_search_summary(search_bundle)

        return {
            "query_type": "行程相关信息查询",
            "query_success": True,
            "results": {
                "summary": summary,
                "destination": destination,
                "event_data": event_data,
                "search_plan": search_plan.get("tasks", []),
                "search_strategy": search_plan.get("search_strategy", ""),
                "demand_profile": search_plan.get("demand_profile", {}),
                "must_verify": search_plan.get("must_verify", []),
                "blocking_missing_fields": search_plan.get("blocking_missing_fields", []),
                "non_blocking_gaps": search_plan.get("non_blocking_gaps", []),
                "search_bundle": search_bundle,
                "transport": transport,
                "geocodes": geocodes,
                "pois": pois,
                "pois_by_category": pois_by_category,
                "nearby": nearby,
                "hotels": nearby.get("hotels", []),
                "restaurants": nearby.get("restaurants", []),
                "stations": nearby.get("stations", []),
                "routes": routes,
                "distances": distances,
                "weather": weather,
                "refinement_requests": refinement_requests or [],
                "search_keywords": [task.get("keywords") for task in poi_tasks],
                "supplemental_search": supplemental_search,
                "search_trust_policy": {
                    "hard_constraints_require_official_sources": True,
                    "unverified_hard_constraints_must_not_be_written_as_confirmed": True,
                },
            },
        }

    async def _execute_transport_tasks(self, search_plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        train_tasks = search_plan.get("tasks_by_type", {}).get("train", [])
        transport = {
            "outbound_trains": [],
            "return_trains": [],
            "queries": [],
            "errors": [],
        }
        if not train_tasks:
            return transport

        async def query_train_task(task: Dict[str, Any]) -> Dict[str, Any]:
            params = {
                "date": str(task.get("date") or ""),
                "from_station": str(task.get("from_station") or ""),
                "to_station": str(task.get("to_station") or ""),
            }
            earliest_start_time, latest_start_time = self._time_window_hours(task.get("time_window"))
            try:
                raw = await self._new_train_service().get_tickets(
                    date=params["date"],
                    from_station=params["from_station"],
                    to_station=params["to_station"],
                    train_filter_flags=str(task.get("train_filter_flags") or self._train_filter_flags(user_query or "")),
                    earliest_start_time=earliest_start_time,
                    latest_start_time=latest_start_time,
                    sort_flag="startTime",
                    sort_reverse=task.get("direction") == "return",
                    limited_num=80,
                    response_format="json",
                )
                tickets = self._normalize_train_tickets(raw)
                selected = self._select_time_fit_trains(tickets, task.get("time_window"))
                return {
                    "task": task,
                    "params": params,
                    "tickets": selected,
                    "raw_count": len(tickets),
                    "summary": self._format_train_ticket_summary(params, selected, raw),
                    "source": "12306 MCP get-tickets",
                }
            except Exception as e:
                logger.warning("Train task failed %s: %s", params, e)
                return {
                    "task": task,
                    "params": params,
                    "tickets": [],
                    "error": str(e),
                    "source": "12306 MCP get-tickets",
                }

        results = await asyncio.gather(*[query_train_task(task) for task in train_tasks])
        for item in results:
            direction = item.get("task", {}).get("direction")
            transport["queries"].append(item)
            if item.get("error"):
                transport["errors"].append(item)
            elif direction == "return":
                transport["return_trains"] = item.get("tickets", [])
            else:
                transport["outbound_trains"] = item.get("tickets", [])
        return transport

    def _time_window_hours(self, time_window: Any) -> tuple[int | None, int | None]:
        if not isinstance(time_window, list) or len(time_window) != 2:
            return None, None

        def parse_hour(value: Any, round_up: bool = False) -> int | None:
            match = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", str(value or "").strip())
            if not match:
                return None
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if round_up and minute > 0:
                hour += 1
            if 0 <= hour <= 24:
                return hour
            return None

        return parse_hour(time_window[0]), parse_hour(time_window[1], round_up=True)

    def _select_time_fit_trains(self, tickets: List[Dict[str, Any]], time_window: Any, limit: int = 6) -> List[Dict[str, Any]]:
        if not tickets:
            return []
        if not isinstance(time_window, list) or len(time_window) != 2:
            return tickets[:limit]
        start, end = str(time_window[0]), str(time_window[1])
        matched = [
            ticket
            for ticket in tickets
            if start <= str(ticket.get("start_time") or "") <= end
        ]
        return (matched or tickets)[:limit]

    def _select_scenic_pois(
        self,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category in ("景点", "博物馆", "公园"):
            selected.extend(pois_by_category.get(category, [])[:3])
        if not selected:
            selected = [poi for poi in pois if poi.get("name")][:8]
        seen = set()
        unique = []
        for poi in selected:
            key = poi.get("id") or poi.get("name")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(poi)
        return unique[:8]

    async def _query_nearby_trip_support(
        self,
        amap: AmapService,
        destination: str,
        selected_scenic: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        support = {
            "by_poi": [],
            "hotels": [],
            "restaurants": [],
            "stations": [],
        }

        async def search_near_poi(poi: Dict[str, Any]) -> Dict[str, Any]:
            name = str(poi.get("name") or "").strip()
            if not name:
                return {}
            try:
                hotels, restaurants, stations = await asyncio.gather(
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 经济型酒店"),
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 餐厅 美食"),
                    amap.maps_text_search(city=destination, keywords=f"{name} 附近 地铁站 火车站"),
                )
                return {
                    "poi": self._compact_poi(poi),
                    "hotels": hotels[:5],
                    "restaurants": restaurants[:5],
                    "stations": stations[:5],
                }
            except Exception as e:
                logger.warning("Nearby query failed for %s: %s", name, e)
                return {"poi": self._compact_poi(poi), "error": str(e)}

        nearby_items = await asyncio.gather(*[search_near_poi(poi) for poi in selected_scenic[:5]])
        seen = {"hotels": set(), "restaurants": set(), "stations": set()}
        for item in nearby_items:
            if not item:
                continue
            support["by_poi"].append(item)
            for key in ("hotels", "restaurants", "stations"):
                for poi in item.get(key, []) or []:
                    poi_key = poi.get("id") or poi.get("name")
                    if not poi_key or poi_key in seen[key]:
                        continue
                    seen[key].add(poi_key)
                    support[key].append(poi)
        support["hotels"] = support["hotels"][:15]
        support["restaurants"] = support["restaurants"][:15]
        support["stations"] = support["stations"][:15]
        return support

    async def _query_trip_routes(
        self,
        amap: AmapService,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        routes: List[Dict[str, Any]] = []
        distances: List[Dict[str, Any]] = []
        route_candidates = self._select_route_candidates(pois_by_category, pois)
        if len(route_candidates) < 2:
            return routes, distances

        route_pairs = list(zip(route_candidates, route_candidates[1:]))[:6]

        async def query_route_pair(pair: tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
            start, end = pair
            origin = self._location_str(start)
            dest = self._location_str(end)
            if not origin or not dest:
                return {}
            route_item = {
                "from": start.get("name"),
                "to": end.get("name"),
                "from_location": origin,
                "to_location": dest,
            }
            try:
                walking, driving = await asyncio.gather(
                    amap.maps_direction_walking(origin=origin, destination=dest),
                    amap.maps_direction_driving(origin=origin, destination=dest),
                )
                route_item["walking"] = walking
                route_item["driving"] = driving
                route_item["recommended_modes"] = ["walking", "driving"]
                return route_item
            except Exception as e:
                logger.warning("Amap route query failed for %s -> %s: %s", start.get("name"), end.get("name"), e)
                route_item["error"] = str(e)
                return route_item

        routes = [item for item in await asyncio.gather(*[query_route_pair(pair) for pair in route_pairs]) if item]

        destination_point = self._location_str(route_candidates[0])
        origin_points = [self._location_str(item) for item in route_candidates[1:8]]
        origin_points = [item for item in origin_points if item]
        if destination_point and origin_points:
            try:
                distance_raw = await amap.maps_distance(
                    origins="|".join(origin_points),
                    destination=destination_point,
                    type="1",
                )
                distances.append(
                    {
                        "to": route_candidates[0].get("name"),
                        "from": [item.get("name") for item in route_candidates[1:8]],
                        "mode": "driving_distance",
                        "raw": distance_raw,
                    }
                )
            except Exception as e:
                logger.warning("Amap distance query failed for %s: %s", route_candidates[0].get("name"), e)
        return routes, distances

    def _compact_poi(self, poi: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "type": poi.get("type"),
            "location": poi.get("location"),
            "source_keywords": poi.get("source_keywords"),
        }

    def _build_search_quality_report(
        self,
        event_data: Dict[str, Any],
        transport: Dict[str, Any],
        pois: List[Dict[str, Any]],
        weather: Any,
        nearby: Dict[str, Any],
        routes: List[Dict[str, Any]],
        search_plan: Dict[str, Any] | None = None,
        supplemental_search: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        search_plan = search_plan or {}
        supplemental_search = supplemental_search or []
        missing = []
        warnings = []
        verified = []
        if event_data.get("origin") and event_data.get("destination") and event_data.get("start_date"):
            if transport.get("outbound_trains"):
                verified.append("outbound_train_options")
            else:
                missing.append("outbound_train_options")
        has_return_query = any(
            item.get("task", {}).get("direction") == "return"
            for item in transport.get("queries", [])
        )
        if has_return_query or (event_data.get("end_date") and event_data.get("return_location")):
            if transport.get("return_trains"):
                verified.append("return_train_options")
            else:
                missing.append("return_train_options")
        if pois:
            verified.append("destination_pois")
        else:
            missing.append("destination_pois")
        if isinstance(weather, dict) and weather.get("error"):
            warnings.append(f"weather_failed: {weather.get('error')}")
        elif weather:
            verified.append("weather")
        else:
            missing.append("weather")
        if nearby.get("hotels"):
            verified.append("nearby_hotels")
        else:
            missing.append("nearby_hotels")
        if nearby.get("restaurants"):
            verified.append("nearby_restaurants")
        if routes:
            verified.append("poi_routes")
        else:
            missing.append("poi_routes")
        for error in transport.get("errors", []):
            warnings.append(f"train_query_failed: {error.get('params')} {error.get('error')}")

        verification_aliases = set(verified)
        if transport.get("outbound_trains") or transport.get("return_trains"):
            verification_aliases.update({"train_tickets", "transport_options"})
        if nearby.get("hotels"):
            verification_aliases.update({"lodging_candidates", "lodging_area"})
        if routes:
            verification_aliases.update({"routes", "route_distance"})
        if any(item.get("verified") for item in supplemental_search):
            verification_aliases.update({"official_attraction_policy", "opening_hours", "ticket_or_reservation_policy"})

        planned_must_verify = [
            str(item)
            for item in search_plan.get("must_verify", [])
            if str(item) not in verification_aliases
        ]
        blocking_missing = [
            str(item)
            for item in search_plan.get("blocking_missing_fields", [])
            if event_data.get(str(item)) in (None, "", [])
        ]
        non_blocking_gaps = [
            str(item)
            for item in search_plan.get("non_blocking_gaps", [])
            if item
        ]
        for item in planned_must_verify:
            if item not in missing:
                missing.append(item)
        for item in blocking_missing:
            if item not in missing:
                missing.append(item)

        return {
            "verified_fields": verified,
            "missing": missing,
            "warnings": warnings,
            "must_verify": search_plan.get("must_verify", []),
            "unverified_must_verify": planned_must_verify,
            "blocking_missing_fields": blocking_missing,
            "non_blocking_gaps": non_blocking_gaps,
            "hard_constraints_require_official_sources": True,
        }

    def _format_trip_search_summary(self, bundle: Dict[str, Any]) -> str:
        destination = bundle.get("destination", {}).get("name", "")
        transport = bundle.get("transport", {})
        dest = bundle.get("destination", {})
        parts = [f"已完成 {destination} 行程检索素材收集。"]
        if transport.get("outbound_trains"):
            parts.append(f"去程候选高铁/火车 {len(transport['outbound_trains'])} 条。")
        if transport.get("return_trains"):
            parts.append(f"返程候选高铁/火车 {len(transport['return_trains'])} 条。")
        if dest.get("pois"):
            parts.append(f"目的地 POI {len(dest['pois'])} 个，含景点、餐饮、住宿和交通枢纽。")
        nearby = dest.get("nearby") or {}
        if nearby.get("hotels"):
            parts.append(f"景点周边住宿候选 {len(nearby['hotels'])} 个。")
        if nearby.get("restaurants"):
            parts.append(f"景点周边餐饮候选 {len(nearby['restaurants'])} 个。")
        if dest.get("routes"):
            parts.append(f"景点间路线/交通方式 {len(dest['routes'])} 组。")
        missing = bundle.get("quality", {}).get("missing") or []
        if missing:
            parts.append("仍缺少：" + "、".join(missing) + "。")
        return "".join(parts)

    def _select_route_candidates(
        self,
        pois_by_category: Dict[str, List[Dict[str, Any]]],
        pois: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for category in ("火车站 高铁站", "机场", "景点", "博物馆", "公园", "美食 餐厅", "经济型酒店"):
            for poi in pois_by_category.get(category, [])[:2]:
                if poi.get("name") and self._location_str(poi):
                    selected.append(poi)
        if not selected:
            selected = [poi for poi in pois if poi.get("name") and self._location_str(poi)]
        seen = set()
        unique = []
        for poi in selected:
            key = poi.get("id") or poi.get("name")
            if key in seen:
                continue
            seen.add(key)
            unique.append(poi)
        return unique[:8]

    def _location_str(self, poi: Dict[str, Any]) -> str:
        loc = poi.get("location") if isinstance(poi, dict) else None
        if not isinstance(loc, dict):
            return ""
        lng = loc.get("longitude")
        lat = loc.get("latitude")
        if lng is None or lat is None:
            return ""
        return f"{lng},{lat}"

