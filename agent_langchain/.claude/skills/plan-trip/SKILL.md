---
name: plan-trip
description: TravelFlow Plan Agent skill. Use it to generate personalized trip itineraries from user preferences, clarified trip fields, and Search Agent data. It maps to the canonical agent name `plan`.
---

# Plan Agent Capability

该文件只描述 TravelFlow 行程规划能力。运行实现位于 `agents.plan_agent.ItineraryPlanningAgent`，对外调度名为 `plan`；旧路径 `plan-trip/script/agent.py` 仅做兼容导入。

## 输入依赖

Plan Agent 应在以下信息之后执行：

1. `memory`：用户长期偏好、历史行程、行为反馈。
2. `clarification`：目的地、出发地、日期、预算、同行人、行程天数等。
3. `search`：高德 POI、天气、地理编码、路线等真实外部数据。

## 输出结构

返回结构化 JSON：

- `itinerary.title`
- `itinerary.duration`
- `itinerary.daily_plans`
- `itinerary.hard_constraints`
- `itinerary.fallback_options`
- `itinerary.notes`
- `itinerary.estimated_budget`
- `verification_summary`
- `planning_complete`
- `summary`

## 规划约束

1. 优先使用 `search.results.pois` 中的真实地点安排活动。
2. 优先参考 `search.results.weather` 做室内/室外搭配。
3. 若路线数据存在，使用 `search.results.routes` 生成景点间交通建议。
4. 结合 `memory.preferences` 中的预算、节奏、住宿、交通、活动偏好。
5. 若关键信息不足，在 `summary` 或 `notes` 中明确指出需要用户补充。
6. 不得编造高铁/航班车次、票价、余票、酒店门店、预约状态。没有官方/实时来源时，必须标注为待核验。
7. 输出必须分清三层：必须确认的硬约束、可调整的游玩动线、约不到/买不到时的备选方案。
8. 经济型预算必须给出粗预算区间，并明确哪些费用尚未核验。
