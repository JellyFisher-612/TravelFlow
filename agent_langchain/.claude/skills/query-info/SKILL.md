---
name: query-info
description: TravelFlow Search Agent skill. Use it when the system needs external travel data such as POI, weather, geocoding, reverse geocoding, or route planning. It maps to the canonical agent name `search`.
---

# Search Agent Skill

该 Skill 是 TravelFlow 信息检索智能体的底层实现，对外调度名为 `search`。

## 能力

1. 高德 POI 搜索：查询景点、餐饮、酒店等地点信息。
2. 高德天气查询：根据目的地获取实时天气或预报。
3. 高德地理编码：地址转经纬度。
4. 高德逆地理编码：经纬度转地址。
5. 高德路径规划：支持步行、驾车、公交路线。
6. 12306 MCP 火车/高铁查询：查询车次、出发到达时间、余票和票价等铁路出行信息。
7. 通用文本搜索兜底：当问题不是结构化旅游数据时使用。

## 输入

通常由 Orchestrator 传入：

```json
{
  "context": {
    "rewritten_query": "用户想从上海出发去杭州玩三天",
    "event_data": {
      "origin": "上海",
      "destination": "杭州",
      "start_date": "2026-05-01",
      "duration_days": 3
    }
  },
  "previous_results": []
}
```

## 输出

返回结构化检索结果，常见字段：

- `results.destination`
- `results.event_data`
- `results.geocodes`
- `results.pois`
- `results.routes`
- `results.weather`
- `results.tickets`
- `results.summary`

## 规划约束

行程规划时优先使用 `pois` 和 `weather` 中的真实数据，不要凭空编造景点、天气或路线。

## 行程检索设计

行程规划场景下，Search Agent 会先生成 `search_plan`，再调用垂直工具，最后返回 `search_bundle`：

- `transport.outbound_trains` / `transport.return_trains`：12306 MCP 查询出的去程和返程车次、时间、余票和价格。
- `destination.pois` / `pois_by_category`：高德 MCP 查询出的景点、餐饮、住宿、交通枢纽等 POI。
- `destination.nearby`：围绕核心景点查询周边酒店、餐饮和站点。
- `destination.routes` / `distances`：核心 POI 之间的步行、驾车路线和距离。
- `quality.verified_fields` / `missing` / `warnings`：说明哪些硬约束已核验，哪些仍缺失或失败。

为了兼容旧版 Plan Agent，Search Agent 仍会在 `results` 顶层保留 `pois`、`weather`、`routes`、`distances` 等字段。
