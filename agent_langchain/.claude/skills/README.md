# TravelFlow Skills

TravelFlow 对外采用五智能体架构，Skill 目录只作为可复用能力模块保留，由 `LazyAgentRegistry` 懒加载。

| Skill 目录 | TravelFlow 调度名 | 用途 |
| --- | --- | --- |
| `query-info` | `search` | 高德 POI、天气、地理编码、路线，以及通用查询兜底 |
| `plan-trip` | `plan` | 生成结构化旅行计划 |
| `event-collection` | `clarification` | 收集目的地、日期、预算、同行人等字段 |
| `memory-query` | `memory` | 查询历史行程、历史会话和已保存偏好 |
| `preference` | `memory` | 抽取并更新长期用户偏好 |

主调度智能体不直接暴露为 Skill。调度顺序通常为：

1. `memory` / `clarification`
2. `search`
3. `plan`

旧目录名仅作为 Skill 文件夹名保留；运行时统一通过 LangChain/LangGraph 共享 state 调度。
