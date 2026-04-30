# TravelFlow Skills

TravelFlow 对外采用五智能体架构，`.claude/skills` 目录只保留能力说明、Prompt 指南和旧路径兼容入口；生产运行代码放在 `agents/`、`context/` 和 `utils/`。

| Skill 目录 | TravelFlow 调度名 | 用途 |
| --- | --- | --- |
| `query-info` | `search` | 文档对应 `agents.search_agent.InformationQueryAgent` |
| `plan-trip` | `plan` | 文档对应 `agents.plan_agent.ItineraryPlanningAgent` |
| `event-collection` | `clarification` | 文档对应 `agents.clarification_agent.EventCollectionAgent` |
| `memory-query` | `memory` | 弱化兼容能力，对应 `context.memory_query.MemoryQueryAgent` |
| `preference` | `memory` | 偏好抽取能力，对应 `agents.preference_agent.PreferenceAgent` |

主调度智能体不直接暴露为 Skill。调度顺序通常为：

1. `memory` / `clarification`
2. `search`
3. `plan`

旧目录名仅作为 Skill 文件夹名和兼容导入路径保留；运行时统一通过 LangChain/LangGraph 共享 state 调度。
