# TravelFlow 旅游出行助手

TravelFlow 是一个基于 Python / FastAPI / LangChain / LangGraph 的多智能体旅游规划系统。系统由面向用户的主智能体统一承接对话，按需进行意图识别、直接回复或委托业务智能体，并使用 LangGraph 共享 state 在信息检索、行程规划、事项收集、记忆与偏好管理等智能体之间传递信息，为用户生成个性化旅行方案。

## 技术框架

### 标准化智能体架构

TravelFlow 对外采用 5 个智能体角色：1 个主智能体和 4 个业务智能体。意图识别是主智能体的内置能力，编排调度是主智能体调用的内部工具，不再作为独立智能体身份对外暴露。

| 身份 | 代码名/调度名 | 职责 |
| --- | --- | --- |
| 主智能体 Main Agent | `MainAgent` | 唯一面向用户的智能体；理解用户意图、直接回复普通对话、按需调用业务智能体并汇总结果 |
| 信息检索智能体 Search Agent | `search` | 通过高德开放平台 API 获取 POI、天气、地理编码、路线等外部数据 |
| 行程规划智能体 Plan Agent | `plan` | 基于用户偏好、行程要素和检索数据生成旅行计划 |
| 事项收集智能体 Clarification Agent | `clarification` | 提取目的地、日期、预算、同行人等字段，并指出缺失信息 |
| 记忆与偏好智能体 Memory Agent | `memory` | 查询和更新长期偏好、历史行程、行为反馈、历史会话 |

主智能体内部组件：

| 组件 | 代码名 | 身份 |
| --- | --- | --- |
| 意图识别能力 | `IntentRecognition` | MainAgent 内部能力；理解用户需求，生成 `agent_schedule` |
| 业务编排工具 | `AgentScheduler` | MainAgent 内部工具；按 priority 分层执行，同层并行、跨层串行 |

兼容说明：旧类名 `IntentionAgent` 和 `OrchestrationAgent` 仍保留为别名，分别指向 `IntentRecognition` 和 `AgentScheduler`。

行程规划调度优先级：

1. `clarification`：先收集目的地、出发地、日期、天数、预算、节奏等行程字段。
2. `search` / `memory`：同层并行补全外部信息和内部记忆。
3. `plan`：整合前序结果生成最终行程。

智能体之间不再传递自定义消息对象，统一使用 LangGraph state：

- `messages`：LangChain message/dict 消息列表
- `intention_data`：`IntentRecognition` 生成的意图与调度计划
- `context`：跨智能体共享上下文
- `previous_results`：已完成智能体的结构化结果
- `results`：调度器聚合后的执行结果
- `final_result`：最终返回给 CLI/Web 的结构化响应

### Skills 模块化设计

底层能力仍保留在 `.claude/skills/` 中，`LazyAgentRegistry` 会懒加载需要的 Skill。对外调度名已统一为 TravelFlow 的业务智能体模型：

- `query-info` → `search`
- `plan-trip` → `plan`
- `event-collection` → `clarification`
- `memory-query` / `preference` → `memory`

## 记忆系统

记忆分为短期记忆和长期记忆：

- 短期记忆：Redis 缓存当前会话上下文，包括目的地、预算、出发地、同行人、缺失字段等。
- 长期记忆：PostgreSQL 存储用户偏好、历史会话、行为反馈、历史行程。
- 开发环境未配置 PostgreSQL / Redis 时，会自动回退到 `data/memory/*.json`，便于本地调试。

配置项在 `config.py`：

- `POSTGRES_DSN`
- `REDIS_URL`
- `MEMORY_CACHE_TTL_SEC`
- `ALLOW_JSON_FALLBACK`

## 高德 API 能力

`utils/amap_service.py` 封装了高德开放平台接口，并提供与高德 MCP 常见工具同名的方法：

- `maps_geo`
- `maps_regeocode`
- `maps_ip_location`
- `maps_weather`
- `maps_direction_driving`
- `maps_direction_walking`
- `maps_bicycling`
- `maps_distance`
- `maps_text_search`
- `maps_around_search`
- `maps_search_detail`

配置 `AMAP_MAPS_API_KEY` 或 `AMAP_API_KEY` 环境变量即可替换默认 Key。

## 12306 MCP 能力

`utils/train_service.py` 通过 `npx -y 12306-mcp` 调用 12306 MCP 的 `get-tickets` 工具，用于火车/高铁车次查询。
用户询问火车、高铁、动车、车次、票价或余票时，`IntentRecognition` 会调度 `search` 智能体优先走 12306 MCP，返回车次、出发/到达站、出发/到达时间、历时、余票和票价等结构化结果。
运行该能力需要本机可用的 Node.js / npm / npx。

可选环境变量：

- `TRAIN_MCP_COMMAND`
- `TRAIN_MCP_ARGS`

## 流式响应与过程可视化

FastAPI 提供两个聊天接口：

- `POST /api/chat`：普通 JSON 响应。
- `POST /api/chat/stream`：NDJSON 流式响应，返回 Agent 加载、调度和执行轨迹，用于前端展示推理过程。

浏览器入口：

```bash
uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
```

然后访问 `http://127.0.0.1:8000`。

## 异步任务

项目依赖中包含 Celery，适合承载长期记忆总结、偏好抽取、历史会话归档等后台任务。当前主链路先保持同步可运行，后续可将 `MemoryManager.get_long_term_summary_async` 等耗时任务迁移到 Celery worker。

## 本地运行

要求 Python 3.10+。项目依赖中的 MCP SDK 不支持 Python 3.9。

安装依赖：

```bash
pip install -r requirements.txt
```

如需本地固定安装 12306 MCP（避免每次通过 npx 临时下载）：

```bash
npm install --prefix .mcp-node 12306-mcp
```

CLI：

```bash
python cli.py
```

Web：

```bash
uvicorn web.app:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```bash
python cli.py health
```
