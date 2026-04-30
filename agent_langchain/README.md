# TravelFlow 旅游出行助手

TravelFlow 是一个基于 Python / FastAPI / LangChain / LangGraph 的多智能体旅游规划系统。系统由面向用户的主智能体统一承接对话，按需进行意图识别、直接回复或委托业务智能体，并使用 LangGraph 共享 state 在信息检索、行程规划、事项收集等智能体之间传递信息，为用户生成个性化旅行方案。MainAgent 驱动记忆读写时机，MemoryManager 负责状态一致性、合并和持久化。

## 技术框架

### 标准化智能体架构

TravelFlow 对外采用 4 个核心智能体角色：1 个主智能体和 3 个规划业务智能体。意图识别是主智能体的内置能力，记忆系统是由 MainAgent 驱动的基础设施服务，编排调度是主智能体调用的内部工具，不再作为独立智能体身份对外暴露。

| 身份 | 代码名/调度名 | 职责 |
| --- | --- | --- |
| 主智能体 Main Agent | `MainAgent` | 唯一面向用户的智能体；理解用户意图、直接回复普通对话、按需调用业务智能体并汇总结果 |
| 信息检索智能体 Search Agent | `search` | 通过高德开放平台 API 获取 POI、天气、地理编码、路线等外部数据 |
| 行程规划智能体 Plan Agent | `plan` | 基于用户偏好、行程要素和检索数据生成旅行计划 |
| 事项收集智能体 Clarification Agent | `clarification` | 提取目的地、日期、预算、同行人等字段，并指出缺失信息 |

兼容说明：`memory` 只作为旧接口别名保留。新的显式记忆查询或长期偏好更新由 `IntentRecognition.direct_action` 表示，并由 MainAgent 直接驱动 MemoryManager / PreferenceAgent；规划链路不会把 `memory` 放进 `agent_schedule`。

主智能体内部组件：

| 组件 | 代码名 | 身份 |
| --- | --- | --- |
| Workflow Skill 路由 | `SkillRouter` | MainAgent 内部前置路由；常见业务场景直接生成标准 `intention_data` |
| 意图识别能力 | `IntentRecognition` | MainAgent 内部能力；理解用户需求，生成业务 `agent_schedule` 或内部 `direct_action` |
| 业务编排工具 | `AgentScheduler` | MainAgent 内部工具；按 priority 分层执行，同层并行、跨层串行 |

兼容说明：旧类名 `IntentionAgent` 和 `OrchestrationAgent` 仍保留为别名，分别指向 `IntentRecognition` 和 `AgentScheduler`。

行程规划调度优先级：

1. `clarification`：先收集目的地、出发地、日期、天数、预算、节奏等行程字段。
2. `search`：补全 POI、天气、路线等外部信息。
3. `plan`：整合事项字段、外部检索结果和 MainAgent 注入的记忆上下文生成最终行程。

智能体之间不再传递自定义消息对象，统一使用 LangGraph state：

- `messages`：LangChain message/dict 消息列表
- `intention_data`：`IntentRecognition` 生成的意图与调度计划
- `context`：跨智能体共享上下文
- `previous_results`：已完成智能体的结构化结果
- `results`：调度器聚合后的执行结果
- `final_result`：最终返回给 CLI/Web 的结构化响应

### Workflow Skills 模块化设计

MainAgent 先通过 `agents.workflow_skills.SkillRouter` 匹配高频业务 workflow。这里的 Skill 是业务 workflow 定义，不承载子智能体实现；它封装一套可复用 `workflow_plan`：执行模式、子智能体组合、工具清单、记忆依赖、数据流、步骤依赖、阻断策略和记忆策略。命中后不会再调用 LLM 意图识别，而是由 workflow skill 直接产出和 `IntentRecognition` 兼容的 `intention_data`，并把 `agent_schedule` 作为调度器兼容字段派生出来。当前内置常见 workflow：

- `travel_planning`：行程规划，固定调度 `clarification -> search -> plan`。
- `weather_query`：天气/气温/预报查询，调度 `search`。
- `train_query`：火车/高铁/车次/票价/余票查询，调度 `search`。
- `information_query`：景点、攻略、门票、开放时间、路线等通用信息查询，调度 `search`。
- `memory_profile`：显式个人偏好、历史或身份记忆查询/更新，产出 `direct_action`，由 MainAgent 驱动 MemoryManager。
- `pending_plan_completion`：用户补充上一轮被阻断的行程字段时，恢复规划工作流。

未命中 workflow skill 的输入会继续走 `IntentRecognition`，保留开放问题和长尾表达的兜底能力。

每个 workflow skill 必须显式写出：

- `child_agents`：本场景组合哪些业务子智能体，例如 `clarification/search/plan`。
- `tools`：子智能体会用到哪些工具，例如 `AmapService`、`TrainService`、`DDGS`、`MemoryManager`。
- `memory_dependencies`：读取或写入哪些记忆层，例如 `LongTermMemory`、`ShortTermMemory`。
- `data_flow`：用户输入、事项字段、检索结果、记忆上下文和最终输出如何流转。
- `steps`：具体执行顺序、优先级、依赖和输出。

运行代码按职责放在正式 Python 包中，`.claude/skills/` 只保留能力说明与旧路径兼容入口。`LazyAgentRegistry` 优先从正式模块装配智能体：

- `query-info` 文档 → `agents.search_agent.InformationQueryAgent` → `search`；内部能力拆在 `agents.search_modules`
- `plan-trip` 文档 → `agents.plan_agent.ItineraryPlanningAgent` → `plan`
- `clarification` 子智能体实现位于 `agents.clarification_agent.EventCollectionAgent`；它不是用户可见 Skill，不放入 `.claude/skills`
- `memory-query` 弱化为 `context.memory_query.MemoryQueryAgent` 兼容能力；主链路优先使用 `MainAgent direct_action + MemoryManager`
- `preference` → `agents.preference_agent.PreferenceAgent`，由 `memory_profile` 或 `MemoryAgent` 间接使用

## 记忆系统

记忆系统按职责分层，MainAgent 负责驱动读写时机，MemoryManager 负责状态一致性、合并和持久化：

- Transcript Log：每轮 user / assistant 消息都会保存到长期聊天历史，用于审计、回放、会话恢复和摘要，不直接等同于长期偏好。
- Working Memory：Redis 或进程内缓存当前会话/当前任务状态，包括 `recent_dialogue`、`pending_plan` 和通用 `working_state`。
- User Profile Memory：长期用户画像，保存稳定偏好。`get_preference()` 仍返回简单 `{type: value}`；`get_preference_records()` 返回带 `source/confidence/status/scope/version/history` 的可审计记录。
- Trip History：规划成功后记录历史行程事件，不直接覆盖用户长期偏好。
- Behavior Feedback：保存用户行为反馈，供后续检索和画像候选判断使用。

当前存储后端：

- 短期和工作记忆：优先 Redis，未配置时回退进程内缓存。
- 长期记忆：优先 PostgreSQL，未配置时回退 `data/memory/*.json`。
- 开发环境未配置 PostgreSQL / Redis 时，会自动回退到 `data/memory/*.json`，便于本地调试。

配置项在 `config.py`：

- `POSTGRES_DSN`
- `REDIS_URL`
- `MEMORY_CACHE_TTL_SEC`
- `ALLOW_JSON_FALLBACK`

## LLM 配置

主链路通过 `utils/langchain_runtime.py` 创建 OpenAI-compatible `ChatOpenAI` 实例。当前项目优先使用小米 MiMo V2.5 Pro：

- `MIMO_API_KEY`：小米 MiMo API Key。`tp-` Token Plan key 默认使用中国区 `https://token-plan-cn.xiaomimimo.com/v1`。
- `LLM_MODEL_NAME`：默认 `mimo-v2.5-pro`。
- `LLM_BASE_URL`：可选覆盖默认地址；普通 `sk-` key 默认使用 `https://api.xiaomimimo.com/v1`。
- `DEEPSEEK_API_KEY`：仅在未配置 `MIMO_API_KEY` 时作为旧环境 fallback。

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

## Search Agent 检索编排

行程规划场景下，`search` 不再只是按关键词分支查询，而是先根据 `event_data` 生成 `search_plan`，再调用 12306 MCP、高德 MCP 和必要的网页兜底检索，最终返回结构化 `search_bundle`。

`search_bundle` 主要包含：

- `transport.outbound_trains` / `transport.return_trains`：按出发日期和返程日期查询到的合适时间高铁/火车候选、价格和余票。
- `destination.pois` / `pois_by_category`：目的地景点、博物馆、公园、餐饮、经济型酒店、火车站/高铁站、机场等 POI。
- `destination.nearby`：围绕核心景点查询的周边住宿、餐饮和交通站点。
- `destination.routes` / `distances`：核心景点之间的步行、驾车路线和距离。
- `quality`：已核验字段、缺失字段和失败警告，供 `plan` 智能体避免编造硬约束。

为了兼容既有规划链路，`results` 顶层仍保留 `pois`、`weather`、`routes`、`distances` 等旧字段。

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
