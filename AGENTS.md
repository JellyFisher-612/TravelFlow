# TravelFlow Codex 项目说明

## 项目定位

当前仓库是 `agent_langchain/`，实现一个 Python FastAPI / LangChain / LangGraph 多智能体旅游规划系统。

主链路：

1. 用户消息从 CLI 或 Web 进入。
2. `MainAgent` 调用 `IntentRecognition` 做意图识别。
3. `AgentScheduler` 按 priority 分批调度业务智能体。
4. 业务智能体调用 `agents/`、`context/` 和 `utils/` 下的正式实现；`.claude/skills/` 只保留能力说明与旧路径兼容入口。
5. Web 端流式返回执行轨迹、文本 delta、建议回复和输入请求。

## 核心文件

- `agent_langchain/cli.py`：CLI 与 Web 共用的查询执行入口，负责初始化模型、记忆、主智能体、调度器和结果渲染。
- `agent_langchain/web/app.py`：FastAPI 应用，提供首页、普通聊天、NDJSON 流式聊天和会话历史接口。
- `agent_langchain/agents/main_agent.py`：面向用户的一轮对话入口，只负责意图识别和委托调度。
- `agent_langchain/agents/intention_agent.py`：意图识别、直接回复、规则兜底、多轮补充和调度计划生成。
- `agent_langchain/agents/orchestration_agent.py`：`AgentScheduler`，使用 LangGraph state 按优先级串并行执行业务智能体，并负责阻断和聚合。
- `agent_langchain/agents/travelflow_agents.py`：标准业务智能体适配层，把 `search/plan/clarification/memory` 映射到正式 Python 实现或偏好逻辑。
- `agent_langchain/agents/lazy_agent_registry.py`：懒加载智能体注册器，优先加载正式适配类，兼容旧 Skill 名。
- `agent_langchain/agents/search_agent.py`：信息检索智能体门面，对外保持 `search` 调度名，内部委托 `agents/search_modules/`。
- `agent_langchain/agents/search_modules/`：搜索内部模块，按铁路、天气、网页兜底、行程检索规划和执行拆分。
- `agent_langchain/agents/clarification_agent.py`：事项收集智能体，抽取出发地、目的地、日期、天数、预算、节奏等字段。
- `agent_langchain/agents/plan_agent.py`：行程规划智能体，整合事项、检索和记忆生成计划。
- `agent_langchain/context/memory_query.py`：旧记忆查询兼容能力，主链路已弱化为 `MainAgent + MemoryManager`。
- `agent_langchain/context/`：短期记忆、长期记忆和记忆管理器。
- `agent_langchain/utils/`：LLM runtime、重试/熔断、高德服务、12306 服务、JSON 解析、LangSmith 配置等。

## 智能体架构

对外保持 5 类角色：

- `MainAgent`：唯一面向用户的主智能体。
- `search`：信息检索，负责高德 POI/天气/路线、12306 车次、必要网页搜索。
- `plan`：行程规划，生成结构化旅行方案。
- `clarification`：事项收集，提取或追问行程关键字段。
- `memory`：记忆与偏好，查询或更新长期偏好、历史行程、行为反馈。

内部组件：

- `IntentRecognition` 不是业务智能体，只是 `MainAgent` 的意图识别能力。
- `AgentScheduler` 不是业务智能体，只是 `MainAgent` 使用的编排工具。
- 旧名 `IntentionAgent`、`OrchestrationAgent` 仍作为兼容别名存在。

规划类调度应保持：

1. P1 `clarification`
2. P2 `search` 与 `memory` 并行
3. P3 `plan`

## 状态与阻断规则

智能体之间通过 LangGraph state 传递数据，不再传自定义消息对象。常用字段：

- `messages`：当前对话上下文消息。
- `intention_data`：意图识别结果，包含 `intents`、`key_entities`、`rewritten_query`、`agent_schedule`。
- `context`：跨智能体共享上下文。
- `previous_results` / `results`：已执行智能体结果。
- `final_result`：最终给 CLI/Web 的结构化结果。

调度器会在以下场景阻断后续规划：

- `clarification` 缺少关键字段：`destination`、`start_date`、`duration_days`、`budget_level`、`pace_preference`。
- `search` 查询失败且后面还有 `plan`。
- `memory` 发现预算或节奏等阻断性偏好缺口。

`plan` 最多可追加一轮 `search -> plan` 补充检索，避免无限循环。

## 记忆系统

- 短期记忆：`ShortTermMemory`，保存当前会话最近对话和 pending plan。
- 长期记忆：`LongTermMemory`，保存偏好、聊天历史、行为反馈、历史行程。
- `MemoryManager.add_message()` 会同时写短期和长期记忆。
- 未配置 Redis/PostgreSQL 时，开发环境走本地 JSON fallback。
- Web 会话恢复时会从长期历史回填短期上下文，助手消息优先使用 metadata 中的可读 `display`。

## Web 与本地运行

Web 入口需要在内层目录启动：

```bash
cd agent_langchain
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

访问：

```text
http://127.0.0.1:8000
```

接口：

- `POST /api/chat`：普通 JSON 响应。
- `POST /api/chat/stream`：NDJSON 流式响应。
- `GET /api/sessions`、`GET /api/sessions/{session_id}`、`DELETE /api/sessions/{session_id}`：会话历史管理。

## LLM 与外部服务

- LLM 默认配置来自 `agent_langchain/config.py` 和 `.env`：`DEEPSEEK_API_KEY`、`LLM_MODEL_NAME`、`LLM_BASE_URL`。
- `utils/langchain_runtime.py` 创建 `ChatOpenAI`，显式使用 `httpx.Client/AsyncClient(trust_env=False)`，LLM 调用不要继承系统代理。
- 高德 API 使用 `AMAP_MAPS_API_KEY` 或 `AMAP_API_KEY`。
- 12306 查询通过 `utils/train_service.py` 调用 `npx -y 12306-mcp`，需要本机 Node.js/npm/npx。
- LangSmith 默认关闭，只在环境变量启用且有 key 时初始化。

## 已知行为与风险点

- 普通问候、身份、能力、感谢、确认等应直接回复，不应加载 `search`。
- LLM 不可用时，意图识别应尽量走规则兜底，不能把明显的预算/偏好/本次行程补充误判成外部搜索。
- “本次行程预算/住宿/餐饮/交通/节奏”是当前行程约束，优先走规划链路；“以后/长期/记住/我的偏好”才偏向长期记忆。
- 住宿预算范围要保留上下限，例如 `300到600元` 应使用 `lodging_budget_per_night_min/max`，不要压成单值。
- `clarification_agent` 对常见字段应优先规则抽取，避免因为 LLM 空输出或非 JSON 导致事项收集失败。
- 硬约束不能编造：车次、票价、余票、酒店库存、门票、预约状态、开放时间等必须来自可靠来源或标记未核验。
- 外部检索失败时应阻断完整规划，避免生成看似确定但无来源的方案。

## 测试规则

自动化测试不得调用真实外部服务。必须 mock 或 fake：

- LLM
- 高德 MCP/API
- 12306 MCP
- DDGS / 网页搜索
- Redis
- PostgreSQL / 数据库

优先使用最窄验证：

```bash
python3 -m py_compile <changed python files>
PYTHONPATH=agent_langchain python3 -m unittest <test module> -v
PYTHONPATH=agent_langchain python3 -m unittest discover -s tests -v
```

项目文档偏好 pytest 命令：

```bash
PYTHONPATH=agent_langchain pytest <test path>
PYTHONPATH=agent_langchain pytest
```

但当前环境可能没有安装 pytest；如果缺失，使用 stdlib `unittest` 并在进度记录中说明。

## Codex 测试改进循环

在 Codex test improvement loop 内工作时，改代码前先读：

- `测试需求文档.md`
- `.codex-loop/progress.md`
- `.codex-loop/learnings.md`

循环规则：

- 每轮选择一个连贯、未完成的改进点。
- 优先补测试和 fixture，再改生产代码。
- 生产行为变化必须在同轮新增或更新聚焦测试。
- 保留无关用户改动，不要清理 `.env`、API key、本地记忆 JSON、运行日志、缓存文件。

## 当前测试覆盖重点

- `tests/test_intent_direct_answer.py`：直接回复、LLM 失败兜底、本次行程预算约束识别。
- `tests/test_event_collection_rules.py`：事项收集规则抽取，尤其预算范围不调用 LLM。
- `tests/test_langchain_runtime.py`：LLM HTTP client 不继承环境代理。
- `tests/test_planning_dispatch.py`：规划调度层级、缺字段阻断。
- `tests/test_search_agent_routing.py`：天气、高铁、规划检索 bundle、硬约束质量策略。
- `tests/fakes.py`：Fake LLM、Fake Agent、Fake Amap、Fake Train、Fake Memory 等复用测试桩。

## 开发约束

- 不提交 `.env`、密钥、本地记忆、日志、缓存。
- 修改共享行为时同步更新测试。
- 保持业务智能体调度名为 `search`、`plan`、`clarification`、`memory`。
- 不把 `MainAgent`、`IntentRecognition`、`AgentScheduler` 放进 `agent_schedule`。
- Web/CLI 共用 `TravelFlowCLI.process_query_for_web()` 主链路，修问题时优先保证两端一致。
