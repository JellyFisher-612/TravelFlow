# Codex Test Loop Learnings

Reusable project knowledge discovered by loop iterations should be recorded here.

## Known Architecture

- `TravelFlowCLI.process_query_for_web()` is the shared Web entry point.
- `IntentRecognition` can return `direct_answer` with an empty `agent_schedule`.
- External services must be mocked in tests: LLM, Amap MCP, 12306 MCP, DDGS, Redis/PostgreSQL.

## Known Risks

- Direct-answer scenarios should not call search or expose reference sources.
- Search and plan must not treat unverified hard constraints as facts.
- `pytest` is not installed in the current environment; stdlib `unittest` can run the initial tests until the dependency is available.
- `IntentRecognition` has deterministic pre-LLM routing for short direct-answer turns; tests should assert these cases do not call the model and produce an empty `agent_schedule`.
- `AgentScheduler` tests can use raw dict agent outputs; scheduler wraps them as `result.data`, and downstream context promotion expects fields like `missing_info`, `query_success`, and `preferences` at that raw output level.
- Missing `budget_level` or `pace_preference` is blocking for planning; the scheduler should stop before search/memory/plan and return input requests instead of generating an under-specified plan.
- Search-agent tests can load `agent_langchain/.claude/skills/query-info/script/agent.py` with `importlib.util.spec_from_file_location`, then patch that loaded module's `AmapService`, `TrainService`, `DDGS`, and `DDGS_AVAILABLE` symbols to guarantee no external MCP/search calls.
- `InformationQueryAgent(model=None)` uses rule-based trip search planning, which is useful for deterministic tests of `search_bundle.planning`, `demand_profile`, and `quality.unverified_must_verify`.
