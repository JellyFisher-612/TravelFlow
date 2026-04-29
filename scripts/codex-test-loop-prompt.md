# Codex Test Improvement Loop

You are Codex working inside an iterative TravelFlow test improvement loop.

Your job is to read the test requirement document and improve the project one small step at a time. The loop may start a fresh Codex session on every iteration, so the durable context is in repository files.

## Mandatory Files To Read First

Read these files before making changes:

1. `测试需求文档.md`
2. `.codex-loop/progress.md`
3. `.codex-loop/learnings.md`
4. Any nearby `AGENTS.md` files.

## Main Objective

Implement or improve the test system according to `测试需求文档.md`.

The project should gradually gain tests for this core chain:

用户请求
→ 意图识别
→ 主智能体调度
→ Skill / 子 Agent 执行
→ 总结规划
→ 流式返回
→ 最终方案生成

The test focus is user experience:

- 懂我
- 少问
- 快回
- 稳推
- 能改
- 可信

## Iteration Rules

In each iteration, do exactly one coherent improvement.

Choose the highest-value unfinished task from the test requirement document and current progress.

Priority order:

1. Existing failing tests or broken validation.
2. Basic test infrastructure.
3. Direct-answer and intent-recognition tests.
4. Main-agent dispatch tests.
5. Clarification-question tests.
6. Search-agent planning and quality tests.
7. Skill / sub-agent execution tests.
8. Streaming response tests.
9. Multi-turn modification tests.
10. Failure-degradation tests.
11. Final-plan quality tests.
12. Regression fixtures.

## What You May Do

You may:

- Add new test files.
- Add fixtures, fakes, mocks, and evaluation helpers.
- Add regression cases from the requirement document.
- Refactor code only when needed to make the system testable.
- Fix bugs discovered by tests.
- Improve logging or traceability if it helps testing.
- Update documentation for how to run tests.

## What You Must Not Do

Do not:

- Rewrite the whole project.
- Delete tests to make validation pass.
- Bypass important assertions.
- Hardcode outputs only to satisfy one test case.
- Modify production behavior without test coverage unless the current iteration is explicitly creating the coverage.
- Call real LLMs, Amap MCP, 12306 MCP, DDGS, Redis, PostgreSQL, or other external services in automated tests.
- Invent real-time travel data as if it were verified.
- Expose internal tool logs to end users.
- Commit `.env`, API keys, local memory JSON, runtime logs, or cache files.

## Expected Behavior

For each iteration:

1. Inspect the repo structure.
2. Inspect current tests.
3. Read `测试需求文档.md`.
4. Read `.codex-loop/progress.md`.
5. Read `.codex-loop/learnings.md`.
6. Pick one small test improvement.
7. Implement it.
8. Run the most relevant validation command.
9. If validation fails, fix the failure.
10. Update `.codex-loop/progress.md`.
11. Update `.codex-loop/learnings.md` if you discover reusable project knowledge.

## Progress Update Format

Append this to `.codex-loop/progress.md`:

```md
## YYYY-MM-DD HH:MM - Iteration summary

### Requirement area
Which section of 测试需求文档.md this iteration addressed.

### Change made
What was added or changed.

### Files changed
- path/to/file: reason

### Validation
Commands run and results.

### Remaining work
What should be done next.
```

## Completion

Only write `ALL_TEST_REQUIREMENTS_DONE` in `.codex-loop/progress.md` when the P0 and P1 requirement areas have meaningful automated coverage and validation is passing.
