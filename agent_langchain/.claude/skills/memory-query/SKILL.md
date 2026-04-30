---
name: memory-query
description: Use this skill when the user asks about their own history, past trips, or saved preferences. Triggers when user asks "我去过哪些地方", "我上次去北京是什么时候", "我之前说过什么偏好", "我的旅行记录". This skill uses MemoryQueryAgent and requires a MemoryManager (user_id, session_id) to access long-term memory.
---

# Memory Query (记忆查询)

这是旧记忆查询能力说明，主链路已弱化为 `MainAgent direct_action + MemoryManager`。兼容实现位于 `context.memory_query.MemoryQueryAgent`，用于旧 `memory-query` 调度或独立记忆问答场景。需传入 **MemoryManager** 以访问长期行程、偏好与聊天摘要。

## When to Use

- 用户问自己的历史行程、偏好、或过往对话内容时

## Agent

- **MemoryQueryAgent** (`context/memory_query.py`)
- 入参：**model**、**memory_manager**（必选，否则无记忆可查）
- **异步**：`run(state)` 为 `async`，需 `await`

## 依赖

- **MemoryManager**：`context.memory_manager.MemoryManager(user_id, session_id, storage_path, llm_model)`
- 长期记忆存储：`data/memory/{user_id}.json`

## 初始化与调用

```python
import asyncio
from context.memory_manager import MemoryManager
from context.memory_query import MemoryQueryAgent
from utils.langchain_runtime import build_chat_model

async def memory_query(user_query: str, user_id: str = "default_user", session_id: str = "default"):
    model = build_chat_model()
    memory_manager = MemoryManager(user_id=user_id, session_id=session_id, llm_model=model)
    agent = MemoryQueryAgent(
        name="MemoryQueryAgent",
        model=model,
        memory_manager=memory_manager,
    )
    return await agent.run({"context": {"rewritten_query": user_query}})

# 使用
data = asyncio.run(memory_query("我去过哪些地方？"))
# data: {"status": "success", "query": "...", "answer": "...", "memory_sources": {"trip_count", "has_preferences", ...}}
```

## 返回格式

- `status`: `"success"` 或 `"error"`
- `query`: 用户问题
- `answer`: 基于记忆的自然语言回答
- `memory_sources`: 如 `trip_count`, `has_preferences`, `has_chat_summary`


## 回答指南

【回答要求】
1. 直接基于上述记忆信息回答问题
2. 如果记忆中没有相关信息，诚实说明"记录中没有相关信息"
3. 回答要自然、准确、有条理
4. 如果有多条记录，可以按时间顺序或分类列举
5. 不要编造不存在的信息

请直接回答用户的问题。
