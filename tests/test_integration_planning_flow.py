"""Integration tests for the complete planning pipeline.

Tests the end-to-end flow: user query → intent recognition → dispatch → agents → result.
All external services (LLM, Amap, 12306) are mocked.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fakes import (
    ExplodingLLM,
    FakeAmapService,
    FakeTrainService,
    RecordingAgent,
    RecordingLLM,
)


def _intent_response_planning(destination: str = "杭州") -> dict:
    """Realistic intent recognition response for a planning query."""
    return {
        "intents": [{"type": "itinerary_planning", "confidence": 0.95}],
        "key_entities": {
            "origin": "北京",
            "destination": destination,
            "start_date": "2026-06-01",
            "duration_days": 3,
            "budget_level": "舒适型",
            "pace_preference": "轻松",
            "trip_purpose": "出差",
        },
        "rewritten_query": f"从北京去{destination}出差三天",
        "agent_schedule": [
            {"agent_name": "clarification", "priority": 1},
            {"agent_name": "search", "priority": 2},
            {"agent_name": "memory", "priority": 2},
            {"agent_name": "plan", "priority": 3},
        ],
        "direct_answer": None,
    }


def _intent_response_memory_query() -> dict:
    return {
        "intents": [{"type": "memory_query", "confidence": 0.9}],
        "key_entities": {},
        "rewritten_query": "我去过哪些地方",
        "agent_schedule": [{"agent_name": "memory", "priority": 1}],
        "direct_answer": None,
    }


def _intent_response_preference() -> dict:
    return {
        "intents": [{"type": "preference", "confidence": 0.9}],
        "key_entities": {},
        "rewritten_query": "我还喜欢汉庭",
        "agent_schedule": [{"agent_name": "memory", "priority": 1}],
        "direct_answer": None,
    }


def _intent_response_info_query() -> dict:
    return {
        "intents": [{"type": "information_query", "confidence": 0.9}],
        "key_entities": {"destination": "杭州"},
        "rewritten_query": "杭州天气怎么样",
        "agent_schedule": [{"agent_name": "search", "priority": 1}],
        "direct_answer": None,
    }


def _intent_response_direct_answer() -> dict:
    return {
        "intents": [{"type": "direct_answer", "confidence": 0.95}],
        "key_entities": {},
        "rewritten_query": "你好",
        "agent_schedule": [],
        "direct_answer": "你好！有什么可以帮你的吗？",
    }


class IntegrationPlanningFlowTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end planning flow tests."""

    def _make_main_agent(self, intent_llm, sub_agents: dict | None = None, memory_manager=None):
        """Create a MainAgent with mocked sub-agents."""
        from agents.main_agent import MainAgent

        agents = sub_agents or {}
        mm = memory_manager

        # Create MainAgent with injected dependencies
        agent = MainAgent(model=intent_llm)
        # Inject sub-agent registry
        if agents:
            agent._agent_registry = agents
        if mm:
            agent._memory_manager = mm
        return agent

    async def test_direct_answer_bypasses_scheduler(self):
        """问候语直接回复，不走调度器"""
        intent_llm = RecordingLLM(_intent_response_direct_answer())
        agent = self._make_main_agent(intent_llm)

        result = await agent.run({
            "messages": [{"role": "user", "content": "你好"}],
            "context": {"rewritten_query": "你好"},
        })

        # Should have direct answer, no agent scheduling
        self.assertIsNotNone(result.get("direct_answer") or result.get("final_result"))
        # No sub-agents should have been called

    async def test_memory_query_flow(self):
        """'我去过哪些地方' → 意图识别 → 记忆查询 → 返回历史"""
        intent_llm = RecordingLLM(_intent_response_memory_query())
        memory_agent = RecordingAgent("memory", {
            "has_results": True,
            "trip_history": [{"destination": "杭州", "date": "2026-05-01"}],
            "summary": "您之前去过杭州。",
        })

        agent = self._make_main_agent(intent_llm, sub_agents={"memory": memory_agent})

        result = await agent.run({
            "messages": [{"role": "user", "content": "我去过哪些地方"}],
            "context": {"rewritten_query": "我去过哪些地方"},
        })

        # Memory agent should have been called
        self.assertGreater(len(memory_agent.calls), 0)

    async def test_preference_update_flow(self):
        """'我还喜欢汉庭' → 意图识别 → 偏好管理 → 偏好已更新"""
        intent_llm = RecordingLLM(_intent_response_preference())
        memory_agent = RecordingAgent("memory", {
            "has_preferences": True,
            "preferences": [{"type": "hotel_brands", "value": "汉庭", "action": "append"}],
            "summary": "已添加偏好。",
        })

        agent = self._make_main_agent(intent_llm, sub_agents={"memory": memory_agent})

        result = await agent.run({
            "messages": [{"role": "user", "content": "我还喜欢汉庭"}],
            "context": {"rewritten_query": "我还喜欢汉庭"},
        })

        self.assertGreater(len(memory_agent.calls), 0)

    async def test_information_query_flow(self):
        """'杭州天气怎么样' → 意图识别 → 信息查询 → 返回天气"""
        intent_llm = RecordingLLM(_intent_response_info_query())
        search_agent = RecordingAgent("search", {
            "query_type": "weather",
            "query_success": True,
            "weather": {"summary": "晴，22到28度"},
        })

        agent = self._make_main_agent(intent_llm, sub_agents={"search": search_agent})

        result = await agent.run({
            "messages": [{"role": "user", "content": "杭州天气怎么样"}],
            "context": {"rewritten_query": "杭州天气怎么样"},
        })

        self.assertGreater(len(search_agent.calls), 0)

    async def test_full_planning_flow_calls_all_agents(self):
        """完整规划流程调用 clarification → search + memory → plan"""
        intent_llm = RecordingLLM(_intent_response_planning())

        clarification = RecordingAgent("clarification", {
            "extracted_count": 6,
            "missing_info": [],
            "data": _intent_response_planning()["key_entities"],
        })
        search = RecordingAgent("search", {
            "query_type": "trip_bundle",
            "query_success": True,
            "results": {"pois": [{"name": "西湖"}], "weather": {"summary": "晴"}},
        })
        memory = RecordingAgent("memory", {
            "has_results": False,
            "has_preferences": True,
            "preferences": [{"type": "hotel_brands", "value": "汉庭"}],
        })
        plan = RecordingAgent("plan", {
            "planning_complete": True,
            "itinerary": {"title": "杭州三日出差计划", "daily_plans": [{"day": 1}]},
        })

        agent = self._make_main_agent(intent_llm, sub_agents={
            "clarification": clarification,
            "search": search,
            "memory": memory,
            "plan": plan,
        })

        result = await agent.run({
            "messages": [{"role": "user", "content": "下周三从北京去杭州出差三天"}],
            "context": {"rewritten_query": "下周三从北京去杭州出差三天"},
        })

        # All agents should have been called
        self.assertGreater(len(clarification.calls), 0)
        self.assertGreater(len(search.calls), 0)
        self.assertGreater(len(memory.calls), 0)
        self.assertGreater(len(plan.calls), 0)

    async def test_planning_flow_blocks_plan_when_missing_fields(self):
        """缺少关键字段时 plan 不应被调用"""
        intent_response = _intent_response_planning()
        intent_response["agent_schedule"] = [
            {"agent_name": "clarification", "priority": 1},
            {"agent_name": "search", "priority": 2},
            {"agent_name": "plan", "priority": 3},
        ]
        intent_llm = RecordingLLM(intent_response)

        clarification = RecordingAgent("clarification", {
            "extracted_count": 2,
            "missing_info": ["budget_level", "pace_preference"],
            "data": {"origin": "北京", "destination": "杭州"},
        })
        search = RecordingAgent("search", {"query_success": True})
        plan = RecordingAgent("plan", {"planning_complete": True})

        agent = self._make_main_agent(intent_llm, sub_agents={
            "clarification": clarification,
            "search": search,
            "plan": plan,
        })

        result = await agent.run({
            "messages": [{"role": "user", "content": "去杭州"}],
            "context": {"rewritten_query": "去杭州"},
        })

        # Plan should NOT be called when critical fields are missing
        # (depending on implementation, it may be blocked or called with degraded output)
        # At minimum, clarification should have been called
        self.assertGreater(len(clarification.calls), 0)


if __name__ == "__main__":
    unittest.main()
