"""LLM invocation and schemas for TravelFlow intent recognition."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.langchain_runtime import ainvoke_text
from utils.llm_json import parse_json_text
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

logger = logging.getLogger(__name__)


class IntentItem(BaseModel):
    type: str = Field(default="search", description="意图类型")
    confidence: float = Field(default=0.5, description="置信度，0-1")
    description: str = Field(default="", description="意图说明")
    reason: str = Field(default="", description="识别原因")


class AgentScheduleItem(BaseModel):
    agent_name: str = Field(default="search", description="子智能体名称")
    priority: int = Field(default=1, description="优先级")
    reason: str = Field(default="默认查询", description="调用原因")
    expected_output: str = Field(default="查询结果", description="期望输出")


class IntentionOutput(BaseModel):
    reasoning: str = Field(default="", description="推理过程")
    intents: List[IntentItem] = Field(default_factory=list)
    key_entities: Dict[str, Any] = Field(default_factory=dict)
    rewritten_query: str = Field(default="", description="标准化查询")
    agent_schedule: List[AgentScheduleItem] = Field(default_factory=list)
    direct_answer: str = Field(default="", description="无需调度业务智能体时的直接回复")
    direct_action: Dict[str, Any] = Field(default_factory=dict, description="MainAgent 内部动作，不进入业务智能体调度")


class IntentLLMClient:
    """Invoke the configured LLM and normalize its output into IntentionOutput."""

    def __init__(self, model=None):
        self.model = model

    async def invoke_structured(self, prompt: str) -> IntentionOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                lc_messages = importlib.import_module("langchain_core.messages")
                SystemMessage = getattr(lc_messages, "SystemMessage")
                HumanMessage = getattr(lc_messages, "HumanMessage")

                structured_llm = lc_model.with_structured_output(IntentionOutput)
                result = await structured_llm.ainvoke(
                    [
                        SystemMessage(content="你是一个高级意图识别专家。请严格按结构化字段输出。"),
                        HumanMessage(content=prompt),
                    ]
                )
                if isinstance(result, IntentionOutput):
                    return result
                if isinstance(result, dict):
                    return IntentionOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(
            self.model,
            [
                {"role": "system", "content": "你是一个高级意图识别专家。请仅返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
        )
        return IntentionOutput.model_validate(parse_json_text(str(text)))
