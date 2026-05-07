"""
偏好智能体
职责：收集用户的长期偏好
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Dict, List, Optional, Union

from utils.langchain_runtime import ainvoke_text
from utils.llm_json import parse_json_text
from utils.budget_utils import detect_budget_level
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

logger = logging.getLogger(__name__)

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")
Field = getattr(_pydantic, "Field")
ConfigDict = getattr(_pydantic, "ConfigDict")


class PreferenceItem(BaseModel):
    """One extracted long-term user preference and how it should update memory."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="other")
    value: Any = Field(default=None)
    action: str = Field(default="replace")


class PreferenceOutput(BaseModel):
    """LLM-normalized preference extraction result for memory updates."""

    model_config = ConfigDict(extra="allow")

    has_preferences: bool = Field(default=False)
    preferences: List[PreferenceItem] = Field(default_factory=list)
    summary: str = Field(default="")


PreferenceItem.model_rebuild()
PreferenceOutput.model_rebuild()


class PreferenceAgent:
    """偏好智能体。"""

    def __init__(self, name: str = "PreferenceAgent", model=None, memory_manager=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        from utils.skill_loader import SkillLoader

        self.skill_loader = SkillLoader()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", {})
        user_query = context.get("rewritten_query") or state.get("user_query") or ""

        direct = self._try_direct_extract(user_query)
        if direct:
            return direct

        current_preferences = {}
        if self.memory_manager:
            current_preferences = self.memory_manager.long_term.get_preference()

        current_prefs_str = json.dumps(current_preferences, ensure_ascii=False, indent=2)

        skill_instruction = self.skill_loader.get_skill_content("preference")
        if not skill_instruction:
            skill_instruction = "请分析用户的偏好。"

        prompt = f"""你是用户偏好分析专家，负责提取用户长期偏好。

【当前已保存的用户偏好】
{current_prefs_str}

【新的用户输入】
{user_query}

【任务说明】
{skill_instruction}

请返回结构化偏好信息：
- has_preferences: 是否识别到偏好
- preferences: 推荐使用列表格式，每项包含 type/value/action（append 或 replace）
- summary: 一句话总结
"""

        try:
            result = await self._invoke_structured(prompt)
            result_data = result.model_dump()

            if "preferences" not in result_data:
                result_data["preferences"] = []
            if not result_data.get("has_preferences"):
                result_data["has_preferences"] = bool(result_data.get("preferences"))
        except Exception as e:
            logger.error("Preference collection failed: %s", e)
            result_data = {"has_preferences": False, "error": str(e), "preferences": []}

        return result_data

    def _try_direct_extract(self, user_query: str) -> Optional[Dict[str, Any]]:
        query = (user_query or "").strip()
        if not query:
            return None
        if not self._is_explicit_long_term_preference(query):
            return None

        preferences: List[Dict[str, Any]] = []
        if "预算偏好" in query or "预算" in query:
            budget_level = detect_budget_level(query)
            if budget_level:
                preferences.append({"type": "budget_level", "value": budget_level, "action": "replace"})

        if "节奏" in query or "不要太赶" in query or "多看" in query:
            if "轻松" in query or "不要太赶" in query:
                preferences.append({"type": "pace_preference", "value": "轻松", "action": "replace"})
            elif "均衡" in query:
                preferences.append({"type": "pace_preference", "value": "均衡", "action": "replace"})
            elif "紧凑" in query or "多看" in query:
                preferences.append({"type": "pace_preference", "value": "紧凑", "action": "replace"})

        if not preferences:
            return None
        return {
            "has_preferences": True,
            "preferences": preferences,
            "summary": "已识别并更新用户偏好。",
        }

    def _is_explicit_long_term_preference(self, query: str) -> bool:
        explicit_markers = (
            "我喜欢",
            "我不喜欢",
            "我的偏好",
            "我偏好",
            "记住",
            "帮我记住",
            "以后",
            "以后都",
            "长期",
            "平时",
            "通常",
            "每次",
            "下次",
        )
        return any(marker in query for marker in explicit_markers)

    async def _invoke_structured(self, prompt: str) -> PreferenceOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(PreferenceOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, PreferenceOutput):
                    return result
                if isinstance(result, dict):
                    return PreferenceOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        return PreferenceOutput.model_validate(parse_json_text(str(text)))
