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
from utils.budget_utils import detect_budget_level, detect_lodging_budget
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
        action = self._detect_update_action(query)

        home_location = self._extract_home_location(query)
        if home_location:
            preferences.append({"type": "home_location", "value": home_location, "action": "replace"})

        hotel_brands = self._extract_hotel_brands(query)
        for brand in hotel_brands:
            preferences.append({"type": "hotel_brands", "value": brand, "action": action})

        airlines = self._extract_airlines(query)
        for airline in airlines:
            preferences.append({"type": "airlines", "value": airline, "action": action})

        seat_preference = self._extract_seat_preference(query)
        if seat_preference:
            preferences.append({"type": "seat_preference", "value": seat_preference, "action": "replace"})

        lodging_budget = self._extract_lodging_budget(query)
        if lodging_budget:
            preferences.append({"type": "lodging_budget_per_night", "value": lodging_budget, "action": "replace"})

        if "预算偏好" in query or "预算" in query:
            budget_level = detect_budget_level(query)
            if budget_level:
                preferences.append({"type": "budget_level", "value": budget_level, "action": "replace"})

        implicit_budget = self._extract_implicit_budget_level(query)
        if implicit_budget and not any(item["type"] == "budget_level" for item in preferences):
            preferences.append({"type": "budget_level", "value": implicit_budget, "action": "replace"})

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
            "一般",
            "每次",
            "下次",
            "我还",
            "我也",
            "我常",
            "常坐",
            "搬家",
            "改成",
            "换成",
            "靠窗",
            "过道",
        )
        return any(marker in query for marker in explicit_markers)

    def _detect_update_action(self, query: str) -> str:
        if any(word in query for word in ("还", "也", "另外", "同时")):
            return "append"
        if any(word in query for word in ("改成", "换成", "搬家")):
            return "replace"
        return "replace"

    def _extract_home_location(self, query: str) -> Optional[str]:
        import re

        match = re.search(r"(?:搬家到|住到|定居到|常住在?)([\u4e00-\u9fa5]{2,8})", query)
        if match:
            return re.sub(r"[了啊呀呢吧吗嘛]$", "", match.group(1))
        return None

    def _extract_hotel_brands(self, query: str) -> List[str]:
        known_brands = [
            "汉庭",
            "如家",
            "全季",
            "亚朵",
            "锦江之星",
            "7天",
            "速8",
            "希尔顿",
            "万豪",
            "洲际",
            "桔子",
            "维也纳",
        ]
        return [brand for brand in known_brands if brand in query]

    def _extract_airlines(self, query: str) -> List[str]:
        aliases = {
            "东航": "东航",
            "东方航空": "东航",
            "南航": "南航",
            "南方航空": "南航",
            "国航": "国航",
            "中国国航": "国航",
            "海航": "海航",
            "海南航空": "海航",
            "厦航": "厦航",
            "春秋": "春秋航空",
            "吉祥": "吉祥航空",
        }
        airlines: List[str] = []
        for alias, normalized in aliases.items():
            if alias in query and normalized not in airlines:
                airlines.append(normalized)
        return airlines

    def _extract_seat_preference(self, query: str) -> Optional[str]:
        if "靠窗" in query or "窗口" in query:
            return "window"
        if "过道" in query or "靠走廊" in query:
            return "aisle"
        return None

    def _extract_lodging_budget(self, query: str) -> Optional[Dict[str, int]]:
        import re

        min_value, max_value = detect_lodging_budget(query)
        if min_value is None and max_value is None:
            range_match = re.search(r"住\s*(\d+)\s*(?:到|至|-|~|－|—)\s*(\d+)\s*(?:元)?(?:的)?酒店", query)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                min_value, max_value = min(low, high), max(low, high)
        if min_value is None and max_value is None:
            return None
        value: Dict[str, int] = {}
        if min_value is not None:
            value["min"] = min_value
        if max_value is not None:
            value["max"] = max_value
        return value

    def _extract_implicit_budget_level(self, query: str) -> Optional[str]:
        if any(word in query for word in ("住好一点", "好一点的酒店", "酒店好一点")):
            return "品质型"
        return None

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
