"""Deterministic intent-routing rules for TravelFlow."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agents.intent_llm import AgentScheduleItem, IntentItem, IntentionOutput


class IntentRuleEngine:
    """Rule fallback and guardrails shared by LLM intent recognition."""

    def _is_planning_intention(self, result_dict: Dict[str, Any]) -> bool:
        intents = result_dict.get("intents") or []
        for item in intents:
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() in {
                "plan",
                "travel_plan",
                "itinerary",
                "itinerary_planning",
            }:
                return True
        schedule = result_dict.get("agent_schedule") or []
        return any(isinstance(item, dict) and item.get("agent_name") == "plan" for item in schedule)

    def _remove_unsupported_history_inferences(self, result_dict: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        query = (user_query or "").strip()
        if not query or not self._is_minimal_destination_plan_query(query):
            return result_dict
        if not self._is_planning_intention(result_dict):
            return result_dict

        destination = self._extract_destination_from_current_query(query)
        key_entities = result_dict.get("key_entities") if isinstance(result_dict.get("key_entities"), dict) else {}
        guarded_entities: Dict[str, Any] = {}
        if destination:
            guarded_entities["destination"] = destination
        elif key_entities.get("destination") and str(key_entities.get("destination")) in query:
            guarded_entities["destination"] = key_entities.get("destination")

        return {**result_dict, "key_entities": guarded_entities, "rewritten_query": query}

    def _is_minimal_destination_plan_query(self, query: str) -> bool:
        if any(marker in query for marker in ("继续", "刚才", "上次", "之前", "这个行程", "这次行程", "按原计划")):
            return False
        if any(marker in query for marker in ("天气", "气温", "预报", "有什么好玩", "攻略", "介绍")):
            return False
        if re.search(r"从[^，。！？\s]{1,20}(?:出发)?(?:去|到|前往)", query):
            return False
        if re.search(r"[一二两三四五六七八九十\d]+\s*天", query):
            return False
        if any(marker in query for marker in ("明天", "后天", "下周", "周末", "月", "日", "预算", "经济", "舒适", "品质", "轻松", "紧凑")):
            return False
        return self._extract_destination_from_current_query(query) is not None

    def _extract_destination_from_current_query(self, query: str) -> Optional[str]:
        destination_tail = r"(?=旅游|旅行|游玩|出游|自由行|度假|玩|，|。|！|？|\s|$)"
        match = re.search(
            rf"(?:想去|要去|打算去|计划去|去|到|前往)(?P<destination>[\u4e00-\u9fa5A-Za-z0-9·]{{2,12}}?){destination_tail}",
            query,
        )
        if not match:
            return None
        destination = match.group("destination").strip()
        destination = re.sub(r"^(一下|一趟)", "", destination).strip()
        return destination or None

    def _is_conversation_intent(self, result_dict: Dict[str, Any], user_query: str) -> bool:
        query = (user_query or "").strip().lower()
        if self._is_direct_conversation_query(query):
            return True

        intents = result_dict.get("intents") or []
        conversation_types = {
            "conversation",
            "chat",
            "greeting",
            "identity",
            "capability",
            "greeting_or_identity",
            "assistant_identity",
            "system_function_inquiry",
            "function_inquiry",
            "system_capability",
            "capability_inquiry",
            "smalltalk",
            "none",
        }
        business_types = {"search", "plan", "clarification", "memory", "preference", "weather"}
        normalized_types = []
        for item in intents:
            if not isinstance(item, dict):
                continue
            intent_type = str(item.get("type", "")).strip().lower()
            if intent_type:
                normalized_types.append(intent_type)

        if normalized_types and all(intent_type in conversation_types for intent_type in normalized_types):
            return True
        if normalized_types and any(intent_type in conversation_types for intent_type in normalized_types):
            if not any(intent_type in business_types for intent_type in normalized_types):
                return True
        return False

    def _is_direct_conversation_query(self, query: str) -> bool:
        if not query:
            return False
        normalized_query = self._normalize_direct_query(query)
        if any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我")):
            return False
        exact_queries = {
            "你好",
            "您好",
            "hello",
            "hi",
            "嗨",
            "你是谁",
            "你是啥",
            "你叫什么",
            "你叫什么名字",
            "你是什么",
            "你能做什么",
            "你会做什么",
            "你可以做什么",
            "你有什么功能",
            "怎么用",
            "在吗",
            "在不在",
            "有人吗",
            "谢谢",
            "谢谢你",
            "多谢",
            "感谢",
            "thanks",
            "thankyou",
            "thx",
            "好的",
            "好",
            "ok",
            "okay",
            "嗯",
            "明白",
            "收到",
        }
        if normalized_query in exact_queries:
            return True
        return any(
            phrase in query
            for phrase in (
                "介绍一下你自己",
                "你是一个什么",
                "你是干什么的",
                "你是做什么的",
                "你做什么的",
                "这个系统怎么用",
            )
        )

    def _build_direct_conversation_answer(self, user_query: str) -> str:
        query = (user_query or "").strip().lower()
        normalized_query = self._normalize_direct_query(query)
        if any(phrase in query for phrase in ("你能做什么", "你会做什么", "你可以做什么", "你有什么功能", "怎么用", "你是做什么的", "你做什么的")):
            return (
                "我可以帮你做旅行规划：收集目的地、出发地、时间、预算和同行人信息；"
                "查询高德 MCP 提供的天气、地点和路线数据；结合你的历史偏好生成个性化行程。"
                "你可以直接说：帮我规划下周从上海去北京玩三天。"
            )
        if normalized_query in {"你好", "您好", "hello", "hi", "嗨"}:
            return "你好，我是 TravelFlow 旅游出行助手。你可以直接告诉我想去哪里、从哪里出发、什么时候去和玩几天。"
        if normalized_query in {"在吗", "在不在", "有人吗"}:
            return "我在。你可以告诉我目的地、出发地、时间、天数和偏好，我来帮你规划行程。"
        if normalized_query in {"谢谢", "谢谢你", "多谢", "感谢", "thanks", "thankyou", "thx"}:
            return "不客气。后续如果要继续调整目的地、时间、预算或节奏，可以直接告诉我。"
        if normalized_query in {"好的", "好", "ok", "okay", "嗯", "明白", "收到"}:
            return "好的。你可以继续补充目的地、出发地、时间、天数、预算或偏好。"
        return (
            "我是 TravelFlow 旅游出行助手，一个基于 LangChain/LangGraph 多智能体架构的旅行规划系统。"
            "我可以帮你收集出行意向、查询目的地信息和天气、结合你的偏好生成行程，也能记住你的长期旅行偏好。"
        )

    def _normalize_direct_query(self, query: str) -> str:
        normalized = (query or "").strip().lower()
        normalized = re.sub(r"[\s\ufeff\u200b]+", "", normalized)
        normalized = re.sub(r"[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+$", "", normalized)
        normalized = re.sub(r"^[。！？!?，,、；;：:\"'“”‘’（）()\[\]{}<>《》~～.]+", "", normalized)
        return normalized

    def _build_error_fallback_intention(self, user_query: str, error: Exception) -> IntentionOutput:
        direct_result = self._try_direct_intent(user_query)
        if direct_result:
            return IntentionOutput.model_validate(
                {
                    **direct_result,
                    "reasoning": f"{direct_result.get('reasoning', '')}（LLM 意图识别不可用，使用规则兜底。错误: {error}）",
                }
            )

        query = (user_query or "").strip()
        if not query:
            return IntentionOutput(
                reasoning=f"意图识别出错且用户输入为空，直接提示用户补充需求。错误: {error}",
                intents=[
                    IntentItem(
                        type="conversation",
                        confidence=1.0,
                        description="空输入",
                        reason="没有可用于搜索或规划的用户需求。",
                    )
                ],
                key_entities={},
                rewritten_query=query,
                direct_answer="我在。你可以告诉我想去哪里、从哪里出发、什么时候去和玩几天。",
                agent_schedule=[],
            )

        return IntentionOutput(
            reasoning=f"意图识别出错，使用保守信息查询兜底。错误: {error}",
            intents=[
                IntentItem(
                    type="search",
                    confidence=0.4,
                    description="保守信息查询兜底",
                    reason="无法调用 LLM 识别意图，且规则未命中直接回复、天气、车次或规划请求。",
                )
            ],
            key_entities={},
            rewritten_query=query,
            agent_schedule=[
                AgentScheduleItem(
                    agent_name="search",
                    priority=1,
                    reason="保守信息查询兜底",
                    expected_output="查询结果",
                )
            ],
        )

    def _try_direct_intent(self, user_query: str) -> Optional[Dict[str, Any]]:
        query = (user_query or "").strip()
        if not query:
            return None

        identity_questions = ("你是谁", "你是啥", "你叫什么", "你叫什么名字", "介绍一下你自己", "你是什么")
        if any(question in query for question in identity_questions) and not any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我")):
            return {
                "reasoning": "用户在询问当前助手身份，属于对话型元问题，应由系统直接回答，不需要调用外部搜索。",
                "intents": [{"type": "conversation", "confidence": 1.0, "description": "系统身份问答", "reason": "用户询问 TravelFlow 助手是谁。"}],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": self._build_direct_conversation_answer(query),
                "agent_schedule": [],
            }

        normalized = self._normalize_direct_query(query)
        direct_map = {
            ("你好", "您好", "hello", "hi", "嗨"): ("普通问候", "用户发送问候语。"),
            ("在吗", "在不在", "有人吗"): ("在线确认", "用户询问助手是否在。"),
            ("谢谢", "谢谢你", "多谢", "感谢", "thanks", "thankyou", "thx"): ("感谢回应", "用户表达感谢。"),
            ("好的", "好", "ok", "okay", "嗯", "明白", "收到"): ("确认回应", "用户发送简短确认。"),
        }
        for words, (description, reason) in direct_map.items():
            if normalized in words:
                return {
                    "reasoning": "用户输入属于普通对话，不需要调用外部搜索或规划智能体。",
                    "intents": [{"type": "conversation", "confidence": 1.0, "description": description, "reason": reason}],
                    "key_entities": {},
                    "rewritten_query": query,
                    "direct_answer": self._build_direct_conversation_answer(query),
                    "agent_schedule": [],
                }

        current_trip_constraints = self._try_current_trip_constraints(query)
        if current_trip_constraints:
            return current_trip_constraints

        memory_result = self._try_direct_memory_intent(query)
        if memory_result:
            return memory_result

        constraints_result = self._try_ambiguous_trip_constraints(query)
        if constraints_result:
            return constraints_result

        capability_questions = ("你能做什么", "你会做什么", "你可以做什么", "你是做什么的", "你做什么的", "怎么用", "你有什么功能")
        if any(question in query for question in capability_questions):
            return {
                "reasoning": "用户询问系统能力，属于产品能力说明，应直接回答并引导用户继续对话。",
                "intents": [{"type": "conversation", "confidence": 1.0, "description": "能力说明", "reason": "用户询问 TravelFlow 能提供哪些帮助。"}],
                "key_entities": {},
                "rewritten_query": query,
                "direct_answer": self._build_direct_conversation_answer(query),
                "agent_schedule": [],
            }

        asks_required_fields = (
            ("需要" in query or "要" in query)
            and any(word in query for word in ("提供", "填写", "告诉", "补充"))
            and any(word in query for word in ("什么", "哪些", "啥"))
            and any(word in query for word in ("意向", "信息", "内容", "资料", "字段"))
        )
        if asks_required_fields:
            return {
                "reasoning": "用户在询问规划旅行前需要提供哪些信息，属于事项收集说明，不应查询或更新记忆。",
                "intents": [{"type": "clarification", "confidence": 1.0, "description": "说明旅行规划所需信息", "reason": "用户询问需要提供哪些意向/信息。"}],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [{"agent_name": "clarification", "priority": 1, "reason": "告知用户需要补充的旅行规划信息", "expected_output": "需要用户提供的信息清单"}],
            }

        if any(word in query for word in ("火车", "高铁", "动车", "城际", "车次", "12306", "余票", "票价", "火车票", "高铁票")):
            return {
                "reasoning": "用户询问火车/高铁车次、时间、票价或余票，直接调度信息检索智能体使用 12306 MCP。",
                "intents": [{"type": "search", "confidence": 1.0, "description": "火车车次查询", "reason": "用户明确询问铁路出行信息。"}],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [{"agent_name": "search", "priority": 1, "reason": "调用 12306 MCP 查询车次、时间、票价或余票", "expected_output": "铁路车次时间、余票与票价信息"}],
            }

        if any(word in query for word in ("天气", "气温", "下雨", "预报")):
            return {
                "reasoning": "用户询问天气，直接调度信息检索智能体使用高德 MCP maps_weather。",
                "intents": [{"type": "search", "confidence": 1.0, "description": "天气查询", "reason": "用户明确询问天气或天气预报。"}],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [{"agent_name": "search", "priority": 1, "reason": "调用高德 MCP maps_weather 查询天气", "expected_output": "城市天气预报"}],
            }

        planning_words = ("规划", "安排", "行程", "路线", "旅游", "玩", "旅行")
        has_trip_movement = ("从" in query and any(word in query for word in ("去", "到", "前往"))) or any(word in query for word in ("三天", "两天", "一天", "下周", "明天", "后天"))
        if any(word in query for word in planning_words) and has_trip_movement:
            return {
                "reasoning": "用户提出明确行程规划请求，按层级执行：先收集行程字段，再检索外部信息，最后结合 MainAgent 注入的记忆上下文生成计划。",
                "intents": [{"type": "plan", "confidence": 1.0, "description": "旅行行程规划", "reason": "用户明确要求规划行程。"}],
                "key_entities": {},
                "rewritten_query": query,
                "agent_schedule": [
                    {"agent_name": "clarification", "priority": 1, "reason": "提取出发地、目的地、时间和天数", "expected_output": "结构化行程字段"},
                    {"agent_name": "search", "priority": 2, "reason": "基于目的地调用高德 API 检索景点、天气和路线", "expected_output": "目的地 POI、天气、路线等外部数据"},
                    {"agent_name": "plan", "priority": 3, "reason": "整合事项字段、外部信息和 MainAgent 注入的记忆上下文生成旅行计划", "expected_output": "三天结构化旅行计划"},
                ],
            }

        # Match explicit travel-intent phrases like "想去/要去/打算去/计划去"
        if re.search(r"(?:想去|要去|打算去|计划去|准备去)[一-龥A-Za-z]{2,}", query):
            destination = self._extract_destination_from_current_query(query)
            key_entities = {"destination": destination} if destination else {}
            return {
                "reasoning": "用户表达明确出行意向（想去/要去/打算去），按行程规划流程处理。",
                "intents": [{"type": "plan", "confidence": 0.95, "description": "旅行行程规划", "reason": "用户明确表达出行意向。"}],
                "key_entities": key_entities,
                "rewritten_query": query,
                "agent_schedule": [
                    {"agent_name": "clarification", "priority": 1, "reason": "提取出发地、目的地、时间和天数", "expected_output": "结构化行程字段"},
                    {"agent_name": "search", "priority": 2, "reason": "基于目的地调用高德 API 检索景点、天气和路线", "expected_output": "目的地 POI、天气、路线等外部数据"},
                    {"agent_name": "plan", "priority": 3, "reason": "整合事项字段、外部信息和 MainAgent 注入的记忆上下文生成旅行计划", "expected_output": "结构化旅行计划"},
                ],
            }

        return None

    def _try_direct_memory_intent(self, query: str) -> Optional[Dict[str, Any]]:
        personal_markers = ("我的", "我过去", "我以前", "我历史", "我去过", "我喜欢", "我不喜欢", "我偏好", "我常", "下次", "以后", "记住", "帮我记住")
        memory_targets = ("偏好", "喜好", "历史", "行程", "去过", "记录", "记得", "预算", "节奏", "酒店", "交通方式", "常住", "航空", "航班", "航空公司", "东航", "南航", "国航", "海航")
        identity_memory = any(phrase in query for phrase in ("我是谁", "知道我是谁", "认识我"))
        is_personal_memory_query = any(marker in query for marker in personal_markers) and any(target in query for target in memory_targets)
        is_preference_statement = any(phrase in query for phrase in ("我喜欢", "我不喜欢", "我偏好", "我的预算", "我的行程节奏", "记住", "帮我记住", "以后都", "长期"))
        if not identity_memory and not is_personal_memory_query and not is_preference_statement:
            return None
        operation = "profile_update" if is_preference_statement else "query"
        return {
            "reasoning": "用户询问或补充自己的长期偏好、历史行程或身份记忆，由 MainAgent 内部记忆能力处理，不调度业务子智能体。",
            "intents": [{"type": "memory", "confidence": 1.0, "description": "个人记忆/偏好处理", "reason": "请求对象是用户自己的偏好、历史或已记录信息。"}],
            "key_entities": {},
            "rewritten_query": query,
            "direct_action": {"type": "memory", "operation": operation, "reason": "读取或更新用户长期偏好、历史行程和行为反馈"},
            "agent_schedule": [],
        }

    def _try_current_trip_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        constraint_words = ("预算", "经济型", "舒适型", "品质型", "住宿", "酒店", "每晚", "餐饮", "吃饭", "交通", "节省", "省钱", "轻松", "均衡", "紧凑", "节奏")
        current_trip_markers = ("本次行程", "这次行程", "此次行程", "本次旅行", "这次旅行", "此次旅行", "这趟", "这一次", "这次")
        long_term_markers = ("以后", "长期", "平时", "通常", "一直", "默认", "记住")
        if not any(word in query for word in constraint_words):
            return None
        if not any(marker in query for marker in current_trip_markers):
            return None
        if any(marker in query for marker in long_term_markers):
            return None

        key_entities: Dict[str, Any] = {"trip_scope": "current"}
        if any(word in query for word in ("经济型", "省钱", "节省")):
            key_entities["budget_level"] = "经济型"
        elif "舒适型" in query:
            key_entities["budget_level"] = "舒适型"
        elif "品质型" in query:
            key_entities["budget_level"] = "品质型"

        lodging_range = re.search(r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*(?:到|至|-|~|－|—)\s*(\d+)\s*元?", query)
        if lodging_range:
            low = int(lodging_range.group(1))
            high = int(lodging_range.group(2))
            key_entities["lodging_budget_per_night_min"] = min(low, high)
            key_entities["lodging_budget_per_night_max"] = max(low, high)
        else:
            lodging_budget = re.search(r"(?:住宿|酒店|每晚)[^0-9一二三四五六七八九十百千万]*(\d+)\s*元?(?:以内|以下|内)?", query)
            if lodging_budget:
                key_entities["lodging_budget_per_night"] = int(lodging_budget.group(1))
                key_entities["lodging_budget_per_night_max"] = int(lodging_budget.group(1))

        if any(word in query for word in ("轻松", "慢节奏")):
            key_entities["pace_preference"] = "轻松"
        elif "紧凑" in query:
            key_entities["pace_preference"] = "紧凑"
        elif "均衡" in query:
            key_entities["pace_preference"] = "均衡"
        if any(word in query for word in ("餐饮", "吃饭")) and any(word in query for word in ("节省", "省钱")):
            key_entities["meal_budget_preference"] = "节省"
        if "交通" in query and any(word in query for word in ("节省", "省钱")):
            key_entities["transport_budget_preference"] = "节省"

        return {
            "reasoning": "用户明确在补充本次行程的预算、住宿、餐饮、交通或节奏约束，应作为当前行程规划字段处理，而不是外部信息查询。",
            "intents": [{"type": "plan", "confidence": 1.0, "description": "本次行程约束补充", "reason": "输入包含“本次/这次行程”等当前行程标记和预算/住宿/餐饮/交通/节奏约束。"}],
            "key_entities": key_entities,
            "rewritten_query": query,
            "agent_schedule": [
                {"agent_name": "clarification", "priority": 1, "reason": "提取并合并本次行程的预算、住宿、餐饮、交通和节奏约束", "expected_output": "更新后的结构化行程字段"},
                {"agent_name": "search", "priority": 2, "reason": "在行程字段完整后检索符合预算约束的外部旅行信息", "expected_output": "目的地 POI、路线、天气和预算相关外部数据"},
                {"agent_name": "plan", "priority": 3, "reason": "按最新预算、节省约束和 MainAgent 注入的记忆上下文生成或调整行程计划", "expected_output": "符合约束的结构化旅行计划"},
            ],
        }

    def _try_ambiguous_trip_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        has_budget_or_pace = any(word in query for word in ("经济型", "舒适型", "品质型", "预算", "轻松", "均衡", "紧凑", "节奏"))
        has_trip_anchor = any(word in query for word in ("从", "去", "到", "前往", "出发", "玩", "旅游", "行程", "天", "月", "日"))
        has_long_term_marker = any(word in query for word in ("我喜欢", "我不喜欢", "我的偏好", "记住", "以后", "长期", "平时", "通常"))
        if not has_budget_or_pace or has_trip_anchor or has_long_term_marker:
            return None
        return {
            "reasoning": "用户只提供了预算/节奏等约束，但没有说明这是本次行程信息还是长期偏好，需要先确认。",
            "intents": [{"type": "conversation", "confidence": 1.0, "description": "澄清约束用途", "reason": "预算和节奏可能是本次行程约束，也可能是长期偏好。"}],
            "key_entities": {},
            "rewritten_query": query,
            "direct_answer": (
                "你说的“"
                + query
                + "”是这次行程的要求，还是希望我保存为长期偏好？"
                "如果是这次行程，请告诉我目的地、出发地、日期和天数；"
                "如果是长期偏好，可以说“以后都按经济型预算、轻松节奏来安排”。"
            ),
            "agent_schedule": [],
        }

    def _try_pending_plan_completion(self, user_query: str, pending_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pending_plan:
            return None
        query = (user_query or "").strip()
        pending_query = str(pending_plan.get("query") or "").strip()
        if not query or not pending_query:
            return None

        # If the user's input looks like a brand-new planning request
        # (planning words + origin-destination movement), treat it as a new
        # request instead of merging into the old pending plan.
        planning_words = ("规划", "安排", "行程", "路线", "旅游", "旅行", "出游", "自由行", "游玩", "玩")
        has_new_planning = any(word in query for word in planning_words)
        has_new_movement = re.search(r"从[一-龥A-Za-z]{1,20}(?:出发)?(?:去|到|前往)", query)
        if has_new_planning and has_new_movement:
            return None

        supplement_markers = (
            "预算",
            "经济",
            "舒适",
            "品质",
            "节奏",
            "轻松",
            "均衡",
            "紧凑",
            "出发",
            "从",
            "在",
            "去",
            "到",
            "目的地",
            "日期",
            "时间",
            "明天",
            "后天",
            "下周",
            "月",
            "日",
            "天",
        )
        if not any(word in query for word in supplement_markers):
            return None
        return {
            "reasoning": "用户正在补充上一轮规划所缺的行程字段；先合并补充信息，再恢复原规划流程。",
            "intents": [{"type": "plan", "confidence": 1.0, "description": "补充字段后继续规划", "reason": "存在待恢复的行程规划请求，当前输入是对缺失行程字段的补充。"}],
            "key_entities": {},
            "rewritten_query": f"{pending_query}；{query}",
            "agent_schedule": [
                {"agent_name": "clarification", "priority": 1, "reason": "合并原始规划请求和用户刚补充的行程字段", "expected_output": "完整结构化行程字段"},
                {"agent_name": "search", "priority": 2, "reason": "基于完整字段调用高德 MCP 检索外部旅行信息", "expected_output": "目的地 POI、天气、路线等外部数据"},
                {"agent_name": "plan", "priority": 3, "reason": "整合补齐后的事项、外部信息和 MainAgent 注入的记忆上下文生成行程", "expected_output": "结构化旅行计划"},
            ],
        }
