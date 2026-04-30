from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List

from utils.json_parser import robust_json_parse
from utils.langchain_runtime import ainvoke_text
from utils.structured_output_guard import (
    is_structured_output_unavailable_error,
    mark_structured_output_unsupported,
    should_attempt_structured_output,
)

from .common import (
    SearchPlanOutput,
    SummaryOutput,
    classify_source,
    is_suspicious_url,
    requires_official_source,
)

logger = logging.getLogger(__name__)

class WebFallbackSearchMixin:
    async def _web_search(self, query: str) -> Dict[str, Any]:
        """
        网络搜索 - 使用 DDGS（Dux Distributed Global Search），开启 safesearch，过滤可疑来源。

        Args:
            query: 用户查询

        Returns:
            搜索结果
        """
        if not self._ddgs_available():
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {
                    "message": "搜索库未安装",
                    "note": "请运行：pip install ddgs",
                },
            }

        try:
            ddgs = self._new_ddgs()
            # 开启安全搜索，优先 bing 后端（质量更稳定），多取几条再过滤
            search_results = []
            for backend in ("bing", "duckduckgo", "auto"):
                try:
                    raw = ddgs.text(
                        query,
                        max_results=10,
                        safesearch="on",
                        region="cn-zh",
                        backend=backend,
                    )
                    search_results = list(raw)
                    if search_results:
                        break
                except Exception as e:
                    logger.debug(f"DDGS backend {backend} failed: {e}")
                    continue

            results = []
            for result in search_results:
                href = result.get("href", "")
                if is_suspicious_url(href):
                    continue
                source_meta = classify_source(href)
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": href,
                    **source_meta,
                })
                if len(results) >= 5:
                    break

            if not results:
                return {
                    "query_type": "网络搜索",
                    "query_success": False,
                    "results": {"message": "未找到相关结果"},
                }

            # 使用 LLM 总结搜索结果
            summary = await self._summarize_search_results(query, results)
            official_sources = [item for item in results if item.get("official")]
            requires_official = requires_official_source(query)

            return {
                "query_type": "网络搜索",
                "query_success": True,
                "verified": bool(official_sources) if requires_official else True,
                "requires_official_source": requires_official,
                "trust_level": "high" if official_sources else "medium",
                "results": {
                    "summary": summary,
                    "sources": results,
                    "official_sources": official_sources,
                    "verification_note": (
                        "已找到官方/准官方来源。"
                        if official_sources
                        else "未找到官方来源；该结果只能作为普通参考，不能用于确认车次、余票、票价、预约或酒店库存。"
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {"error": f"搜索失败: {str(e)}"},
            }

    async def _summarize_search_results(self, query: str, results: List[Dict]) -> str:
        """
        使用 LLM 总结搜索结果

        Args:
            query: 用户查询
            results: 搜索结果列表

        Returns:
            总结文本
        """
        if not results:
            return "未找到相关信息"

        # 构建搜索结果文本
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result['title']}\n{result['snippet']}\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("query-info")
        if not skill_instruction:
            skill_instruction = "请直接回答用户的问题，保持简洁。"

        prompt = f"""根据以下搜索结果，简洁地回答用户的问题。

【当前时间】
{current_date} {weekday}
（用户查询中的相对时间请基于此日期理解，如"明天"、"2月28日"等）

【用户问题】
{query}

【搜索结果】
{results_text}

【任务说明】
{skill_instruction}
"""

        try:
            summarized = await self._invoke_summary(prompt)
            return summarized.summary.strip() if summarized.summary else "无法生成摘要"
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "搜索成功，但摘要生成失败"

    async def _invoke_summary(self, prompt: str) -> SummaryOutput:
        lc_model = self.model
        if should_attempt_structured_output(lc_model):
            try:
                structured_llm = lc_model.with_structured_output(SummaryOutput)
                result = await structured_llm.ainvoke(prompt)
                if isinstance(result, SummaryOutput):
                    return result
                if isinstance(result, dict):
                    return SummaryOutput.model_validate(result)
            except Exception as e:
                if is_structured_output_unavailable_error(e):
                    mark_structured_output_unsupported(lc_model)
                    logger.info("Structured output disabled for current model, fallback to text parsing")
                else:
                    logger.warning("Structured output failed, fallback to text parsing: %s", e)

        text = await ainvoke_text(self.model, [{"role": "user", "content": prompt}])
        return SummaryOutput(summary=str(text).strip())
