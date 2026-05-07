"""
JSON 解析辅助工具
提供更鲁棒的 JSON 解析功能，处理常见的格式问题
"""
import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def robust_json_parse(text: str | dict, fallback: Any = None) -> dict:
    """
    鲁棒的 JSON 解析函数

    Args:
        text: 要解析的文本（可能包含 JSON）
        fallback: 解析失败时的默认返回值

    Returns:
        解析后的字典，或 fallback 值

    Raises:
        ValueError: 如果所有解析尝试都失败且没有提供 fallback
    """
    if not text:
        if fallback is not None:
            return fallback
        raise ValueError("Empty text provided")

    # 如果已经是字典，直接返回
    if isinstance(text, dict):
        return text

    json_str = _extract_json_payload(text)
    if not json_str:
        if fallback is not None:
            logger.warning("No JSON found in text, using fallback")
            return fallback
        raise ValueError("No JSON found in response")

    strategies = [
        (_try_direct_parse, "Direct JSON parse failed", _log_direct_parse_context),
        (_try_clean_control_chars, "JSON parse failed after cleaning", None),
        (_try_fix_quotes, "JSON parse failed after fixing quotes", None),
        (_try_remove_trailing_commas, "JSON parse failed after removing trailing commas", None),
        (_try_escape_newlines, "JSON parse failed after smart escaping", None),
    ]
    last_error = None

    for parser, failure_message, extra_logger in strategies:
        try:
            return parser(json_str)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"{failure_message}: {e}")
            if extra_logger:
                extra_logger(json_str, e)

    # 尝试6: 使用 json5 或其他宽松解析器（如果可用）
    try:
        import json5
        result = json5.loads(json_str)
        logger.info("JSON parsed successfully using json5")
        return result
    except ImportError:
        logger.debug("json5 not available")
    except Exception as e:
        logger.warning(f"JSON5 parse failed: {e}")

    # 所有尝试都失败
    logger.error(f"All JSON parsing attempts failed. Full JSON:\n{json_str}")

    if fallback is not None:
        logger.warning("Using fallback value")
        return fallback

    raise ValueError(f"Failed to parse JSON after all attempts. Last error: {last_error}")


def _extract_json_payload(text: str) -> str:
    """移除 markdown 包装并提取 JSON 对象文本。"""
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return ""
    return text[start_idx:end_idx + 1]


def _try_direct_parse(json_str: str):
    """尝试直接 json.loads。"""
    return json.loads(json_str)


def _try_clean_control_chars(json_str: str):
    """移除控制字符后重试。"""
    json_str_cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    result = json.loads(json_str_cleaned)
    logger.info("JSON parsed successfully after removing control characters")
    return result


def _try_fix_quotes(json_str: str):
    """修复常见单引号键和值后重试。"""
    json_str_fixed = re.sub(r"'([^']*)'(\s*:\s*)", r'"\1"\2', json_str)
    json_str_fixed = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str_fixed)
    result = json.loads(json_str_fixed)
    logger.info("JSON parsed successfully after fixing quotes")
    return result


def _try_remove_trailing_commas(json_str: str):
    """移除对象和数组中的尾部逗号后重试。"""
    json_str_fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    result = json.loads(json_str_fixed)
    logger.info("JSON parsed successfully after removing trailing commas")
    return result


def _try_escape_newlines(json_str: str):
    """只转义字符串内部的换行、回车和制表符后重试。"""
    json_str_fixed = _escape_newlines_in_strings(json_str)
    result = json.loads(json_str_fixed)
    logger.info("JSON parsed successfully after smart escaping")
    return result


def _escape_newlines_in_strings(json_str: str) -> str:
    result = []
    in_string = False
    escape_next = False

    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string and char in ('\n', '\r', '\t'):
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
        else:
            result.append(char)

    return ''.join(result)


def _log_direct_parse_context(json_str: str, error: json.JSONDecodeError) -> None:
    error_pos = getattr(error, 'pos', 0)
    start = max(0, error_pos - 50)
    end = min(len(json_str), error_pos + 50)
    logger.warning(f"Error context: ...{json_str[start:end]}...")
