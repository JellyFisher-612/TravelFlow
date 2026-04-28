"""LangChain runtime helpers for TravelFlow."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_CONFIG, SYSTEM_CONFIG


def build_chat_model() -> ChatOpenAI:
    """Create the shared LangChain chat model used by all agents."""

    return ChatOpenAI(
        model=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        timeout=float(SYSTEM_CONFIG.get("timeout", 60)),
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        streaming=False,
    )


def to_lc_messages(messages: Iterable[Any]) -> List[BaseMessage]:
    """Convert dict/BaseMessage inputs to LangChain message objects."""

    converted: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            converted.append(msg)
            continue
        if not isinstance(msg, dict):
            converted.append(HumanMessage(content=str(msg)))
            continue

        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=str(content)))
        elif role == "assistant":
            converted.append(AIMessage(content=str(content)))
        else:
            converted.append(HumanMessage(content=str(content)))
    return converted


def message_to_dict(message: BaseMessage) -> Dict[str, str]:
    role = "assistant"
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, SystemMessage):
        role = "system"
    return {"role": role, "content": str(message.content)}


async def ainvoke_text(model: ChatOpenAI, messages: Iterable[Any]) -> str:
    """Invoke a LangChain chat model and return plain text content."""

    response = await model.ainvoke(to_lc_messages(messages))
    content = response.content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)
