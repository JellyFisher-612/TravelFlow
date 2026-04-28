"""
短期记忆 (Short-term Memory)
优先使用 Redis 作为会话级缓存，失败时回退到进程内内存。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期记忆：最近对话上下文缓存。"""

    def __init__(self, user_id: str, session_id: str, max_turns: int = 10):
        self.user_id = user_id
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_messages = self.max_turns * 2
        self._memory_messages: List[Dict[str, Any]] = []

        self._redis = None
        self._cache_ttl = int(MEMORY_CONFIG.get("cache_ttl_sec", 3600))
        self._key = f"travelflow:stmem:{self.user_id}:{self.session_id}"
        self._pending_plan_key = f"{self._key}:pending_plan"
        self._pending_plan: Optional[Dict[str, Any]] = None

        self._init_redis()

    def _init_redis(self):
        redis_url = MEMORY_CONFIG.get("redis_url", "")
        if not redis_url:
            logger.warning("Redis URL not configured, short-term memory uses in-process fallback")
            return

        try:
            import redis

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("Short-term memory connected to Redis")
        except Exception as e:
            logger.warning("Redis unavailable (%s), short-term memory uses in-process fallback", e)
            self._redis = None

    def _build_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def add_message(self, role: str, content: str, metadata: Dict = None):
        message = self._build_message(role, content, metadata)

        if self._redis:
            try:
                self._redis.rpush(self._key, json.dumps(message, ensure_ascii=False))
                self._redis.ltrim(self._key, -self.max_messages, -1)
                self._redis.expire(self._key, self._cache_ttl)
                return
            except Exception as e:
                logger.warning("Redis write failed, fallback to in-memory short-term cache: %s", e)

        self._memory_messages.append(message)
        if len(self._memory_messages) > self.max_messages:
            self._memory_messages = self._memory_messages[-self.max_messages:]

    def _read_all_messages(self) -> List[Dict[str, Any]]:
        if self._redis:
            try:
                raw_messages = self._redis.lrange(self._key, 0, -1)
                parsed = []
                for raw in raw_messages:
                    try:
                        parsed.append(json.loads(raw))
                    except Exception:
                        continue
                return parsed
            except Exception as e:
                logger.warning("Redis read failed, fallback to in-memory short-term cache: %s", e)

        return self._memory_messages.copy()

    def get_recent_context(self, n_turns: int = None) -> List[Dict[str, Any]]:
        messages = self._read_all_messages()
        if n_turns is None:
            return messages

        n_messages = n_turns * 2
        return messages[-n_messages:] if len(messages) > n_messages else messages

    def get_context_string(self, n_turns: int = 5) -> str:
        messages = self.get_recent_context(n_turns)
        if not messages:
            return "无历史对话"

        lines = []
        for msg in messages:
            role_name = "用户" if msg.get("role") == "user" else "助手"
            lines.append(f"{role_name}: {msg.get('content', '')}")
        return "\n".join(lines)

    def clear(self):
        if self._redis:
            try:
                self._redis.delete(self._key)
                self._redis.delete(self._pending_plan_key)
            except Exception as e:
                logger.warning("Redis delete failed: %s", e)
        self._memory_messages = []
        self._pending_plan = None

    def set_pending_plan(self, query: str, metadata: Optional[Dict[str, Any]] = None):
        payload = {
            "query": query,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        if self._redis:
            try:
                self._redis.setex(self._pending_plan_key, self._cache_ttl, json.dumps(payload, ensure_ascii=False))
                return
            except Exception as e:
                logger.warning("Redis pending plan write failed: %s", e)
        self._pending_plan = payload

    def get_pending_plan(self) -> Optional[Dict[str, Any]]:
        if self._redis:
            try:
                raw = self._redis.get(self._pending_plan_key)
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.warning("Redis pending plan read failed: %s", e)
        return self._pending_plan

    def clear_pending_plan(self):
        if self._redis:
            try:
                self._redis.delete(self._pending_plan_key)
            except Exception as e:
                logger.warning("Redis pending plan delete failed: %s", e)
        self._pending_plan = None

    def get_statistics(self) -> Dict[str, Any]:
        messages = self._read_all_messages()
        return {
            "total_messages": len(messages),
            "max_turns": self.max_turns,
            "oldest_message_time": messages[0].get("timestamp") if messages else None,
            "newest_message_time": messages[-1].get("timestamp") if messages else None,
        }
