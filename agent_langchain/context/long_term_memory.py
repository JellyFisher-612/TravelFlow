"""
长期记忆 (Long-term Memory)
工业化实现：PostgreSQL 持久化 + Redis 热缓存；数据库不可用时可回退本地 JSON。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, user_id: str, storage_path: str = "data/memory") -> None:
        self.user_id = user_id
        self.storage_path = storage_path
        self.db_path = os.path.join(storage_path, f"{user_id}.json")

        self._pg = None
        self._redis = None
        self._cache_ttl = int(MEMORY_CONFIG.get("cache_ttl_sec", 3600))
        self._allow_json_fallback = bool(MEMORY_CONFIG.get("allow_json_fallback", True))
        self._json_max_chat_messages = int(MEMORY_CONFIG.get("json_max_chat_messages", 2000))
        self._json_chat_ttl_days = int(MEMORY_CONFIG.get("json_chat_ttl_days", 180))
        self._json_max_bytes = int(MEMORY_CONFIG.get("json_max_bytes", 10 * 1024 * 1024))

        self._init_redis()
        self._init_postgres()

        self._json_data: Dict[str, Any] = {}
        if not self._pg and self._allow_json_fallback:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self._json_data = self._load_json()
            self._prune_json_chat_history(save=True)

        logger.info(
            "Long-term memory initialized for user=%s, backend=%s",
            user_id,
            "postgres" if self._pg else "json-fallback",
        )

    # ------------------------- backend init -------------------------

    def _init_redis(self):
        redis_url = MEMORY_CONFIG.get("redis_url", "")
        if not redis_url:
            return

        try:
            import redis

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as e:
            logger.warning("Redis unavailable for long-term cache: %s", e, exc_info=True)
            self._redis = None

    def _init_postgres(self):
        dsn = MEMORY_CONFIG.get("postgres_dsn", "")
        if not dsn:
            logger.info("PostgreSQL DSN not configured, using JSON fallback")
            return

        try:
            import psycopg

            conn = psycopg.connect(dsn, autocommit=True)
            self._pg = conn
            self._ensure_schema()
        except Exception as e:
            logger.warning("PostgreSQL unavailable: %s", e, exc_info=True)
            self._pg = None

    def _ensure_schema(self):
        if not self._pg:
            return

        ddl = """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT NOT NULL,
            pref_type TEXT NOT NULL,
            pref_value JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, pref_type)
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_chat_user_ts ON chat_history (user_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_user_session_ts ON chat_history (user_id, session_id, ts DESC);

        CREATE TABLE IF NOT EXISTS trip_history (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            trip_id TEXT NOT NULL,
            trip_data JSONB NOT NULL,
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_trip_user_ts ON trip_history (user_id, ts DESC);

        CREATE TABLE IF NOT EXISTS behavior_feedback (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            feedback_data JSONB NOT NULL,
            ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_user_ts ON behavior_feedback (user_id, ts DESC);

        CREATE TABLE IF NOT EXISTS session_meta (
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            preview TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            summary_updated_at TIMESTAMPTZ,
            summary_message_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, session_id)
        );
        CREATE INDEX IF NOT EXISTS idx_session_user_active ON session_meta (user_id, last_active DESC);

        ALTER TABLE session_meta ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';
        ALTER TABLE session_meta ADD COLUMN IF NOT EXISTS summary_updated_at TIMESTAMPTZ;
        ALTER TABLE session_meta ADD COLUMN IF NOT EXISTS summary_message_count INTEGER NOT NULL DEFAULT 0;
        """

        with self._pg.cursor() as cur:
            cur.execute(ddl)

    # ------------------------- json fallback -------------------------

    def _init_json_data(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": [],
            "chat_history": [],
            "trip_history": [],
            "behavior_feedback": [],
            "session_meta": {},
            "statistics": {
                "total_trips": 0,
                "total_messages": 0,
                "frequent_destinations": {},
            },
        }

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            logger.debug("Failed to parse stored timestamp: %r", value, exc_info=True)
            return None

    def _prune_json_chat_history(self, save: bool = False):
        if self._pg or not self._allow_json_fallback or not self._json_data:
            return

        history = list(self._json_data.get("chat_history", []))
        original_count = len(history)

        if self._json_chat_ttl_days > 0:
            cutoff = datetime.now() - timedelta(days=self._json_chat_ttl_days)
            history = [
                msg for msg in history
                if (self._parse_timestamp(msg.get("timestamp")) or datetime.now()) >= cutoff
            ]

        if self._json_max_chat_messages > 0 and len(history) > self._json_max_chat_messages:
            history = history[-self._json_max_chat_messages:]

        self._json_data["chat_history"] = history
        self._json_data.setdefault("statistics", {})["total_messages"] = len(history)
        self._prune_empty_session_meta()

        if self._json_max_bytes > 0:
            while history and len(json.dumps(self._json_data, ensure_ascii=False).encode("utf-8")) > self._json_max_bytes:
                history = history[1:]
                self._json_data["chat_history"] = history
                self._json_data.setdefault("statistics", {})["total_messages"] = len(history)
                self._prune_empty_session_meta()

        if save and len(history) != original_count:
            self._save_json()

    def _prune_empty_session_meta(self):
        meta = self._json_data.get("session_meta")
        if not isinstance(meta, dict):
            return
        active_sessions = {
            msg.get("session_id")
            for msg in self._json_data.get("chat_history", [])
            if msg.get("session_id")
        }
        for sid in list(meta.keys()):
            item = meta.get(sid, {})
            has_summary = isinstance(item, dict) and item.get("summary")
            if sid not in active_sessions and not has_summary:
                meta.pop(sid, None)

    def _load_json(self) -> Dict[str, Any]:
        if not os.path.exists(self.db_path):
            return self._init_json_data()

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "session_meta" not in data:
                data["session_meta"] = {}
            if "preferences" not in data:
                data["preferences"] = []
            if "chat_history" not in data:
                data["chat_history"] = []
            if "trip_history" not in data:
                data["trip_history"] = []
            if "behavior_feedback" not in data:
                data["behavior_feedback"] = []
            if "statistics" not in data:
                data["statistics"] = {"total_trips": 0, "total_messages": 0, "frequent_destinations": {}}
            for sid, meta in data.get("session_meta", {}).items():
                if not isinstance(meta, dict):
                    continue
                meta.setdefault("session_id", sid)
                meta.setdefault("user_id", self.user_id)
                meta.setdefault("summary", "")
                meta.setdefault("summary_updated_at", None)
                meta.setdefault("summary_message_count", 0)
            return data
        except Exception as e:
            logger.error("Failed to load JSON fallback data: %s", e, exc_info=True)
            return self._init_json_data()

    def _save_json(self):
        if not self._allow_json_fallback:
            return
        try:
            self._json_data["updated_at"] = datetime.now().isoformat()
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self._json_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save JSON fallback data: %s", e, exc_info=True)

    # ------------------------- cache helpers -------------------------

    def _pref_cache_key(self) -> str:
        return f"travelflow:pref:{self.user_id}"

    def _pref_records_cache_key(self) -> str:
        return f"travelflow:pref_records:{self.user_id}"

    def _session_meta_cache_key(self) -> str:
        return f"travelflow:smeta:{self.user_id}"

    def _cache_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception:
            logger.debug("Failed to read Redis JSON cache for key=%s", key, exc_info=True)
            return None

    def _cache_set_json(self, key: str, value: Dict[str, Any]):
        if not self._redis:
            return
        try:
            self._redis.setex(key, self._cache_ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            logger.debug("Failed to write Redis JSON cache for key=%s", key, exc_info=True)

    def _cache_delete(self, key: str):
        if not self._redis:
            return
        try:
            self._redis.delete(key)
        except Exception:
            logger.debug("Failed to delete Redis cache key=%s", key, exc_info=True)

    # ------------------------- preference -------------------------

    def save_preference(
        self,
        pref_type: str,
        value: Any,
        metadata: Dict[str, Any] = None,
        action: str = "replace",
    ) -> None:
        metadata = metadata or {}
        existing_record = self.get_preference_record(pref_type)
        record = self._build_preference_record(
            pref_type=pref_type,
            value=value,
            metadata=metadata,
            action=action,
            existing_record=existing_record,
        )

        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_preferences(user_id, pref_type, pref_value, updated_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (user_id, pref_type)
                    DO UPDATE SET pref_value = EXCLUDED.pref_value, updated_at = NOW()
                    """,
                    (self.user_id, pref_type, json.dumps(record, ensure_ascii=False)),
                )
            self._cache_delete(self._pref_cache_key())
            self._cache_delete(self._pref_records_cache_key())
            return

        prefs = self._json_data["preferences"]
        found = False
        for pref in prefs:
            if pref.get("type") == pref_type:
                pref.clear()
                pref.update(record)
                found = True
                break
        if not found:
            prefs.append(record)
        self._save_json()

    def get_preference(self, pref_type: str = None) -> Any:
        if self._pg:
            cache = self._cache_get_json(self._pref_cache_key())
            if cache is None:
                with self._pg.cursor() as cur:
                    cur.execute(
                        "SELECT pref_type, pref_value FROM user_preferences WHERE user_id=%s",
                        (self.user_id,),
                    )
                    rows = cur.fetchall()
                cache = {}
                for ptype, pvalue in rows:
                    cache[ptype] = self._preference_value(pvalue)
                self._cache_set_json(self._pref_cache_key(), cache)
            return cache.get(pref_type) if pref_type else cache

        prefs = self._json_data["preferences"]
        if pref_type is None:
            return {
                p.get("type"): self._preference_value(p)
                for p in prefs
                if self._preference_status(p) == "active"
            }
        for p in prefs:
            if p.get("type") == pref_type and self._preference_status(p) == "active":
                return self._preference_value(p)
        return None

    def get_preference_record(self, pref_type: str) -> Optional[Dict[str, Any]]:
        records = self.get_preference_records()
        return records.get(pref_type)

    def get_preference_records(self, include_inactive: bool = False) -> Dict[str, Dict[str, Any]]:
        if self._pg:
            cache = self._cache_get_json(self._pref_records_cache_key())
            if cache is None:
                with self._pg.cursor() as cur:
                    cur.execute(
                        "SELECT pref_type, pref_value FROM user_preferences WHERE user_id=%s",
                        (self.user_id,),
                    )
                    rows = cur.fetchall()
                cache = {}
                for ptype, pvalue in rows:
                    record = self._normalize_preference_record(ptype, pvalue)
                    cache[ptype] = record
                self._cache_set_json(self._pref_records_cache_key(), cache)
            if include_inactive:
                return cache
            return {
                ptype: record
                for ptype, record in cache.items()
                if record.get("status", "active") == "active"
            }

        result = {}
        for pref in self._json_data["preferences"]:
            ptype = pref.get("type")
            if not ptype:
                continue
            record = self._normalize_preference_record(ptype, pref)
            if include_inactive or record.get("status", "active") == "active":
                result[ptype] = record
        return result

    def _normalize_preference_record(self, pref_type: str, raw: Any) -> Dict[str, Any]:
        if self._is_preference_record(raw):
            record = dict(raw)
            record.setdefault("type", pref_type)
            record.setdefault("status", "active")
            record.setdefault("confidence", 1.0)
            record.setdefault("scope", "global")
            record.setdefault("version", 1)
            record.setdefault("source", {})
            record.setdefault("history", [])
            return record

        now = datetime.now().isoformat()
        return {
            "type": pref_type,
            "value": raw.get("value") if isinstance(raw, dict) and "value" in raw else raw,
            "status": "active",
            "scope": "global",
            "confidence": 1.0,
            "source": {},
            "version": 1,
            "created_at": raw.get("created_at", now) if isinstance(raw, dict) else now,
            "updated_at": raw.get("updated_at", now) if isinstance(raw, dict) else now,
            "history": [],
        }

    def _build_preference_record(
        self,
        pref_type: str,
        value: Any,
        metadata: Dict[str, Any],
        action: str,
        existing_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        previous_record = dict(existing_record) if isinstance(existing_record, dict) else None
        version = int(previous_record.get("version", 0)) + 1 if previous_record else 1
        history = list(previous_record.get("history", [])) if previous_record else []
        if previous_record:
            history.append(
                {
                    "value": previous_record.get("value"),
                    "status": previous_record.get("status", "active"),
                    "source": previous_record.get("source", {}),
                    "updated_at": previous_record.get("updated_at"),
                    "version": previous_record.get("version", 1),
                }
            )

        source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
        if not source:
            source = {
                key: metadata.get(key)
                for key in ("session_id", "turn_id", "quote", "owner", "reason")
                if metadata.get(key) is not None
            }

        return {
            "type": pref_type,
            "value": value,
            "status": metadata.get("status", "active"),
            "scope": metadata.get("scope", "global"),
            "confidence": float(metadata.get("confidence", 1.0) or 1.0),
            "source": source,
            "action": action or "replace",
            "version": version,
            "created_at": previous_record.get("created_at", now) if previous_record else now,
            "updated_at": now,
            "history": history[-20:],
        }

    def _is_preference_record(self, value: Any) -> bool:
        return isinstance(value, dict) and "type" in value and "value" in value and (
            "version" in value or "source" in value or "status" in value
        )

    def _preference_value(self, raw: Any) -> Any:
        if isinstance(raw, dict) and "value" in raw:
            return raw.get("value")
        return raw

    def _preference_status(self, raw: Any) -> str:
        if isinstance(raw, dict):
            return str(raw.get("status", "active"))
        return "active"

    def add_hotel_brand(self, brand: str) -> None:
        current = self.get_preference("hotel_brands")
        if isinstance(current, list):
            values = current
        elif current:
            values = [current]
        else:
            values = []
        if brand not in values:
            values.append(brand)
        self.save_preference("hotel_brands", values)

    def add_airline(self, airline: str) -> None:
        current = self.get_preference("airlines")
        if isinstance(current, list):
            values = current
        elif current:
            values = [current]
        else:
            values = []
        if airline not in values:
            values.append(airline)
        self.save_preference("airlines", values)

    # ------------------------- chat history -------------------------

    def add_chat_message(
        self,
        role: str,
        content: str,
        session_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        metadata = metadata or {}
        now = datetime.now().isoformat()
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_history(user_id, session_id, role, content, metadata, ts)
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    """,
                    (self.user_id, session_id, role, content, json.dumps(metadata, ensure_ascii=False)),
                )
            return

        self._json_data["chat_history"].append(
            {
                "role": role,
                "content": content,
                "timestamp": now,
                "session_id": session_id,
                "metadata": metadata,
            }
        )
        self._prune_json_chat_history()
        self._json_data["statistics"]["total_messages"] = len(self._json_data["chat_history"])
        self._save_json()

    def get_chat_history(self, limit: int = None, session_id: str = None) -> List[Dict[str, Any]]:
        if self._pg:
            clauses = ["user_id = %s"]
            params: List[Any] = [self.user_id]
            if session_id:
                clauses.append("session_id = %s")
                params.append(session_id)

            limit_sql = ""
            if limit:
                limit_sql = " LIMIT %s"
                params.append(limit)

            query = (
                "SELECT role, content, ts, session_id, metadata "
                f"FROM chat_history WHERE {' AND '.join(clauses)} "
                "ORDER BY ts ASC" + limit_sql
            )
            with self._pg.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2].isoformat() if row[2] else None,
                    "session_id": row[3],
                    "metadata": row[4] or {},
                }
                for row in rows
            ]

        msgs = self._json_data["chat_history"]
        if session_id:
            msgs = [m for m in msgs if m.get("session_id") == session_id]
        if limit:
            return msgs[-limit:]
        return msgs

    # ------------------------- trip history -------------------------

    def save_trip_history(self, trip_info: Dict[str, Any]) -> None:
        trip_id = f"trip_{int(datetime.now().timestamp() * 1000)}"
        payload = {"trip_id": trip_id, "timestamp": datetime.now().isoformat(), **trip_info}

        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO trip_history(user_id, trip_id, trip_data, ts) VALUES (%s, %s, %s::jsonb, NOW())",
                    (self.user_id, trip_id, json.dumps(payload, ensure_ascii=False)),
                )
            return

        self._json_data["trip_history"].append(payload)
        self._json_data["statistics"]["total_trips"] = len(self._json_data["trip_history"])
        dest = trip_info.get("destination")
        if dest:
            freq = self._json_data["statistics"].setdefault("frequent_destinations", {})
            freq[dest] = freq.get(dest, 0) + 1
        self._save_json()

    def get_trip_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._pg:
            limit = limit or 1000000
            with self._pg.cursor() as cur:
                cur.execute(
                    "SELECT trip_data FROM trip_history WHERE user_id=%s ORDER BY ts ASC LIMIT %s",
                    (self.user_id, limit),
                )
                rows = cur.fetchall()
            return [row[0] for row in rows]

        trips = self._json_data["trip_history"]
        return trips[-limit:] if limit else trips

    def get_frequent_destinations(self, top_n: int = 5) -> List[tuple]:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    SELECT trip_data->>'destination' AS destination, COUNT(*) AS cnt
                    FROM trip_history
                    WHERE user_id=%s AND trip_data ? 'destination'
                    GROUP BY destination
                    ORDER BY cnt DESC
                    LIMIT %s
                    """,
                    (self.user_id, top_n),
                )
                rows = cur.fetchall()
            return [(r[0], int(r[1])) for r in rows if r[0]]

        freq = self._json_data["statistics"].get("frequent_destinations", {})
        return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------- behavior feedback -------------------------

    def save_behavior_feedback(self, feedback: Any, metadata: Dict[str, Any] = None) -> None:
        payload = {
            "feedback": feedback,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    "INSERT INTO behavior_feedback(user_id, feedback_data, ts) VALUES (%s, %s::jsonb, NOW())",
                    (self.user_id, json.dumps(payload, ensure_ascii=False)),
                )
            return

        self._json_data.setdefault("behavior_feedback", []).append(payload)
        self._save_json()

    def get_behavior_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._pg:
            limit = limit or 1000000
            with self._pg.cursor() as cur:
                cur.execute(
                    "SELECT feedback_data FROM behavior_feedback WHERE user_id=%s ORDER BY ts ASC LIMIT %s",
                    (self.user_id, limit),
                )
                rows = cur.fetchall()
            return [row[0] for row in rows]

        feedback = self._json_data.get("behavior_feedback", [])
        return feedback[-limit:] if limit else feedback

    # ------------------------- session meta -------------------------

    def ensure_session_meta(self, session_id: str) -> None:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_meta(user_id, session_id, created_at, last_active, preview)
                    VALUES (%s, %s, NOW(), NOW(), '')
                    ON CONFLICT (user_id, session_id) DO NOTHING
                    """,
                    (self.user_id, session_id),
                )
            self._cache_delete(self._session_meta_cache_key())
            return

        meta = self._json_data.setdefault("session_meta", {})
        if session_id not in meta:
            now = datetime.now().isoformat()
            meta[session_id] = {
                "session_id": session_id,
                "user_id": self.user_id,
                "created_at": now,
                "last_active": now,
                "preview": "",
                "summary": "",
                "summary_updated_at": None,
                "summary_message_count": 0,
            }
            self._save_json()

    def update_session_meta(self, session_id: str, preview: str = "") -> None:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_meta(user_id, session_id, created_at, last_active, preview)
                    VALUES (%s, %s, NOW(), NOW(), %s)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET last_active = NOW(), preview = CASE WHEN EXCLUDED.preview <> '' THEN EXCLUDED.preview ELSE session_meta.preview END
                    """,
                    (self.user_id, session_id, (preview or "")[:80]),
                )
            self._cache_delete(self._session_meta_cache_key())
            return

        self.ensure_session_meta(session_id)
        meta = self._json_data["session_meta"][session_id]
        meta["last_active"] = datetime.now().isoformat()
        if preview:
            meta["preview"] = preview[:80]
        self._save_json()

    def get_session_meta_map(self) -> Dict[str, Dict[str, Any]]:
        if self._pg:
            cache = self._cache_get_json(self._session_meta_cache_key())
            if cache is not None:
                return cache

            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, created_at, last_active, preview,
                           summary, summary_updated_at, summary_message_count
                    FROM session_meta WHERE user_id=%s
                    """,
                    (self.user_id,),
                )
                rows = cur.fetchall()
            result = {}
            for sid, created_at, last_active, preview, summary, summary_updated_at, summary_message_count in rows:
                result[sid] = {
                    "session_id": sid,
                    "user_id": self.user_id,
                    "created_at": created_at.isoformat() if created_at else None,
                    "last_active": last_active.isoformat() if last_active else None,
                    "preview": preview or "",
                    "summary": summary or "",
                    "summary_updated_at": summary_updated_at.isoformat() if summary_updated_at else None,
                    "summary_message_count": int(summary_message_count or 0),
                }
            self._cache_set_json(self._session_meta_cache_key(), result)
            return result

        return self._json_data.get("session_meta", {})

    def update_session_summary(self, session_id: str, summary: str, message_count: int) -> None:
        """Persist an LLM summary for a chat-log session without promoting it to memory."""
        summary = (summary or "").strip()
        message_count = int(message_count or 0)
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_meta(
                        user_id, session_id, created_at, last_active, preview,
                        summary, summary_updated_at, summary_message_count
                    )
                    VALUES (%s, %s, NOW(), NOW(), '', %s, NOW(), %s)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET
                        summary = EXCLUDED.summary,
                        summary_updated_at = NOW(),
                        summary_message_count = EXCLUDED.summary_message_count
                    """,
                    (self.user_id, session_id, summary, message_count),
                )
            self._cache_delete(self._session_meta_cache_key())
            return

        self.ensure_session_meta(session_id)
        meta = self._json_data["session_meta"][session_id]
        meta["summary"] = summary
        meta["summary_updated_at"] = datetime.now().isoformat()
        meta["summary_message_count"] = message_count
        self._save_json()

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        meta = self.get_session_meta_map().get(session_id, {})
        return {
            "session_id": session_id,
            "summary": meta.get("summary", "") if isinstance(meta, dict) else "",
            "summary_updated_at": meta.get("summary_updated_at") if isinstance(meta, dict) else None,
            "summary_message_count": int(meta.get("summary_message_count", 0) or 0) if isinstance(meta, dict) else 0,
        }

    def get_session_summaries(self, limit: int = 5) -> List[Dict[str, Any]]:
        metas = self.get_session_meta_map()
        summaries = []
        for sid, meta in metas.items():
            if not isinstance(meta, dict) or not meta.get("summary"):
                continue
            summaries.append(
                {
                    "session_id": sid,
                    "summary": meta.get("summary", ""),
                    "summary_updated_at": meta.get("summary_updated_at"),
                    "summary_message_count": int(meta.get("summary_message_count", 0) or 0),
                    "last_active": meta.get("last_active"),
                    "preview": meta.get("preview", ""),
                }
            )
        summaries.sort(key=lambda item: item.get("last_active") or "", reverse=True)
        return summaries[:limit] if limit else summaries

    def delete_session(self, session_id: str) -> int:
        """删除指定会话的聊天记录和元数据，返回删除消息数。"""
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_history WHERE user_id=%s AND session_id=%s RETURNING id",
                    (self.user_id, session_id),
                )
                deleted = cur.fetchall()
                cur.execute(
                    "DELETE FROM session_meta WHERE user_id=%s AND session_id=%s",
                    (self.user_id, session_id),
                )
            self._cache_delete(self._session_meta_cache_key())
            return len(deleted)

        original = self._json_data.get("chat_history", [])
        filtered = [m for m in original if m.get("session_id") != session_id]
        deleted_count = len(original) - len(filtered)
        self._json_data["chat_history"] = filtered
        self._json_data.get("session_meta", {}).pop(session_id, None)
        self._json_data.setdefault("statistics", {})["total_messages"] = len(filtered)
        self._save_json()
        return deleted_count

    # ------------------------- stats / maintenance -------------------------

    def increment_query_count(self) -> None:
        # 保留兼容接口；当前未做专门表存储，可按需扩展
        return None

    def get_statistics(self) -> Dict[str, Any]:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM trip_history WHERE user_id=%s", (self.user_id,))
                total_trips = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM chat_history WHERE user_id=%s", (self.user_id,))
                total_messages = int(cur.fetchone()[0])
            frequent = dict(self.get_frequent_destinations(top_n=20))
            return {
                "total_trips": total_trips,
                "total_messages": total_messages,
                "frequent_destinations": frequent,
                "total_preferences": len(self.get_preference_records(include_inactive=True)),
            }

        stats = self._json_data.get("statistics", {}).copy()
        stats["total_preferences"] = len(self.get_preference_records(include_inactive=True))
        return stats

    def clear_history(self) -> None:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute("DELETE FROM chat_history WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM trip_history WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM behavior_feedback WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM session_meta WHERE user_id=%s", (self.user_id,))
            self._cache_delete(self._session_meta_cache_key())
            return

        self._json_data["chat_history"] = []
        self._json_data["trip_history"] = []
        self._json_data["behavior_feedback"] = []
        self._json_data["session_meta"] = {}
        self._json_data["statistics"] = {
            "total_trips": 0,
            "total_messages": 0,
            "frequent_destinations": {},
        }
        self._save_json()

    def delete_all(self) -> None:
        if self._pg:
            with self._pg.cursor() as cur:
                cur.execute("DELETE FROM user_preferences WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM chat_history WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM trip_history WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM behavior_feedback WHERE user_id=%s", (self.user_id,))
                cur.execute("DELETE FROM session_meta WHERE user_id=%s", (self.user_id,))
            self._cache_delete(self._pref_cache_key())
            self._cache_delete(self._pref_records_cache_key())
            self._cache_delete(self._session_meta_cache_key())
            return

        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    # ------------------------- compatibility -------------------------

    @property
    def data(self) -> Dict[str, Any]:
        """兼容旧代码：仅在 JSON 回退模式返回可写 data。"""
        if self._pg:
            return {
                "session_meta": self.get_session_meta_map(),
                "statistics": self.get_statistics(),
            }
        return self._json_data

    def _save(self):
        """兼容旧代码调用。"""
        if not self._pg:
            self._save_json()
