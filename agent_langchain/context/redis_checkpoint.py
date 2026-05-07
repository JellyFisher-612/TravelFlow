"""Redis-backed checkpoint saver for LangGraph StateGraph.

Replaces the in-memory MemorySaver so that orchestration state survives
process restarts.  Each thread's checkpoints are stored as Redis strings
(JSON blobs) keyed by ``{prefix}:{thread_id}:...``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence

import redis

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    WRITES_IDX_MAP,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpointer backed by Redis.

    Stores checkpoint blobs as JSON strings.  Channel values are stored
    separately so that only changed channels are written on each step.
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "travelflow:checkpoint",
        ttl_sec: int = 3600,
    ) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix
        self._ttl = ttl_sec

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key_latest(self, thread_id: str, ns: str = "") -> str:
        return f"{self._prefix}:{thread_id}:{ns}:latest" if ns else f"{self._prefix}:{thread_id}:latest"

    def _key_checkpoint(self, thread_id: str, cp_id: str, ns: str = "") -> str:
        return f"{self._prefix}:{thread_id}:{ns}:cp:{cp_id}" if ns else f"{self._prefix}:{thread_id}:cp:{cp_id}"

    def _key_meta(self, thread_id: str, cp_id: str, ns: str = "") -> str:
        return f"{self._prefix}:{thread_id}:{ns}:meta:{cp_id}" if ns else f"{self._prefix}:{thread_id}:meta:{cp_id}"

    def _key_blob(self, thread_id: str, ns: str, channel: str, version: str) -> str:
        return f"{self._prefix}:{thread_id}:{ns}:blob:{channel}:{version}"

    def _key_writes(self, thread_id: str, ns: str, cp_id: str) -> str:
        return f"{self._prefix}:{thread_id}:{ns}:writes:{cp_id}"

    def _key_parent(self, thread_id: str, cp_id: str, ns: str = "") -> str:
        return f"{self._prefix}:{thread_id}:{ns}:parent:{cp_id}" if ns else f"{self._prefix}:{thread_id}:parent:{cp_id}"

    # ------------------------------------------------------------------
    # BaseCheckpointSaver interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            return self._load_tuple(thread_id, ns, checkpoint_id, config)

        # No specific checkpoint requested – return the latest.
        latest_id = self._redis.get(self._key_latest(thread_id, ns))
        if not latest_id:
            return None
        return self._load_tuple(thread_id, ns, latest_id, config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = checkpoint["id"]

        # Store channel blobs.
        channel_values: dict[str, Any] = checkpoint.pop("channel_values")  # type: ignore[misc]
        pipe = self._redis.pipeline()
        for ch, ver in new_versions.items():
            blob_key = self._key_blob(thread_id, ns, ch, ver)
            if ch in channel_values:
                typed = self.serde.dumps_typed(channel_values[ch])
                pipe.setex(blob_key, self._ttl, json.dumps(typed))
            else:
                pipe.setex(blob_key, self._ttl, json.dumps(("empty", b"")))
        pipe.execute()

        # Store checkpoint (without channel_values) and metadata.
        cp_blob = self.serde.dumps_typed(checkpoint)
        meta_blob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        parent_id = config["configurable"].get("checkpoint_id")

        pipe = self._redis.pipeline()
        pipe.setex(self._key_checkpoint(thread_id, cp_id, ns), self._ttl, json.dumps(cp_blob))
        pipe.setex(self._key_meta(thread_id, cp_id, ns), self._ttl, json.dumps(meta_blob))
        if parent_id:
            pipe.setex(self._key_parent(thread_id, cp_id, ns), self._ttl, parent_id)
        pipe.setex(self._key_latest(thread_id, ns), self._ttl, cp_id)
        pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": cp_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cp_id = config["configurable"]["checkpoint_id"]
        writes_key = self._key_writes(thread_id, ns, cp_id)

        pipe = self._redis.pipeline()
        for idx, (channel, value) in enumerate(writes):
            inner_idx = WRITES_IDX_MAP.get(channel, idx)
            if inner_idx < 0:
                continue
            field = f"{task_id}:{inner_idx}"
            blob = self.serde.dumps_typed(value)
            pipe.hset(writes_key, field, json.dumps((task_id, channel, blob, task_path)))
        pipe.expire(writes_key, self._ttl)
        pipe.execute()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        latest_id = self._redis.get(self._key_latest(thread_id, ns))
        if not latest_id:
            return
        # Walk backwards through parent pointers.
        cp_id = latest_id
        count = 0
        while cp_id:
            if limit and count >= limit:
                break
            before_id = get_checkpoint_id(before) if before else None
            if before_id and cp_id >= before_id:
                cp_id = self._redis.get(self._key_parent(thread_id, cp_id, ns))
                continue
            tup = self._load_tuple(thread_id, ns, cp_id, config)
            if tup:
                yield tup
                count += 1
            cp_id = self._redis.get(self._key_parent(thread_id, cp_id, ns))

    def delete_thread(self, thread_id: str) -> None:
        pattern = f"{self._prefix}:{thread_id}:*"
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break

    # ------------------------------------------------------------------
    # Simple checkpoint helpers used by non-LangGraph callers/tests
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        data: Any,
        ns: str = "",
    ) -> bool:
        """Store a JSON checkpoint payload without raising Redis errors."""
        try:
            self._redis.setex(
                self._key_checkpoint(thread_id, checkpoint_id, ns),
                self._ttl,
                json.dumps(data, ensure_ascii=False),
            )
            self._redis.setex(self._key_latest(thread_id, ns), self._ttl, checkpoint_id)
            return True
        except Exception as e:
            logger.warning("Redis checkpoint save failed: %s", e)
            return False

    def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
        ns: str = "",
    ) -> Any | None:
        """Load a JSON checkpoint payload, returning None when absent/unavailable."""
        try:
            resolved_id = checkpoint_id or self._redis.get(self._key_latest(thread_id, ns))
            if not resolved_id:
                return None

            raw = self._redis.get(self._key_checkpoint(thread_id, resolved_id, ns))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Redis checkpoint load failed: %s", e)
            return None

    def delete_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
        ns: str = "",
    ) -> bool:
        """Delete a checkpoint payload or the whole thread, without raising Redis errors."""
        try:
            if checkpoint_id is None:
                self.delete_thread(thread_id)
                return True

            keys = [
                self._key_checkpoint(thread_id, checkpoint_id, ns),
                self._key_meta(thread_id, checkpoint_id, ns),
                self._key_parent(thread_id, checkpoint_id, ns),
                self._key_writes(thread_id, ns, checkpoint_id),
            ]
            self._redis.delete(*keys)
            if self._redis.get(self._key_latest(thread_id, ns)) == checkpoint_id:
                self._redis.delete(self._key_latest(thread_id, ns))
            return True
        except Exception as e:
            logger.warning("Redis checkpoint delete failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tuple(
        self,
        thread_id: str,
        ns: str,
        cp_id: str,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
        cp_raw = self._redis.get(self._key_checkpoint(thread_id, cp_id, ns))
        meta_raw = self._redis.get(self._key_meta(thread_id, cp_id, ns))
        if not cp_raw:
            return None

        checkpoint: Checkpoint = self.serde.loads_typed(json.loads(cp_raw))
        metadata = self.serde.loads_typed(json.loads(meta_raw)) if meta_raw else {}

        # Load channel blobs.
        channel_versions = checkpoint.get("channel_versions") or {}
        channel_values: dict[str, Any] = {}
        for ch, ver in channel_versions.items():
            blob_raw = self._redis.get(self._key_blob(thread_id, ns, ch, ver))
            if blob_raw:
                typed = json.loads(blob_raw)
                channel_values[ch] = self.serde.loads_typed(typed)

        checkpoint["channel_values"] = channel_values

        # Load pending writes.
        writes_key = self._key_writes(thread_id, ns, cp_id)
        raw_writes = self._redis.hgetall(writes_key)
        pending_writes = []
        for _field, blob_json in sorted(raw_writes.items()):
            task_id_w, ch, val_blob, _tp = json.loads(blob_json)
            pending_writes.append((task_id_w, ch, self.serde.loads_typed(val_blob)))

        parent_id = self._redis.get(self._key_parent(thread_id, cp_id, ns))
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            pending_writes=pending_writes,
            parent_config=parent_config,
        )
