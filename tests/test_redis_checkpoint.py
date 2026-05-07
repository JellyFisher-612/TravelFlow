from __future__ import annotations

import fnmatch
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "agent_langchain" / "context" / "redis_checkpoint.py"


@dataclass
class FakeCheckpointTuple:
    config: dict
    checkpoint: dict
    metadata: dict
    pending_writes: list
    parent_config: dict | None


class FakeBaseCheckpointSaver:
    def __init__(self, serde=None):
        self.serde = serde

    def __class_getitem__(cls, item):
        return cls


class FakeJsonPlusSerializer:
    def dumps_typed(self, value):
        return ["json", value]

    def loads_typed(self, typed):
        return typed[1]


class FakePipeline:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def setex(self, key, ttl, value):
        self.redis_client.setex(key, ttl, value)
        return self

    def hset(self, key, field, value):
        self.redis_client.hset(key, field, value)
        return self

    def expire(self, key, ttl):
        self.redis_client.expire(key, ttl)
        return self

    def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.hashes = {}
        self.expirations = {}

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.expirations[key] = ttl
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            deleted += int(key in self.store or key in self.hashes)
            self.store.pop(key, None)
            self.hashes.pop(key, None)
            self.expirations.pop(key, None)
        return deleted

    def scan(self, cursor, match=None, count=None):
        keys = list(self.store) + list(self.hashes)
        if match is not None:
            keys = [key for key in keys if fnmatch.fnmatch(key, match)]
        return 0, keys

    def pipeline(self):
        return FakePipeline(self)

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def expire(self, key, ttl):
        self.expirations[key] = ttl
        return True


class FailingRedis(FakeRedis):
    def setex(self, key, ttl, value):
        raise ConnectionError("redis down")

    def get(self, key):
        raise ConnectionError("redis down")

    def delete(self, *keys):
        raise ConnectionError("redis down")

    def scan(self, cursor, match=None, count=None):
        raise ConnectionError("redis down")


def load_redis_checkpoint_module(fake_redis):
    redis_module = types.SimpleNamespace(
        Redis=types.SimpleNamespace(from_url=lambda redis_url, decode_responses=True: fake_redis)
    )
    checkpoint_base_module = types.SimpleNamespace(
        BaseCheckpointSaver=FakeBaseCheckpointSaver,
        ChannelVersions=dict,
        Checkpoint=dict,
        CheckpointMetadata=dict,
        CheckpointTuple=FakeCheckpointTuple,
        RunnableConfig=dict,
        WRITES_IDX_MAP={},
        get_checkpoint_id=lambda config: (config or {}).get("configurable", {}).get("checkpoint_id"),
        get_checkpoint_metadata=lambda config, metadata: metadata,
    )
    jsonplus_module = types.SimpleNamespace(JsonPlusSerializer=FakeJsonPlusSerializer)

    with patch.dict(
        sys.modules,
        {
            "redis": redis_module,
            "langgraph": types.ModuleType("langgraph"),
            "langgraph.checkpoint": types.ModuleType("langgraph.checkpoint"),
            "langgraph.checkpoint.base": checkpoint_base_module,
            "langgraph.serde": types.ModuleType("langgraph.serde"),
            "langgraph.serde.jsonplus": jsonplus_module,
        },
    ):
        spec = importlib.util.spec_from_file_location("redis_checkpoint_under_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class RedisCheckpointSaverTests(unittest.TestCase):
    def make_saver(self, fake_redis):
        module = load_redis_checkpoint_module(fake_redis)
        return module.RedisCheckpointSaver("redis://example.test/0", key_prefix="test:checkpoint", ttl_sec=60)

    def test_save_checkpoint_stores_data_correctly(self):
        fake_redis = FakeRedis()
        saver = self.make_saver(fake_redis)

        saved = saver.save_checkpoint("thread-1", "cp-1", {"state": {"destination": "杭州"}})

        self.assertTrue(saved)
        self.assertEqual('{"state": {"destination": "杭州"}}', fake_redis.store["test:checkpoint:thread-1:cp:cp-1"])
        self.assertEqual("cp-1", fake_redis.store["test:checkpoint:thread-1:latest"])

    def test_load_checkpoint_retrieves_stored_data(self):
        fake_redis = FakeRedis()
        saver = self.make_saver(fake_redis)
        saver.save_checkpoint("thread-1", "cp-1", {"state": {"days": 3}})

        self.assertEqual({"state": {"days": 3}}, saver.load_checkpoint("thread-1", "cp-1"))
        self.assertEqual({"state": {"days": 3}}, saver.load_checkpoint("thread-1"))

    def test_load_checkpoint_returns_none_for_nonexistent_key(self):
        saver = self.make_saver(FakeRedis())

        self.assertIsNone(saver.load_checkpoint("missing-thread", "missing-cp"))
        self.assertIsNone(saver.load_checkpoint("missing-thread"))

    def test_delete_checkpoint_removes_data(self):
        fake_redis = FakeRedis()
        saver = self.make_saver(fake_redis)
        saver.save_checkpoint("thread-1", "cp-1", {"state": {"budget": "经济型"}})

        deleted = saver.delete_checkpoint("thread-1", "cp-1")

        self.assertTrue(deleted)
        self.assertIsNone(saver.load_checkpoint("thread-1", "cp-1"))
        self.assertIsNone(saver.load_checkpoint("thread-1"))

    def test_handles_redis_connection_errors_gracefully(self):
        saver = self.make_saver(FailingRedis())

        self.assertFalse(saver.save_checkpoint("thread-1", "cp-1", {"state": {}}))
        self.assertIsNone(saver.load_checkpoint("thread-1", "cp-1"))
        self.assertFalse(saver.delete_checkpoint("thread-1", "cp-1"))


if __name__ == "__main__":
    unittest.main()
