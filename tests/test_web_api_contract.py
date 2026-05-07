from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from config import SYSTEM_CONFIG
from web import app as web_app
from web.app import ChatRequest, app


class FakeShortTermMemory:
    def __init__(self) -> None:
        self.max_turns = 5
        self.messages = []

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        self.messages.append({"role": role, "content": content, "metadata": metadata or {}})


class FakeLongTermMemory:
    def __init__(self, user_id: str = "default_user", storage_path: str | None = None) -> None:
        self.user_id = user_id
        self.storage_path = storage_path
        self.messages: list[dict] = []
        self.session_meta: dict[str, dict] = {}

    def ensure_session_meta(self, session_id: str) -> None:
        if session_id in self.session_meta:
            return
        now = datetime.now().isoformat()
        self.session_meta[session_id] = {
            "session_id": session_id,
            "user_id": self.user_id,
            "created_at": now,
            "last_active": now,
            "preview": "",
        }

    def update_session_meta(self, session_id: str, preview: str = "") -> None:
        self.ensure_session_meta(session_id)
        self.session_meta[session_id]["last_active"] = datetime.now().isoformat()
        if preview:
            self.session_meta[session_id]["preview"] = preview[:80]

    def get_session_meta_map(self) -> dict[str, dict]:
        return self.session_meta

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
        session_id: str | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        if session_id:
            self.ensure_session_meta(session_id)
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": now,
                "session_id": session_id,
                "metadata": metadata or {},
            }
        )

    def get_chat_history(self, limit: int | None = None, session_id: str | None = None) -> list[dict]:
        messages = self.messages
        if session_id:
            messages = [msg for msg in messages if msg.get("session_id") == session_id]
        if limit:
            return messages[-limit:]
        return list(messages)

    def delete_session(self, session_id: str) -> int:
        original_count = len(self.messages)
        self.messages = [msg for msg in self.messages if msg.get("session_id") != session_id]
        self.session_meta.pop(session_id, None)
        return original_count - len(self.messages)


class FakeMemoryManager:
    def __init__(self, long_term: FakeLongTermMemory | None = None) -> None:
        self.long_term = long_term or FakeLongTermMemory()
        self.short_term = FakeShortTermMemory()


class FakeTravelFlowCLI:
    should_fail = False

    def __init__(self) -> None:
        self.user_id = "default_user"
        self.session_id = None
        self.memory_manager = FakeMemoryManager()
        self.runtime_event_callback = None

    async def initialize_system(
        self,
        user_id: str | None = None,
        interactive: bool = True,
        session_id: str | None = None,
    ) -> None:
        self.user_id = user_id or "default_user"
        self.session_id = session_id
        self.memory_manager = FakeMemoryManager(FakeLongTermMemory(self.user_id))

    def set_runtime_event_callback(self, callback) -> None:
        self.runtime_event_callback = callback

    async def process_query_for_web(self, user_input: str):
        if self.should_fail:
            raise RuntimeError("fake processing failure")
        if self.runtime_event_callback:
            self.runtime_event_callback({"type": "trace", "message": "fake trace"})

        reply = f"回复: {user_input}"
        session_id = self.session_id or "generated-session"
        long_term = self.memory_manager.long_term
        long_term.add_message("user", user_input, session_id=session_id)
        long_term.add_message("assistant", reply, metadata={"display": reply}, session_id=session_id)
        return (
            reply,
            {"suggested_replies": ["继续"], "input_requests": []},
            None,
            ["fake trace"],
        )


@app.get("/__test__/boom", include_in_schema=False)
async def _test_boom():
    raise RuntimeError("boom")


class WebApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = FakeLongTermMemory()
        FakeTravelFlowCLI.should_fail = False
        web_app._sessions.clear()
        web_app.app.state.long_term_memory = self.memory
        self.cli_patcher = patch("web.app.TravelFlowCLI", FakeTravelFlowCLI)
        self.memory_patcher = patch("web.app.LongTermMemory", return_value=self.memory)
        self.cli_patcher.start()
        self.memory_patcher.start()
        self.client = TestClient(web_app.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self.client.close()
        self.cli_patcher.stop()
        self.memory_patcher.stop()
        web_app._sessions.clear()

    def test_chat_request_rejects_oversized_message(self):
        too_long = "x" * (int(SYSTEM_CONFIG["max_chat_message_chars"]) + 1)
        with self.assertRaises(ValidationError):
            ChatRequest(message=too_long)

    def test_openapi_documents_sse_stream(self):
        schema = app.openapi()
        stream = schema["paths"]["/api/chat/stream"]["post"]

        self.assertEqual(["chat"], stream["tags"])
        self.assertEqual("Run one chat turn and stream events with SSE", stream["summary"])
        content = stream["responses"]["200"]["content"]
        self.assertIn("text/event-stream", content)
        examples = content["text/event-stream"]["examples"]
        self.assertIn("delta", examples)
        self.assertIn("done", examples)

    def test_index_returns_html(self):
        """首页返回 HTML 200"""
        response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<!doctype html>", response.text.lower())

    def test_index_contains_chat_form(self):
        """首页包含聊天表单元素"""
        response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn('id="input"', response.text)
        self.assertIn('id="send"', response.text)
        self.assertIn("/api/chat/stream", response.text)

    def test_chat_valid_message_returns_200(self):
        """有效消息返回 200 + ChatResponse schema"""
        response = self.client.post("/api/chat", json={"message": "你好"})

        self.assertEqual(200, response.status_code)
        self.assertEqual("default_user", response.json()["user_id"])
        self.assertEqual("回复: 你好", response.json()["reply"])

    def test_chat_empty_message_returns_400(self):
        """空消息返回 400"""
        response = self.client.post("/api/chat", json={"message": "   "})

        self.assertEqual(400, response.status_code)
        self.assertEqual({"error": "message 不能为空", "detail": None, "code": "400"}, response.json())

    def test_chat_oversized_message_returns_422(self):
        """超长消息返回 422 (Pydantic validation)"""
        too_long = "x" * (int(SYSTEM_CONFIG["max_chat_message_chars"]) + 1)
        response = self.client.post("/api/chat", json={"message": too_long})

        self.assertEqual(422, response.status_code)

    def test_chat_response_has_required_fields(self):
        """响应包含 session_id, user_id, reply, latency_ms"""
        response = self.client.post("/api/chat", json={"message": "帮我规划行程"})

        self.assertEqual(200, response.status_code)
        body = response.json()
        for field in ("session_id", "user_id", "reply", "latency_ms"):
            self.assertIn(field, body)
        self.assertIsInstance(body["latency_ms"], int)

    def test_chat_creates_new_session(self):
        """不传 session_id 时自动创建新会话"""
        response = self.client.post("/api/chat", json={"message": "新会话"})

        self.assertEqual(200, response.status_code)
        session_id = response.json()["session_id"]
        self.assertTrue(session_id)
        self.assertIn(session_id, web_app._sessions)

    def test_chat_reuses_existing_session(self):
        """传 session_id 时复用已有会话"""
        first = self.client.post("/api/chat", json={"message": "第一轮", "session_id": "session-a"})
        second = self.client.post("/api/chat", json={"message": "第二轮", "session_id": "session-a"})

        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual("session-a", first.json()["session_id"])
        self.assertEqual("session-a", second.json()["session_id"])
        self.assertEqual(1, len(web_app._sessions))

    def test_stream_returns_sse_content_type(self):
        """流式端点返回 text/event-stream"""
        with self.client.stream("POST", "/api/chat/stream", json={"message": "你好"}) as response:
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

    def test_stream_emits_delta_events(self):
        """流式响应包含 event: delta 帧"""
        response = self.client.post("/api/chat/stream", json={"message": "你好"})

        self.assertEqual(200, response.status_code)
        self.assertIn("event: delta", response.text)

    def test_stream_emits_done_event(self):
        """流式响应最后包含 event: done 帧"""
        response = self.client.post("/api/chat/stream", json={"message": "你好"})

        self.assertEqual(200, response.status_code)
        frames = [frame for frame in response.text.strip().split("\n\n") if frame]
        self.assertTrue(frames[-1].startswith("event: done"))

    def test_stream_error_emits_error_event(self):
        """处理失败时发送 event: error 帧而非 HTTP 500"""
        FakeTravelFlowCLI.should_fail = True
        response = self.client.post("/api/chat/stream", json={"message": "触发错误"})

        self.assertEqual(200, response.status_code)
        self.assertIn("event: error", response.text)
        self.assertIn('"code": "stream_error"', response.text)

    def test_stream_empty_message_returns_400(self):
        """空消息在流式端点也返回 400"""
        response = self.client.post("/api/chat/stream", json={"message": "   "})

        self.assertEqual(400, response.status_code)
        self.assertEqual("message 不能为空", response.json()["error"])

    def test_list_sessions_returns_array(self):
        """会话列表返回数组"""
        response = self.client.get("/api/sessions")

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json(), list)

    def test_list_sessions_empty_initially(self):
        """初始状态返回空数组"""
        response = self.client.get("/api/sessions")

        self.assertEqual([], response.json())

    def test_list_sessions_after_chat(self):
        """发送消息后会话列表包含该会话"""
        chat = self.client.post("/api/chat", json={"message": "记录会话"})
        session_id = chat.json()["session_id"]
        response = self.client.get("/api/sessions")

        self.assertEqual(200, response.status_code)
        self.assertEqual([session_id], [item["session_id"] for item in response.json()])

    def test_get_session_detail_returns_200(self):
        """存在的会话返回 200 + 详情"""
        chat = self.client.post("/api/chat", json={"message": "查看详情"})
        session_id = chat.json()["session_id"]
        response = self.client.get(f"/api/sessions/{session_id}")

        self.assertEqual(200, response.status_code)
        self.assertEqual(session_id, response.json()["session_id"])

    def test_get_session_detail_returns_404(self):
        """不存在的会话返回 404"""
        response = self.client.get("/api/sessions/missing-session")

        self.assertEqual(404, response.status_code)
        self.assertEqual("会话不存在", response.json()["error"])

    def test_get_session_detail_contains_messages(self):
        """会话详情包含消息历史"""
        chat = self.client.post("/api/chat", json={"message": "历史消息"})
        session_id = chat.json()["session_id"]
        response = self.client.get(f"/api/sessions/{session_id}")

        self.assertEqual(200, response.status_code)
        messages = response.json()["messages"]
        self.assertEqual(["user", "assistant"], [msg["role"] for msg in messages])
        self.assertEqual("历史消息", messages[0]["content"])
        self.assertEqual("回复: 历史消息", messages[1]["content"])

    def test_delete_session_returns_200(self):
        """删除存在的会话返回 200"""
        chat = self.client.post("/api/chat", json={"message": "删除会话"})
        session_id = chat.json()["session_id"]
        response = self.client.delete(f"/api/sessions/{session_id}")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"ok": True, "session_id": session_id}, response.json())

    def test_delete_session_returns_404(self):
        """删除不存在的会话返回 404"""
        response = self.client.delete("/api/sessions/missing-session")

        self.assertEqual(404, response.status_code)
        self.assertEqual("会话不存在", response.json()["error"])

    def test_delete_session_removes_from_list(self):
        """删除后会话不再出现在列表中"""
        chat = self.client.post("/api/chat", json={"message": "待删除"})
        session_id = chat.json()["session_id"]
        delete_response = self.client.delete(f"/api/sessions/{session_id}")
        list_response = self.client.get("/api/sessions")

        self.assertEqual(200, delete_response.status_code)
        self.assertEqual([], list_response.json())

    def test_cors_preflight_returns_200(self):
        """OPTIONS 预检请求返回 200"""
        response = self.client.options(
            "/api/chat",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"},
        )

        self.assertEqual(200, response.status_code)

    def test_cors_headers_present(self):
        """响应包含 Access-Control-Allow-Origin 头"""
        response = self.client.get("/", headers={"Origin": "http://example.com"})

        self.assertEqual(200, response.status_code)
        self.assertIn("access-control-allow-origin", response.headers)

    def test_unhandled_exception_returns_500_with_error_schema(self):
        """未处理异常返回统一错误格式 {"error": ..., "detail": ..., "code": ...}"""
        response = self.client.get("/__test__/boom")

        self.assertEqual(500, response.status_code)
        self.assertEqual({"error": "内部服务器错误", "detail": "boom", "code": "500"}, response.json())

    def test_404_returns_error_schema(self):
        """404 返回统一错误格式"""
        response = self.client.get("/not-found")

        self.assertEqual(404, response.status_code)
        self.assertEqual({"error": "Not Found", "detail": None, "code": "404"}, response.json())


if __name__ == "__main__":
    unittest.main()
