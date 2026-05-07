#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Web 交互入口：提供浏览器聊天界面与 API。"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from cli import TravelFlowCLI
from config import SYSTEM_CONFIG, WEB_HOST, WEB_PORT
from context.long_term_memory import LongTermMemory
from utils.langsmith_setup import setup_langsmith_tracing

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
MAX_CHAT_MESSAGE_CHARS = int(SYSTEM_CONFIG["max_chat_message_chars"])
SESSION_IDLE_TIMEOUT_SEC = int(SYSTEM_CONFIG["session_idle_timeout_sec"])
SESSION_TITLE_MAX_CHARS = int(SYSTEM_CONFIG["session_title_max_chars"])
logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "ui", "description": "Browser UI entrypoint."},
    {"name": "chat", "description": "Chat execution APIs, including SSE streaming."},
    {"name": "sessions", "description": "Persisted chat-log session browsing and deletion."},
]

SSE_EVENT_SCHEMA = {
    "description": "Server-Sent Events stream. Each frame uses `event: <type>` and JSON `data:`.",
    "content": {
        "text/event-stream": {
            "schema": {"type": "string"},
            "examples": {
                "delta": {
                    "summary": "Assistant text chunk",
                    "value": 'event: delta\ndata: {"text":"你好"}\n\n',
                },
                "done": {
                    "summary": "Stream completion",
                    "value": 'event: done\ndata: {"session_id":"abcd1234","user_id":"default_user","latency_ms":1200}\n\n',
                },
                "error": {
                    "summary": "Stream error",
                    "value": 'event: error\ndata: {"error":"处理失败","detail":null,"code":"stream_error"}\n\n',
                },
            },
        }
    },
}


class ChatRequest(BaseModel):
    """Incoming chat payload for a single TravelFlow turn."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CHAT_MESSAGE_CHARS,
        description="User message. Whitespace-only messages are rejected after trimming.",
    )
    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    """Complete non-streaming chat reply with session identity and trace."""

    session_id: str
    user_id: str
    reply: str
    latency_ms: int
    trace: list[str] = []


class ErrorResponse(BaseModel):
    """Unified API error payload."""

    error: str
    detail: str | None = None
    code: str | None = None


class SessionState:
    """In-memory runtime state for one active Web session."""

    def __init__(self, cli: TravelFlowCLI, user_id: str, session_id: str):
        self.cli = cli
        self.user_id = user_id
        self.session_id = session_id
        now = datetime.now().isoformat()
        self.created_at = now
        self.last_active = now
        self.lock = asyncio.Lock()


class SessionListItem(BaseModel):
    """Compact session summary used by the session list endpoint."""

    session_id: str
    user_id: str
    created_at: str
    last_active: str
    message_count: int
    preview: str


class SessionMessage(BaseModel):
    """Rendered chat message returned when browsing a persisted session."""

    role: str
    content: str
    timestamp: str | None = None


class SessionDetail(BaseModel):
    """Full persisted session view including rendered messages."""

    session_id: str
    user_id: str
    created_at: str
    last_active: str
    messages: list[SessionMessage]


class DeleteSessionResponse(BaseModel):
    """Deletion result for one persisted chat-log session."""

    ok: bool
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_langsmith_tracing()
    app.state.long_term_memory = LongTermMemory(user_id="default_user", storage_path="data/memory")
    yield


app = FastAPI(title="TravelFlow 旅游出行助手 Web", openapi_tags=OPENAPI_TAGS, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_sessions: Dict[str, SessionState] = {}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error, "detail": None, "code": str(exc.status_code)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "detail": str(exc), "code": "500"},
    )


def _sse_frame(event: dict) -> str:
    event_type = str(event.get("type") or "message")
    payload = dict(event)
    payload.pop("type", None)
    if event_type == "error":
        message = payload.pop("error", payload.pop("message", "流式响应错误"))
        payload = {
            "error": str(message),
            "detail": payload.pop("detail", None),
            "code": str(payload.pop("code", "stream_error")),
        }
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


def _chunk_text(text: str, chunk_size: int = 2):
    """Yield small display chunks for token-like streaming in the browser."""
    buffer = ""
    for char in text:
        buffer += char
        if char in "\n。！？；，,.!?;:" or len(buffer) >= chunk_size:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _hydrate_short_term_memory(state: SessionState):
    """从长期记忆回填当前会话的最近消息，确保可续聊。"""
    try:
        long_term = state.cli.memory_manager.long_term
        short_term = state.cli.memory_manager.short_term

        history = long_term.get_chat_history(limit=None, session_id=state.session_id)
        if not history:
            return

        # 新建会话实例时 short_term 为空，回填最近 max_turns*2 条。
        max_messages = max(1, short_term.max_turns * 2)
        for msg in history[-max_messages:]:
            role = str(msg.get("role", "assistant"))
            content = str(msg.get("content", ""))
            metadata = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}

            # 对历史助手消息，优先使用可读 display，避免把结构化 JSON 直接注入上下文。
            if role == "assistant" and metadata.get("display"):
                content = str(metadata.get("display"))

            short_term.add_message(role, content, metadata)
    except Exception:
        logger.debug("Failed to hydrate short-term memory from persisted session history", exc_info=True)


def _get_or_create_metadata(memory: LongTermMemory, session_id: str, user_id: str) -> dict:
    memory.ensure_session_meta(session_id)
    meta_map = memory.get_session_meta_map()
    return meta_map.get(
        session_id,
        {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "preview": "",
        },
    )


def _update_session_meta(memory: LongTermMemory, session_id: str, user_id: str, preview: str = ""):
    memory.update_session_meta(session_id=session_id, preview=preview[:SESSION_TITLE_MAX_CHARS])


def _render_history_content(raw_msg: dict) -> str:
    metadata = raw_msg.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("display"):
        return str(metadata.get("display"))

    content = str(raw_msg.get("content", ""))
    if raw_msg.get("role") == "assistant":
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # 回放场景下优先抽取通用字段
                for key in ("answer", "message", "summary"):
                    if isinstance(data.get(key), str) and data.get(key).strip():
                        return data.get(key).strip()
        except Exception:
            logger.debug("Failed to parse assistant history content as JSON", exc_info=True)
    return content


async def _get_or_create_session(
    user_id: str,
    session_id: str | None,
    long_term_memory: LongTermMemory | None = None,
) -> tuple[str, SessionState]:
    current_session_id = session_id or str(uuid.uuid4())

    if current_session_id in _sessions:
        return current_session_id, _sessions[current_session_id]

    cli = TravelFlowCLI()
    await cli.initialize_system(user_id=user_id, interactive=False, session_id=current_session_id)
    if long_term_memory is not None:
        cli.memory_manager.long_term = long_term_memory
    state = SessionState(cli, user_id=user_id, session_id=current_session_id)
    _sessions[current_session_id] = state

    memory = cli.memory_manager.long_term
    _get_or_create_metadata(memory, current_session_id, user_id)
    _hydrate_short_term_memory(state)

    return current_session_id, state


@app.get("/", response_class=HTMLResponse, tags=["ui"], summary="Render the browser chat UI")
async def index() -> str:
    try:
        html_path = TEMPLATE_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=500, detail="前端页面不存在")
        return html_path.read_text(encoding="utf-8")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Run one chat turn and return a complete JSON response",
)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    try:
        text = req.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="message 不能为空")

        long_term_memory = request.app.state.long_term_memory
        session_id, state = await _get_or_create_session(req.user_id, req.session_id, long_term_memory)

        async with state.lock:
            started = time.time()
            reply, _, _, trace = await state.cli.process_query_for_web(text)

            latency_ms = int((time.time() - started) * 1000)
            state.last_active = datetime.now().isoformat()
            _update_session_meta(state.cli.memory_manager.long_term, session_id, req.user_id, preview=text)

        return ChatResponse(
            session_id=session_id,
            user_id=req.user_id,
            reply=reply,
            latency_ms=latency_ms,
            trace=trace,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


@app.post(
    "/api/chat/stream",
    tags=["chat"],
    summary="Run one chat turn and stream events with SSE",
    responses={200: SSE_EVENT_SCHEMA},
)
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message 不能为空")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        started = time.time()

        def on_runtime_event(message):
            try:
                if isinstance(message, dict):
                    queue.put_nowait(message)
                else:
                    queue.put_nowait({"type": "trace", "message": str(message)})
            except Exception:
                logger.debug("Failed to enqueue runtime event for stream response", exc_info=True)

        async def run_query():
            session_id = req.session_id
            state = None
            try:
                if not session_id:
                    queue.put_nowait({"type": "trace", "message": "正在创建新会话..."})

                session_id, state = await _get_or_create_session(
                    req.user_id,
                    req.session_id,
                    request.app.state.long_term_memory,
                )

                async with state.lock:
                    state.cli.set_runtime_event_callback(on_runtime_event)
                    reply, result_data, _, _ = await state.cli.process_query_for_web(text)
                    state.last_active = datetime.now().isoformat()
                    _update_session_meta(state.cli.memory_manager.long_term, session_id, req.user_id, preview=text)

                for chunk in _chunk_text(reply or "未返回结果"):
                    queue.put_nowait({"type": "delta", "text": chunk})
                    await asyncio.sleep(0.01)

                suggestions = []
                if isinstance(result_data, dict):
                    suggestions = result_data.get("suggested_replies") or []
                if suggestions:
                    queue.put_nowait({"type": "suggestions", "items": suggestions})

                input_requests = []
                if isinstance(result_data, dict):
                    input_requests = result_data.get("input_requests") or []
                if input_requests:
                    queue.put_nowait({"type": "input_requests", "items": input_requests})

                latency_ms = int((time.time() - started) * 1000)
                queue.put_nowait({
                    "type": "done",
                    "session_id": session_id,
                    "user_id": req.user_id,
                    "latency_ms": latency_ms,
                })
            except Exception as e:
                logger.warning("Streaming chat query failed: %s", e, exc_info=True)
                queue.put_nowait({"type": "error", "error": str(e), "detail": None, "code": "stream_error"})
            finally:
                if state:
                    state.cli.set_runtime_event_callback(None)
                queue.put_nowait({"type": "eof"})

        task = asyncio.create_task(run_query())

        try:
            while True:
                event = await queue.get()
                if event.get("type") == "eof":
                    break
                yield _sse_frame(event)
        finally:
            await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/sessions",
    response_model=list[SessionListItem],
    tags=["sessions"],
    summary="List persisted chat-log sessions for a user",
)
async def list_sessions(request: Request, user_id: str = "default_user") -> list[SessionListItem]:
    try:
        memory = request.app.state.long_term_memory
        chat_history = memory.get_chat_history(limit=None)
        session_meta = memory.get_session_meta_map()

        grouped: Dict[str, list] = {}
        for msg in chat_history:
            sid = msg.get("session_id")
            if not sid:
                continue
            grouped.setdefault(sid, []).append(msg)

        items: list[SessionListItem] = []
        for sid, messages in grouped.items():
            meta = session_meta.get(sid, {}) if isinstance(session_meta, dict) else {}
            preview = meta.get("preview", "")
            if not preview:
                for m in messages:
                    if m.get("role") == "user" and m.get("content"):
                        preview = str(m.get("content"))[:SESSION_TITLE_MAX_CHARS]
                        break
            created_at = meta.get("created_at") or (messages[0].get("timestamp") if messages else datetime.now().isoformat())
            last_active = meta.get("last_active") or (messages[-1].get("timestamp") if messages else created_at)
            items.append(
                SessionListItem(
                    session_id=sid,
                    user_id=user_id,
                    created_at=created_at,
                    last_active=last_active,
                    message_count=len(messages),
                    preview=preview or "新会话",
                )
            )

        items.sort(key=lambda item: item.last_active, reverse=True)
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionDetail,
    tags=["sessions"],
    summary="Get rendered messages for one chat-log session",
)
async def get_session_detail(
    session_id: str,
    request: Request,
    user_id: str = "default_user",
) -> SessionDetail:
    try:
        memory = request.app.state.long_term_memory
        messages = memory.get_chat_history(limit=None, session_id=session_id)
        session_meta = memory.get_session_meta_map()
        meta = session_meta.get(session_id, {}) if isinstance(session_meta, dict) else {}

        if not messages:
            raise HTTPException(status_code=404, detail="会话不存在")

        rendered = [
            SessionMessage(
                role=str(msg.get("role", "assistant")),
                content=_render_history_content(msg),
                timestamp=msg.get("timestamp"),
            )
            for msg in messages
        ]

        created_at = meta.get("created_at") or messages[0].get("timestamp") or datetime.now().isoformat()
        last_active = meta.get("last_active") or messages[-1].get("timestamp") or created_at

        return SessionDetail(
            session_id=session_id,
            user_id=user_id,
            created_at=created_at,
            last_active=last_active,
            messages=rendered,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


@app.delete(
    "/api/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["sessions"],
    summary="Delete one persisted chat-log session",
)
async def delete_session(
    session_id: str,
    request: Request,
    user_id: str = "default_user",
) -> DeleteSessionResponse:
    try:
        memory = request.app.state.long_term_memory
        deleted_count = memory.delete_session(session_id)
        if deleted_count <= 0:
            raise HTTPException(status_code=404, detail="会话不存在")

        if session_id in _sessions:
            _sessions.pop(session_id, None)

        return DeleteSessionResponse(ok=True, session_id=session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


def main() -> None:
    import uvicorn

    uvicorn.run("web.app:app", host=WEB_HOST, port=WEB_PORT, reload=False)


if __name__ == "__main__":
    main()
