from __future__ import annotations

import unittest

from pydantic import ValidationError

from config import SYSTEM_CONFIG
from web.app import ChatRequest, app


class WebApiContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
