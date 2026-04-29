from __future__ import annotations

import unittest
from unittest.mock import patch

from utils import langchain_runtime


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LangChainRuntimeTests(unittest.TestCase):
    def test_build_chat_model_disables_environment_proxy_for_llm_http_clients(self):
        client_kwargs = []
        async_client_kwargs = []

        class FakeHttpClient:
            def __init__(self, **kwargs):
                client_kwargs.append(kwargs)

        class FakeAsyncHttpClient:
            def __init__(self, **kwargs):
                async_client_kwargs.append(kwargs)

        with patch.object(langchain_runtime, "ChatOpenAI", FakeChatOpenAI), patch.object(
            langchain_runtime.httpx, "Client", FakeHttpClient
        ), patch.object(langchain_runtime.httpx, "AsyncClient", FakeAsyncHttpClient):
            model = langchain_runtime.build_chat_model()

        self.assertIsInstance(model, FakeChatOpenAI)
        self.assertEqual(False, client_kwargs[0]["trust_env"])
        self.assertEqual(False, async_client_kwargs[0]["trust_env"])
        self.assertIs(model.kwargs["http_client"].__class__, FakeHttpClient)
        self.assertIs(model.kwargs["http_async_client"].__class__, FakeAsyncHttpClient)


if __name__ == "__main__":
    unittest.main()
