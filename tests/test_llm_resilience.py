from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, call, patch

from agent_langchain.utils import llm_resilience


class RetryWithBackoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_on_transient_timeout_error(self):
        attempts = []

        async def flaky_call():
            attempts.append("called")
            if len(attempts) == 1:
                raise TimeoutError("timed out")
            return "ok"

        with patch.object(llm_resilience.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            result = await llm_resilience.retry_with_backoff(
                flaky_call,
                max_retries=3,
                base_delay_sec=0.5,
                jitter=False,
            )

        self.assertEqual("ok", result)
        self.assertEqual(2, len(attempts))
        sleep.assert_awaited_once_with(0.5)

    async def test_retry_stops_after_max_retries_attempts(self):
        attempts = []

        async def always_fails():
            attempts.append("called")
            raise TimeoutError("timeout")

        with patch.object(llm_resilience.asyncio, "sleep", new_callable=AsyncMock):
            with self.assertRaises(TimeoutError):
                await llm_resilience.retry_with_backoff(
                    always_fails,
                    max_retries=2,
                    base_delay_sec=0.1,
                    jitter=False,
                )

        self.assertEqual(3, len(attempts))

    async def test_exponential_backoff_delays_increase_correctly(self):
        attempts = []

        async def succeeds_after_two_retries():
            attempts.append("called")
            if len(attempts) < 3:
                raise ConnectionError("503 service unavailable")
            return "ok"

        with patch.object(llm_resilience.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await llm_resilience.retry_with_backoff(
                succeeds_after_two_retries,
                max_retries=3,
                base_delay_sec=1.0,
                max_delay_sec=10.0,
                jitter=False,
            )

        self.assertEqual([call(1.0), call(2.0)], sleep.await_args_list)

    async def test_non_retryable_errors_are_not_retried(self):
        attempts = []

        async def invalid_request():
            attempts.append("called")
            raise ValueError("bad prompt")

        with patch.object(llm_resilience.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(ValueError):
                await llm_resilience.retry_with_backoff(invalid_request, max_retries=3)

        self.assertEqual(1, len(attempts))
        sleep.assert_not_called()


class HealthCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_check_returns_true_when_llm_is_reachable(self):
        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def ainvoke(self, prompt):
                return types.SimpleNamespace(content="1")

        fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
        with patch.dict(sys.modules, {"langchain_openai": fake_module}):
            ok, message = await llm_resilience.run_health_check(
                base_url="https://llm.example.test/v1",
                api_key="test-key",
                model_name="test-model",
                timeout_sec=1.0,
            )

        self.assertTrue(ok)
        self.assertEqual("ok", message)

    async def test_health_check_returns_false_when_llm_is_unreachable(self):
        class FailingChatOpenAI:
            def __init__(self, **kwargs):
                pass

            async def ainvoke(self, prompt):
                raise TimeoutError("llm timeout")

        fake_module = types.SimpleNamespace(ChatOpenAI=FailingChatOpenAI)
        with patch.dict(sys.modules, {"langchain_openai": fake_module}):
            ok, message = await llm_resilience.run_health_check(
                base_url="https://llm.example.test/v1",
                api_key="test-key",
                model_name="test-model",
                timeout_sec=1.0,
            )

        self.assertFalse(ok)
        self.assertIn("llm timeout", message)


if __name__ == "__main__":
    unittest.main()
