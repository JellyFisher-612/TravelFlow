from __future__ import annotations

import unittest

import config


class LLMConfigTests(unittest.TestCase):
    def test_mimo_token_plan_key_uses_cn_token_plan_url_by_default(self):
        resolved = config._resolve_llm_config({"MIMO_API_KEY": "tp-test"})

        self.assertEqual("tp-test", resolved["api_key"])
        self.assertEqual("mimo-v2.5-pro", resolved["model_name"])
        self.assertEqual("https://token-plan-cn.xiaomimimo.com/v1", resolved["base_url"])

    def test_mimo_pay_as_you_go_key_uses_official_url_by_default(self):
        resolved = config._resolve_llm_config({"MIMO_API_KEY": "sk-test"})

        self.assertEqual("sk-test", resolved["api_key"])
        self.assertEqual("mimo-v2.5-pro", resolved["model_name"])
        self.assertEqual("https://api.xiaomimimo.com/v1", resolved["base_url"])

    def test_deepseek_remains_fallback_when_mimo_key_is_absent(self):
        resolved = config._resolve_llm_config({"DEEPSEEK_API_KEY": "ds-test"})

        self.assertEqual("ds-test", resolved["api_key"])
        self.assertEqual("deepseek-v4-flash", resolved["model_name"])
        self.assertEqual("https://api.deepseek.com/v1", resolved["base_url"])

    def test_explicit_llm_env_overrides_provider_defaults(self):
        resolved = config._resolve_llm_config(
            {
                "MIMO_API_KEY": "tp-test",
                "LLM_MODEL_NAME": "custom-model",
                "LLM_BASE_URL": "https://example.test/v1",
                "LLM_TEMPERATURE": "0.2",
                "LLM_MAX_TOKENS": "256",
            }
        )

        self.assertEqual("custom-model", resolved["model_name"])
        self.assertEqual("https://example.test/v1", resolved["base_url"])
        self.assertEqual(0.2, resolved["temperature"])
        self.assertEqual(256, resolved["max_tokens"])

    def test_mimo_key_overrides_legacy_deepseek_defaults(self):
        resolved = config._resolve_llm_config(
            {
                "MIMO_API_KEY": "tp-test",
                "LLM_MODEL_NAME": "deepseek-v4-flash",
                "LLM_BASE_URL": "https://api.deepseek.com/v1",
            }
        )

        self.assertEqual("mimo-v2.5-pro", resolved["model_name"])
        self.assertEqual("https://token-plan-cn.xiaomimimo.com/v1", resolved["base_url"])


if __name__ == "__main__":
    unittest.main()
