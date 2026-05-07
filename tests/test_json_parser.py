from __future__ import annotations

import unittest

from utils.json_parser import robust_json_parse


class RobustJsonParseTests(unittest.TestCase):
    def test_parse_valid_json_string(self):
        self.assertEqual({"city": "杭州", "days": 3}, robust_json_parse('{"city": "杭州", "days": 3}'))

    def test_parse_json_embedded_in_markdown_code_block(self):
        text = """```json
{"intent": "plan", "entities": {"destination": "上海"}}
```"""

        self.assertEqual(
            {"intent": "plan", "entities": {"destination": "上海"}},
            robust_json_parse(text),
        )

    def test_parse_json_with_trailing_commas(self):
        text = '{"destination": "北京", "interests": ["博物馆", "胡同",],}'

        self.assertEqual(
            {"destination": "北京", "interests": ["博物馆", "胡同"]},
            robust_json_parse(text),
        )

    def test_parse_malformed_json_with_fallback(self):
        fallback = {"ok": False}

        self.assertIs(fallback, robust_json_parse("{not valid json", fallback=fallback))

    def test_handle_empty_string_input_with_fallback(self):
        fallback = {"empty": True}

        self.assertIs(fallback, robust_json_parse("", fallback=fallback))

    def test_handle_empty_string_input_without_fallback_raises(self):
        with self.assertRaises(ValueError):
            robust_json_parse("")

    def test_handle_none_input_with_fallback(self):
        fallback = {"none": True}

        self.assertIs(fallback, robust_json_parse(None, fallback=fallback))


if __name__ == "__main__":
    unittest.main()
