from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from tests.fakes import ExplodingLLM


def load_event_collection_agent_class():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "agent_langchain" / ".claude" / "skills" / "event-collection" / "script" / "agent.py"
    spec = importlib.util.spec_from_file_location("test_event_collection_agent", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EventCollectionAgent


class EventCollectionRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_range_input_is_extracted_without_calling_llm(self):
        agent_cls = load_event_collection_agent_class()
        model = ExplodingLLM()
        agent = agent_cls(model=model)

        result = await agent.run(
            {
                "context": {
                    "rewritten_query": "本次行程预算选择舒适型，住宿每晚300到600元，兼顾体验和性价比。"
                }
            }
        )

        self.assertEqual("舒适型", result["budget_level"])
        self.assertEqual(300, result["lodging_budget_per_night_min"])
        self.assertEqual(600, result["lodging_budget_per_night_max"])
        self.assertIsNone(result["lodging_budget_per_night"])
        self.assertIn("origin", result["missing_info"])
        self.assertIn("destination", result["missing_info"])
        self.assertNotIn("budget_level", result["missing_info"])
        self.assertEqual([], model.calls)


if __name__ == "__main__":
    unittest.main()
