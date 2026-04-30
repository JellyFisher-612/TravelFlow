"""Workflow skill primitives used before LLM intent recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillMatch:
    """A deterministic workflow match owned by a workflow skill."""

    skill_name: str
    confidence: float
    reason: str
    intention_data: Dict[str, Any]
    workflow_plan: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)


class WorkflowSkill:
    """Base class for common business workflows."""

    name = "workflow"
    priority = 100

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        raise NotImplementedError
