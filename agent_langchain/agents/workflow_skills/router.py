"""Router for deterministic workflow skills."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from agents.workflow_skills.base import SkillMatch, WorkflowSkill
from agents.workflow_skills.common import DEFAULT_WORKFLOW_SKILLS

logger = logging.getLogger(__name__)


class SkillRouter:
    """Match high-frequency business workflows before LLM intent fallback."""

    def __init__(self, skills: Optional[Iterable[WorkflowSkill]] = None, min_confidence: float = 0.5):
        self.skills = sorted(list(skills or DEFAULT_WORKFLOW_SKILLS), key=lambda item: item.priority)
        self.min_confidence = min_confidence

    def match(self, state: Dict[str, Any]) -> Optional[SkillMatch]:
        for skill in self.skills:
            try:
                match = skill.match(state)
            except Exception:
                logger.debug("Workflow skill %s failed during match", getattr(skill, "name", skill), exc_info=True)
                continue
            if not match or match.confidence < self.min_confidence:
                continue
            return match

        return None
