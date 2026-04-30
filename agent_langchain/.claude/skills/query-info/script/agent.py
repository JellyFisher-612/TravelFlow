"""Compatibility entry for the legacy query-info skill path.

The runnable search agent now lives in ``agents.search_agent``. Keep this
module so old tests or external imports that load the historical skill script
continue to work while ``.claude/skills`` remains documentation-oriented.
"""

from agents.search_agent import InformationQueryAgent, SearchPlanOutput, SummaryOutput

__all__ = ["InformationQueryAgent", "SearchPlanOutput", "SummaryOutput"]
