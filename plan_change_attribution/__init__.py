"""Plan-Change Attribution — explain WHY an agent's plan changed.

For a Plan-and-Execute agent, attribute each dropped/added step in a
replan to the single triggering observation that caused it, by ablating
that observation in a replay of the planner.
"""

__version__ = "0.2.0"

from plan_change_attribution.models import Change, Observation, Plan, Step, Trace
from plan_change_attribution.attribution import AttributionReport, explain_replan
from plan_change_attribution.stability import (
    StabilityReport,
    TriggerStability,
    score_stability,
)

__all__ = [
    "__version__",
    "AttributionReport",
    "Change",
    "Observation",
    "Plan",
    "StabilityReport",
    "Step",
    "Trace",
    "TriggerStability",
    "explain_replan",
    "score_stability",
]
