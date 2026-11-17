"""Example: healthcare agent replans on contradictory guidance.

Run: ``python examples/demo_healthcare.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.models import Observation
from plan_change_attribution.planner import plan

initial_obs = []
new_obs = [Observation(
    "contra-G1", "contradiction", "guideline-db",
    "Guideline G1 contradicts the retrieved answer.", 10,
    {"guideline": "G1"})]

old = plan("healthcare", initial_obs)
new = plan("healthcare", initial_obs + new_obs)
report = explain_replan(
    "healthcare", initial_obs + new_obs, {}, old, new, new_obs)
print(report.summary())
