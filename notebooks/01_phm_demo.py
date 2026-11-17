"""Notebook 01 — PHM demo: a sensor failure replans the agent.

Run: ``python notebooks/01_phm_demo.py``
Shows the initial plan, the replanned plan, and the attribution that
names the sensor-failure observation as the single trigger.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.scenarios import build_suite

sc = {s.name: s for s in build_suite()}["phm-vibration-failure-0"]
report = explain_replan(
    sc.goal, sc.all_observations(), sc.context,
    sc.old_plan(), sc.new_plan(), sc.candidates())

print("Initial plan: ", [s.id for s in report.old_plan.steps])
print("Replanned    : ", [s.id for s in report.new_plan.steps])
print(report.summary())
print()
print("Candidate evidence (observation -> changes repaired):")
for ca in report.per_change[:1]:
    for ts in ca.ranked_triggers:
        print(f"  {ts.observation.id:20s} score {ts.score}")
