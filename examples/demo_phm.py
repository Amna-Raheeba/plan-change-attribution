"""Example: PHM agent replans when the vibration sensor fails.

Run: ``python examples/demo_phm.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.models import Observation
from plan_change_attribution.planner import plan

initial_obs = []
new_obs = [Observation(
    "sf-vibration", "sensor_failure", "vibration-sensor",
    "Vibration sensor feed failed.", 10, {"sensor": "vibration"})]

old = plan("phm", initial_obs)
new = plan("phm", initial_obs + new_obs)
report = explain_replan("phm", initial_obs + new_obs, {}, old, new, new_obs)
print(report.summary())
