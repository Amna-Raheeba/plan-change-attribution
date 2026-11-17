"""Notebook 02 — run the full 30-scenario evaluation.

Run: ``python notebooks/02_evaluation.py``
Prints the headline metrics and the per-family breakdown.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_change_attribution.evaluation import evaluate_suite

out = evaluate_suite()
m = out["metrics"]
print("Plan-Change Attribution — 30-scenario evaluation")
print(f"  scenarios            : {m['n_scenarios']}")
print(f"  plan changes         : {m['n_changes']}")
print(f"  top-1 trigger        : {m['top1']:.0%} ({m['top1_hits']}/{m['top1_total']})")
print(f"  verdict accuracy     : {m['verdict_accuracy']:.0%}")
print(f"  mean triggering set  : {m['mean_triggering_set_size']}")
print(f"  no-trigger correct   : {m['no_single_trigger_correct']:.0%} "
      f"({m['no_single_trigger_n']} scenarios)")
print(f"  joint not-single     : {m['joint_not_single_correct']:.0%} "
      f"({m['joint_n']} scenarios)")
print(f"  false-trigger rate   : {m['false_trigger_rate']:.0%}")
print()
print("Honest scope: synthetic scenarios, deterministic rule-based")
print("planner, known ground truth. The numbers measure the attribution")
print("method — not performance on real LLM agents.")
