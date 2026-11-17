"""Notebook 03 — trigger stability: does the attributed cause survive noise?

Run: ``python notebooks/03_stability.py``
Prints the headline stability metrics over the 30-scenario suite and
the duplicate-probe finding. Used by CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan_change_attribution.stability import evaluate_stability

out = evaluate_stability()
m = out["metrics"]
print("Plan-Change Attribution — trigger stability (30 scenarios)")
print(f"  changes scored           : {m['n_changes_scored']}")
print(f"  perturbed replays/change  : {m['n_replays']} "
      f"(+ {m['n_duplicate_replays']} duplicate probes)")
print(f"  mean stability, true trig : {m['mean_stability_true']:.2f} "
      f"({m['n_true_triggers']} true triggers)")
print(f"  mean stability, best distr: {m['mean_stability_best_distractor']:.2f}")
print(f"  high-confidence coverage  : {m['high_confidence_coverage']:.0%} "
      f"(stability >= 0.8)")
print(f"  high-confidence accuracy  : {m['high_confidence_accuracy']:.0%} "
      f"({m['high_confidence_n']} changes)")
print(f"  false-confidence rate     : {m['false_confidence_rate']:.0%}")
print(f"  mean minimal-set stability: {m['mean_minimal_set_stability']:.2f}")
print(f"  duplicate probe: cause duplicated -> 'no trigger found' "
      f"in {m['duplicate_true_collapse_rate']:.0%} "
      f"({m['duplicate_true_n']} replays); distractor duplicated -> "
      f"verdict unchanged in {m['duplicate_other_unchanged_rate']:.0%} "
      f"({m['duplicate_other_n']} replays)")
print()
print("Honest scope: stability 1.0 is expected here — the four headline")
print("perturbations are trace noise the deterministic planner ignores")
print("by design, so the score proves the attribution does not depend")
print("on trace shape. The duplicate probe is the real finding: it")
print("exposes that single-observation ablation cannot separate a")
print("cause from its identical copy. Stability is evidence about the")
print("replay, not about a real LLM planner.")
