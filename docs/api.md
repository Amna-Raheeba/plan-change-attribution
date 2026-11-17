# API

## 1. Core call

```python
from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.planner import plan

old = plan("phm", initial_obs)
new = plan("phm", initial_obs + new_obs)
report = explain_replan("phm", initial_obs + new_obs, {},
                        old, new, new_obs)
print(report.summary())
print(report.verdict)                  # "single trigger" | ...
print(report.minimal_triggering_set)   # ["sf-vibration-0"]
```

## 2. Functions

- `plan_change_attribution.planner.plan(goal, observations, context)`
  — deterministic plan. `goal` is `"phm"` or `"healthcare"`.
- `plan_change_attribution.diff.diff_plans(old, new)` — list of
  `Change` (added / removed / modified / moved).
- `plan_change_attribution.diff.change_is_fixed_by(change, ablated, old)`
  — did the ablated plan restore the old plan for this change?
- `plan_change_attribution.attribution.explain_replan(goal, observations,
  context, old_plan, new_plan, candidates)` — full `AttributionReport`.
- `plan_change_attribution.attribution.rank_triggers_for_change(...)`
  — ranked `TriggerScore` list for one change.
- `plan_change_attribution.attribution.greedy_minimal_set(...)`
  — observation ids whose removal restores the old plan, or `[]`.
- `plan_change_attribution.scenarios.build_suite()` — 30 scenarios.
- `plan_change_attribution.evaluation.evaluate_suite()` — metrics +
  per-change rows. `run_evaluation(out_dir)` also writes CSV/JSON.
- `plan_change_attribution.stability.score_stability(goal,
  observations, context, old_plan, new_plan, candidates, n_replays=20,
  n_duplicate=5, seed=20261117)` — per-change trigger stability
  (survival across perturbed replays) + minimal-set stability +
  duplicate probe. `evaluate_stability()` runs the whole suite;
  `run_stability_evaluation(out_dir)` also writes CSV/JSON.
- `plan_change_attribution.stability.perturb(observations, candidates,
  kind, rng)` — one perturbation: `shuffle`, `add_distractor`,
  `remove_distractor`, `jitter_t`, `duplicate`.

## 3. CLI

```
python -m plan_change_attribution demo
python -m plan_change_attribution evaluate --out results
python -m plan_change_attribution explain phm-vibration-failure-0
python -m plan_change_attribution stability --out results
```
