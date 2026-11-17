# Plan-Change Attribution

**When an agent replans, why did the plan change?** This library
attributes each dropped or added step in a replan to the single
observation that caused it — by ablating that observation in a
replay of the planner.

Replanning tools tell you *how* to replan. LangGraph Studio lets you
rewind and edit a trace. Agent-Flow runs authored "what if this step
failed?" counterfactuals. None of them outputs a per-change causal
attribution for a plan diff. This one does.

## 1. The idea in 30 seconds

1. Freeze the trace at the replan point.
2. Diff the old plan and the new plan (added / removed / modified /
   moved steps).
3. For each observation that arrived after the first plan, remove it
   alone, re-run the planner, and count how many diff changes that
   removal repairs. That count is the evidence.
4. Rank the candidates by evidence, then greedily build the smallest
   set whose removal restores the original plan.

Verdicts: **single trigger** (one observation restores the plan),
**joint trigger** (two or more are jointly needed), or **no trigger
found** (no observation caused it — e.g. the step budget ran out).
"No trigger found" is an honest verdict, not a failure to look.

**Trigger stability (new in v0.2.0):** every attributed trigger is
re-scored over 20 seeded perturbed replays — shuffled observation
order, distractors added or removed, jittered timestamps — plus a
duplicate probe. A trigger that survives noise is confident; one
that vanishes is fragile. The duplicate probe exposes a real blind
spot: duplicating the true cause collapses the verdict to "no
trigger found" in 94/94 replays, because single-observation
ablation cannot separate a cause from its identical copy.

## 2. Quick start

```bash
pip install -e .
python -m plan_change_attribution demo
python -m plan_change_attribution evaluate --out results
python -m plan_change_attribution stability --out results
python -m plan_change_attribution explain phm-vibration-failure-0
```

```python
from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.planner import plan

old = plan("phm", initial_obs)
new = plan("phm", initial_obs + new_obs)
report = explain_replan("phm", initial_obs + new_obs, {},
                        old, new, new_obs)
print(report.summary())
```

## 3. Measured results

30 scenarios with planted ground truth (12 single-trigger, 10
joint-trigger, 8 no-trigger), 70 plan changes. Measured 2026-11-17:

| Metric | Value |
|---|---|
| Top-1 trigger accuracy | **1.00** (62/62 changes) |
| Verdict accuracy | **1.00** (30/30 scenarios) |
| Mean triggering-set size | **1.45** |
| No-trigger correct | **1.00** (8/8) |
| Joint correctly not-single | **1.00** (10/10) |
| False-trigger rate | **0.00** |
| Mean stability, true triggers | **1.00** (62/62) |
| Mean stability, best distractor | **0.00** |
| High-confidence accuracy (≥ 0.8) | **1.00** (62/62) |
| Duplicate cause → "no trigger found" | **1.00** (94/94 replays) |

Full per-change rows: `results/evaluation.csv`. Stability rows:
`results/stability.csv`.

**Honest scope:** the planner is a deterministic rule-based mock and
the scenarios are synthetic with known ground truth. These numbers
prove the attribution *method* recovers planted causes; they are not
a claim about real LLM agents, whose planners are non-deterministic.

## 4. What is inside

- `plan_change_attribution/` — models, deterministic planner (PHM +
  healthcare domains), plan diff, ablation attribution, trigger
  stability scoring, 30-scenario suite, evaluation.
- `dashboard/` — Streamlit app: scenario explorer + evaluation +
  trigger-stability view.
- `notebooks/` — PHM demo, full evaluation, stability evaluation.
- `examples/` — runnable PHM and healthcare demos.
- `docs/` — architecture, API, metrics, experiment log.
- `tests/` — 75 tests, all passing.

## 5. Install

```bash
pip install -e .            # core, stdlib only, offline
pip install -e ".[dashboard]"  # + Streamlit dashboard
```

Python 3.10+ · stdlib-only core · CPU-only · fully offline.

## 6. License

MIT — see `LICENSE`.
