# Architecture

## 1. Problem

Plan-and-Execute agents replan when the world changes, but no existing
tool explains *why* a specific plan changed. Replanning literature
describes how to replan; LangGraph Studio rewinds and edits traces;
Agent-Flow runs authored "what if this step failed?" counterfactuals.
None outputs a per-change causal attribution for a plan diff.

## 2. Method: plan diff + single-observation ablation

1. **Freeze the trace** at the replan point: planner, goal,
   observations visible at the replan, and context are fixed.
2. **Diff** the initial plan and the replanned plan (added, removed,
   modified, moved — where "moved" means relative order changed, so
   adding/removing a step does not create spurious moves).
3. **Ablate**: for each observation that arrived after the first plan,
   remove it alone, re-run the planner, and count how many diff
   changes that removal repairs. That count is the evidence score.
4. **Rank** candidates by score (ties: most recent first, then id).
5. **Greedy minimal triggering set**: repeatedly add the observation
   whose removal repairs the most remaining changes, until the
   ablated plan equals the original plan or nothing improves.
6. **Stability**: re-score every attributed trigger over 20 seeded
   perturbed replays (shuffle, add/remove distractors, jittered
   timestamps) plus 5 duplicate probes. A trigger that survives
   noise is confident; one that vanishes is fragile. Duplicating a
   true cause is a known blind spot: single-observation ablation
   cannot separate a cause from its identical copy.

## 3. Verdicts

- `single trigger` — one observation's removal restores the plan.
- `joint trigger` — two or more are jointly needed.
- `no trigger found` — no removal restores the plan (e.g. the replan
  was caused by internal state, not by any observation).

## 4. Modules

- `models.py` — Step, Plan, Observation, Change, Trace (dataclasses).
- `planner.py` — deterministic rule-based planner (PHM, healthcare).
  Pure function of `(goal, observations, context)`.
- `diff.py` — plan diff + per-change "is this fixed?" test.
- `attribution.py` — ranking, greedy minimal set, verdicts.
- `stability.py` — trigger stability scoring over perturbed replays
  (shuffle, add/remove distractor, jittered timestamps) plus the
  duplicate probe; suite-level stability evaluation.
- `scenarios.py` — 30 scenarios with planted ground truth.
- `evaluation.py` — suite metrics + CSV/JSON output.

## 5. Honest scope

The planner is a deterministic rule-based mock, and the scenarios are
synthetic with known ground truth. The measured numbers therefore
evaluate the *attribution method* — whether ablation recovers the
planted cause — not performance on real LLM agents, whose planners
are non-deterministic. That limitation is stated in the README.
