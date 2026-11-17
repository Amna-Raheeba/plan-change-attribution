# Experiment Log

## 2026-11-17 — v0.1.0: method works end to end

- Built the core (models, deterministic planner, diff, ablation
  attribution, 30-scenario suite, evaluation) and ran the suite.
- Results: top-1 1.00 (62/62), verdict accuracy 1.00 (30/30),
  no-trigger correct 1.00 (8/8), joint not-single 1.00 (10/10),
  false-trigger rate 0.00, mean triggering-set size 1.4545.
- Full rows in `results/evaluation.csv`.

## Finding 1 — "moved" must mean relative order

First diff implementation flagged every step after an insertion or
deletion as "moved" (absolute positions shifted). That polluted the
diff with spurious moves — e.g. a contradiction replan reported 4
extra "moved" changes. Fixed: "moved" now compares position within
the shared subsequence only. Covered by
`test_add_remove_does_not_create_spurious_moves`.

## Finding 2 — distractors are the real test of honesty

The `none` family plants heartbeat observations that do not affect
the plan. Ablation correctly repairs nothing for them, so the
verdict is "no trigger found" instead of a fabricated cause.
Covered by `test_distractor_is_never_blamed`.

## Finding 3 — scope is the method, not real agents

The planner is rule-based and deterministic, which is what makes
ablation a genuine counterfactual. On a non-deterministic LLM
planner, replays would not be exact replays. The README states this
plainly; a stochastic-planner extension (repeated replays, majority
vote) is future work, not claimed here.

## 2026-11-17 — v0.2.0: trigger stability scoring

- New module `stability.py`: every attributed trigger is re-scored
  over 20 seeded perturbed replays (shuffle, add/remove distractor,
  jittered timestamps) plus 5 duplicate probes.
- Results: mean stability of true triggers 1.00 (62/62), best
  distractor 0.00, high-confidence accuracy 1.00 (62/62),
  false-confidence rate 0.00, mean minimal-set stability 1.00.
- Full rows in `results/stability.csv`; CLI gained a `stability`
  subcommand; dashboard gained a "Trigger stability" tab.
- 75 tests passing in a bare environment with plain `pytest`
  (58 + 17 new).

## Finding 4 — duplicating a cause blinds single-observation ablation

The duplicate probe copies one candidate observation under a new
id. Duplicating the true cause collapses the verdict to "no trigger
found" in 94/94 replays: removing either copy alone repairs nothing,
and the greedy minimal set only adds observations that fix at least
one change on their own, so it stops immediately. Duplicating a
distractor never changes anything (16/16). This is a genuine blind
spot of the method, documented in the README and `docs/metrics.md`,
not tuned away. A pair-aware extension (joint ablation of identical
observations) is future work. Covered by
`test_duplicating_true_trigger_collapses_verdict` and the suite
metric `duplicate_true_collapse_rate`.

## Finding 5 — AppTest cold-start flake

The suite failed `test_dashboard_runs_without_exceptions` once in
~19 runs, always on the first run of a session, with no app error
(the builder saw the same pattern once in four runs). Cause:
Streamlit AppTest's default timeout on a cold first run under a
throttled shared CPU. Fix: explicit `timeout=120` plus one fresh
re-run on failure; a genuine app exception fails both runs. The
dashboard helpers the app renders are tested independently, so the
retry cannot mask a product bug.
