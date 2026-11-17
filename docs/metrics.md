# Metrics

All numbers are measured on the 30-scenario suite
(`python -m plan_change_attribution evaluate`), 2026-11-17.
Full per-change rows: `results/evaluation.csv`.

## 1. Headline results

| Metric | Value |
|---|---|
| Scenarios | 30 (12 single, 10 joint, 8 none) |
| Plan changes | 70 |
| Top-1 trigger accuracy | 1.00 (62/62 changes with a known trigger) |
| Verdict accuracy | 1.00 (30/30 scenarios) |
| Mean triggering-set size | 1.4545 (12 x 1 + 10 x 2, over 22 explained) |
| No-trigger correct | 1.00 (8/8 none-family scenarios) |
| Joint correctly not-single | 1.00 (10/10 joint scenarios) |
| False-trigger rate | 0.00 (0 wrong top triggers, single family) |

## 2. What each metric means

- **Top-1**: for a change whose true trigger is known, is the true
  trigger ranked first by ablation evidence?
- **Verdict accuracy**: does the reported verdict (single / joint /
  no trigger found) equal the planted ground truth?
- **Mean triggering-set size**: size of the greedy minimal set that
  restores the original plan.
- **No-trigger correct**: in the `none` family (replan caused by the
  step budget, with distractor observations planted), is the verdict
  honestly "no trigger found" instead of blaming a distractor?
- **False-trigger rate**: how often the top-ranked trigger is wrong.

## 3. Scope caveat

Synthetic scenarios, deterministic planner, known ground truth. The
numbers prove the attribution method recovers planted causes; they
are not a claim about real LLM agents.

## 4. Trigger stability

Each attributed trigger is re-scored over 20 perturbed replays
(shuffle, add/remove distractors, jittered timestamps; seed
20261117) plus 5 duplicate probes. Full per-change rows:
`results/stability.csv`. Measured 2026-11-17:

| Metric | Value |
|---|---|
| Mean stability, true triggers | 1.00 (62/62) |
| Mean stability, best distractor | 0.00 |
| High-confidence coverage (≥ 0.8) | 1.00 (62/62 changes) |
| High-confidence accuracy | 1.00 (62/62) |
| False-confidence rate | 0.00 |
| Mean minimal-set stability | 1.00 |
| Duplicate probe: cause duplicated → "no trigger found" | 1.00 (94/94 replays) |
| Duplicate probe: distractor duplicated → verdict unchanged | 1.00 (16/16 replays) |
| Duplicate replays collapsing to "no trigger found" (all) | 0.8933 (134/150) |

**Reading the numbers honestly:** stability 1.00 is expected here —
the four headline perturbations are trace noise the deterministic
planner ignores by design, so the score proves the attribution does
not depend on trace shape, not that it would survive a real LLM
planner. The duplicate probe is the real finding: it exposes that
single-observation ablation cannot separate a cause from its
identical copy — duplicating the true cause collapses the verdict
to "no trigger found" in 94/94 replays, while duplicating a
distractor never changes anything (16/16). Stability is evidence
about the replay, not about a real LLM planner.
