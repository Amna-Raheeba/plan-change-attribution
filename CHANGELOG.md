# Changelog

## 0.2.0 — 2026-11-17

- New: trigger stability scoring (`stability.py`) — every attributed
  trigger re-scored over 20 seeded perturbed replays (shuffle,
  add/remove distractor, jittered timestamps) plus a 5-replay
  duplicate probe; per-change stability + verdict
  (confident/mixed/fragile), minimal-set stability.
- Measured: mean stability of true triggers 1.00 (62/62), best
  distractor 0.00, high-confidence accuracy 1.00 (62/62),
  false-confidence rate 0.00, mean minimal-set stability 1.00.
- Duplicate probe finding: duplicating the true cause collapses the
  verdict to "no trigger found" in 94/94 replays; duplicating a
  distractor leaves it unchanged in 16/16 — a documented blind spot
  of single-observation ablation.
- CLI gained a `stability` subcommand; dashboard gained a
  "Trigger stability" tab; notebook `03_stability.py` added.
- Fixed: Streamlit AppTest cold-start flake — explicit 120 s timeout
  plus one fresh re-run on failure (failed once in ~19 runs).
- 75 tests passing in a bare environment with plain `pytest`.

## 0.1.0 — 2026-11-17

- First release: plan-diff + single-observation ablation attribution.
- Deterministic mock planner for PHM and healthcare domains.
- Verdicts: single trigger, joint trigger, no trigger found.
- Greedy minimal triggering set; per-change ranked triggers.
- 30-scenario suite with planted ground truth (12 single, 10 joint,
  8 none); evaluation metrics + CSV/JSON output.
- Streamlit dashboard (scenario explorer + evaluation), verified
  with AppTest, 0 exceptions.
- 58 tests passing in a bare environment with plain `pytest`.
- Measured: top-1 1.00 (62/62), verdict accuracy 1.00 (30/30),
  mean triggering-set size 1.4545, false-trigger rate 0.00.
