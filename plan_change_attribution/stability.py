"""Trigger stability scoring: is an attributed cause robust to noise?

An ablation attribution says "observation X caused this plan change"
because removing X in a replay restores the old plan. That claim is
only as trustworthy as the replay. If the same trigger survives when
the trace is perturbed — observations shuffled, distractors added or
removed, timestamps jittered — the cause is confident. If it vanishes
under small noise, it was fragile: an artefact of one particular
trace shape, not evidence.

Perturbations (all stdlib ``random``, seeded, so every number here is
reproducible):

- ``shuffle`` — reorder the observations and candidates. The planner
  sorts by ``(t, id)`` anyway, so this is a control: a stable trigger
  must survive it, and any failure here means non-determinism.
- ``add_distractor`` — add 1-2 heartbeat observations (kind the
  planner ignores) as extra candidates. A trigger that loses to a
  heartbeat was never real evidence.
- ``remove_distractor`` — drop the heartbeat observations from the
  trace. The attribution must not depend on irrelevant observations
  being present.
- ``jitter_t`` — permute the timestamps among observations. This
  exercises the recency tie-break in the trigger ranking: equal-score
  candidates are ordered by ``t``, so a trigger that only wins on
  recency is exposed.
- ``duplicate`` — a separate probe, not part of the headline score:
  copy one candidate observation under a new id. Single-observation
  ablation cannot separate a cause from its identical copy (removing
  either copy alone repairs nothing), so this measures a known blind
  spot of the method and is reported on its own.

Stability of a trigger for one change = fraction of perturbed replays
in which it is still the top trigger for that change. Verdicts:
``confident`` (>= 0.8), ``mixed`` (0.5-0.8), ``fragile`` (< 0.5).

Honest limits: the perturbations are synthetic trace noise, and the
planner is a deterministic rule-based mock. Stability is evidence
about the replay — that the attribution is not an artefact of one
trace shape — not evidence about a real LLM planner, whose replays
would not be exact.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.models import Observation

STABILITY_KINDS = ["shuffle", "add_distractor", "remove_distractor",
                   "jitter_t"]
DUPLICATE_KIND = "duplicate"

CONFIDENT_THRESHOLD = 0.8
MIXED_THRESHOLD = 0.5


def stability_verdict(stability: float) -> str:
    """confident / mixed / fragile from a survival fraction."""
    if stability >= CONFIDENT_THRESHOLD:
        return "confident"
    if stability >= MIXED_THRESHOLD:
        return "mixed"
    return "fragile"


def perturb(observations, candidates, kind, rng):
    """Return (observations, candidates) with one perturbation applied.

    ``candidates`` must stay a subset of ``observations``. Every
    perturbation is a pure function of ``(kind, rng)`` — no global
    state, so the same seed always gives the same perturbed trace.
    """
    observations = list(observations)
    candidates = list(candidates)
    before_obs = [(o.id, o.t) for o in observations]
    before_cand = [o.id for o in candidates]

    if kind == "shuffle":
        rng.shuffle(observations)
        rng.shuffle(candidates)

    elif kind == "add_distractor":
        n = rng.randint(1, 2)
        ts = [o.t for o in observations] or [0]
        for j in range(n):
            d = Observation(
                f"dist-pert-{rng.randint(0, 999999)}-{j}", "heartbeat",
                "monitor", "Heartbeat OK, all feeds nominal.",
                rng.choice(ts))
            observations.append(d)
            candidates.append(d)

    elif kind == "remove_distractor":
        observations = [o for o in observations if o.kind != "heartbeat"]
        candidates = [o for o in candidates if o.kind != "heartbeat"]

    elif kind == "jitter_t":
        ts = [o.t for o in observations]
        rng.shuffle(ts)
        observations = [
            Observation(o.id, o.kind, o.source, o.text, t, dict(o.facts))
            for o, t in zip(observations, ts)
        ]
        by_id = {o.id: o for o in observations}
        candidates = [by_id[o.id] for o in candidates if o.id in by_id]

    elif kind == "duplicate":
        if candidates:
            src = rng.choice(candidates)
            dup = Observation(f"{src.id}-dup", src.kind, src.source,
                              src.text, src.t, dict(src.facts))
            observations.append(dup)
            candidates.append(dup)

    else:
        raise ValueError(f"unknown perturbation kind: {kind!r}")

    # A no-op perturbation (e.g. remove_distractor with no heartbeat
    # in the trace) is replaced by a shuffle, so every replay still
    # exercises the attribution. The comparison uses (id, t) pairs,
    # so a jitter_t that actually permutes timestamps is not
    # mistaken for a no-op.
    if ([(o.id, o.t) for o in observations] == before_obs
            and [o.id for o in candidates] == before_cand):
        rng.shuffle(observations)
        rng.shuffle(candidates)

    return observations, candidates


@dataclass
class TriggerStability:
    change_key: str
    change_desc: str
    trigger_id: str | None  # base top trigger for this change
    n_replays: int
    n_survived: int
    stability: float
    verdict: str
    per_kind: dict = field(default_factory=dict)  # kind -> survival frac
    # candidate id -> fraction of replays where it was the top trigger
    candidate_survival: dict = field(default_factory=dict)


@dataclass
class StabilityReport:
    scenario_name: str
    per_change: list  # list[TriggerStability]
    minimal_set_stability: float  # replays with identical minimal set
    n_replays: int
    n_duplicate_replays: int = 0
    # Fraction of duplicate replays where at least one change's base
    # top trigger still survives (lenient "any" measure; see
    # duplicate_replays for the per-replay breakdown).
    duplicate_survival: float | None = None
    duplicate_verdicts: dict = field(default_factory=dict)
    # One entry per duplicate replay: {"duplicated": source obs id,
    # "verdict": verdict of that replay}. Lets the evaluation split
    # the probe by whether the duplicated observation was a cause.
    duplicate_replays: list = field(default_factory=list)


def _top_trigger_for(report, change_key):
    for ca in report.per_change:
        if ca.change.key() == change_key:
            return ca.top_trigger_id()
    return None


def score_stability(goal, observations, context, old_plan, new_plan,
                    candidates, n_replays=20, n_duplicate=5,
                    seed=20261117, kinds=None) -> StabilityReport:
    """Score every attributed trigger by survival across perturbed replays.

    ``n_replays`` replays cycle through ``kinds`` (default the four
    headline kinds); ``n_duplicate`` extra replays run the duplicate
    probe. Returns a ``StabilityReport`` with a per-change stability
    for the base top trigger, per-kind survival, and minimal-set
    stability.
    """
    kinds = list(kinds) if kinds else list(STABILITY_KINDS)
    base = explain_replan(goal, observations, context, old_plan,
                          new_plan, candidates)
    base_minimal = set(base.minimal_triggering_set)
    base_keys = [ca.change.key() for ca in base.per_change]

    survived = Counter()
    survived_kind = defaultdict(Counter)
    kind_totals = Counter()
    cand_survival = defaultdict(Counter)
    minimal_same = 0
    total = 0

    for i in range(n_replays):
        kind = kinds[i % len(kinds)]
        rng = random.Random(seed + i)
        p_obs, p_cand = perturb(observations, candidates, kind, rng)
        rep = explain_replan(goal, p_obs, context, old_plan, new_plan,
                             p_cand)
        total += 1
        kind_totals[kind] += 1
        if set(rep.minimal_triggering_set) == base_minimal:
            minimal_same += 1
        for ca in base.per_change:
            top = _top_trigger_for(rep, ca.change.key())
            cand_survival[ca.change.key()][top] += 1
            if top == ca.top_trigger_id():
                survived[ca.change.key()] += 1
                survived_kind[kind][ca.change.key()] += 1

    dup_survived = 0
    dup_verdicts: Counter = Counter()
    dup_replays: list = []
    # Only changes with a real (non-None) base top trigger count for
    # the duplicate probe: a "no trigger" staying "no trigger" is not
    # a trigger surviving.
    dup_changes = [ca for ca in base.per_change
                   if ca.top_trigger_id() is not None]
    for j in range(n_duplicate):
        rng = random.Random(seed + 1000 + j)
        p_obs, p_cand = perturb(observations, candidates, DUPLICATE_KIND,
                                rng)
        rep = explain_replan(goal, p_obs, context, old_plan, new_plan,
                             p_cand)
        dup_verdicts[rep.verdict] += 1
        dup_src = next((o.id[:-4] for o in p_cand
                        if o.id.endswith("-dup")), None)
        dup_replays.append({"duplicated": dup_src,
                            "verdict": rep.verdict})
        if dup_changes and any(
                _top_trigger_for(rep, ca.change.key())
                == ca.top_trigger_id() for ca in dup_changes):
            dup_survived += 1

    per_change: list = []
    for ca in base.per_change:
        key = ca.change.key()
        n = total
        s = survived[key] / n if n else 1.0
        per_kind = {
            k: round(survived_kind[k][key] / kind_totals[k], 4)
            for k in kinds if kind_totals[k]
        }
        cand = {
            (cid or "none"): round(c / n, 4)
            for cid, c in sorted(cand_survival[key].items(),
                                 key=lambda kv: (-kv[1], str(kv[0])))
        }
        per_change.append(TriggerStability(
            change_key=key, change_desc=ca.change.describe(),
            trigger_id=ca.top_trigger_id(), n_replays=n,
            n_survived=survived[key], stability=round(s, 4),
            verdict=stability_verdict(s), per_kind=per_kind,
            candidate_survival=cand))

    return StabilityReport(
        scenario_name="",
        per_change=per_change,
        minimal_set_stability=(
            round(minimal_same / total, 4) if total else 1.0),
        n_replays=total,
        n_duplicate_replays=n_duplicate,
        duplicate_survival=(
            round(dup_survived / n_duplicate, 4)
            if n_duplicate and dup_changes else None),
        duplicate_verdicts=dict(dup_verdicts),
        duplicate_replays=dup_replays)


def evaluate_stability(scenarios=None, n_replays=20, n_duplicate=5,
                       seed=20261117) -> dict:
    """Score stability over the whole suite; return metrics + rows.

    Metrics:
    - ``mean_stability_true`` — mean stability of the top trigger
      over changes where the top trigger is the true trigger.
    - ``mean_stability_best_distractor`` — mean, over scored changes,
      of the highest survival fraction of any non-truth candidate
      (how close a distractor ever came to looking like a cause).
    - ``high_confidence_coverage`` — fraction of scored changes with
      stability >= 0.8.
    - ``high_confidence_accuracy`` — among high-confidence changes,
      fraction whose top trigger is the true trigger.
    - ``false_confidence_rate`` — fraction of scored changes where a
      non-truth candidate's survival is >= 0.8.
    - ``mean_minimal_set_stability`` — mean over scenarios.
    - ``duplicate_survival`` — mean survival of the true trigger
      under the duplicate probe (lenient "any change" measure).
    - ``duplicate_no_trigger_rate`` — fraction of duplicate replays
      whose verdict collapsed to "no trigger found".
    - ``duplicate_true_collapse_rate`` — over duplicate replays where
      the duplicated observation WAS a true cause: fraction whose
      verdict collapsed to "no trigger found".
    - ``duplicate_other_unchanged_rate`` — over duplicate replays
      where the duplicated observation was NOT a cause (a
      distractor): fraction whose verdict stayed as expected.
    """
    from plan_change_attribution.scenarios import build_suite

    scenarios = scenarios if scenarios is not None else build_suite()
    rows: list = []
    true_stabs: list = []
    distract_stabs: list = []
    high_conf_total = high_conf_correct = 0
    false_conf = 0
    minimal_stabs: list = []
    dup_survivals: list = []
    dup_no_trigger = dup_total = 0
    dup_true_total = dup_true_collapsed = 0
    dup_other_total = dup_other_unchanged = 0
    scored = 0

    for sc in scenarios:
        old, new = sc.old_plan(), sc.new_plan()
        rep = score_stability(
            sc.goal, sc.all_observations(), sc.context, old, new,
            sc.candidates(), n_replays=n_replays,
            n_duplicate=n_duplicate, seed=seed)
        rep.scenario_name = sc.name
        minimal_stabs.append(rep.minimal_set_stability)
        if rep.duplicate_survival is not None:
            dup_survivals.append(rep.duplicate_survival)
        dup_total += n_duplicate
        dup_no_trigger += rep.duplicate_verdicts.get("no trigger found", 0)
        truth_ids = {v for v in sc.trigger_map.values() if v}
        # The cause-vs-distractor split only means something where an
        # attribution exists; no-trigger scenarios have no cause to
        # duplicate, so they are left out of this breakdown (they
        # still count in duplicate_no_trigger_rate above).
        if truth_ids:
            for dr in rep.duplicate_replays:
                if dr["duplicated"] in truth_ids:
                    dup_true_total += 1
                    dup_true_collapsed += int(
                        dr["verdict"] == "no trigger found")
                elif dr["duplicated"] is not None:
                    dup_other_total += 1
                    dup_other_unchanged += int(
                        dr["verdict"] == sc.expected_verdict)
        for ts in rep.per_change:
            # trigger_map is keyed by step id; change key is kind:step.
            step_id = ts.change_key.split(":", 1)[1]
            truth = sc.trigger_map.get(step_id)
            is_true = truth is not None and ts.trigger_id == truth
            best_distractor = max(
                (v for cid, v in ts.candidate_survival.items()
                 if cid != truth and cid != "none"),
                default=0.0)
            if truth is not None:
                scored += 1
                distract_stabs.append(best_distractor)
                if is_true:
                    true_stabs.append(ts.stability)
                if ts.stability >= CONFIDENT_THRESHOLD:
                    high_conf_total += 1
                    high_conf_correct += int(is_true)
                if best_distractor >= CONFIDENT_THRESHOLD:
                    false_conf += 1
            rows.append({
                "scenario": sc.name, "family": sc.family,
                "change": ts.change_key, "truth": truth or "",
                "top_trigger": ts.trigger_id or "",
                "stability": ts.stability, "verdict": ts.verdict,
                "best_distractor_stability": best_distractor,
                "minimal_set_stability": rep.minimal_set_stability,
            })

    metrics = {
        "n_scenarios": len(scenarios),
        "n_changes_scored": scored,
        "n_replays": n_replays,
        "n_duplicate_replays": n_duplicate,
        "mean_stability_true": (
            round(sum(true_stabs) / len(true_stabs), 4)
            if true_stabs else None),
        "n_true_triggers": len(true_stabs),
        "mean_stability_best_distractor": (
            round(sum(distract_stabs) / len(distract_stabs), 4)
            if distract_stabs else None),
        "high_confidence_coverage": (
            round(high_conf_total / scored, 4) if scored else None),
        "high_confidence_accuracy": (
            round(high_conf_correct / high_conf_total, 4)
            if high_conf_total else None),
        "high_confidence_n": high_conf_total,
        "false_confidence_rate": (
            round(false_conf / scored, 4) if scored else None),
        "mean_minimal_set_stability": (
            round(sum(minimal_stabs) / len(minimal_stabs), 4)
            if minimal_stabs else None),
        "duplicate_survival": (
            round(sum(dup_survivals) / len(dup_survivals), 4)
            if dup_survivals else None),
        "duplicate_no_trigger_rate": (
            round(dup_no_trigger / dup_total, 4) if dup_total else None),
        "duplicate_true_collapse_rate": (
            round(dup_true_collapsed / dup_true_total, 4)
            if dup_true_total else None),
        "duplicate_true_n": dup_true_total,
        "duplicate_other_unchanged_rate": (
            round(dup_other_unchanged / dup_other_total, 4)
            if dup_other_total else None),
        "duplicate_other_n": dup_other_total,
    }
    return {"metrics": metrics, "rows": rows}


def run_stability_evaluation(results_dir: str | Path = "results",
                             n_replays=20, n_duplicate=5,
                             seed=20261117) -> dict:
    """Run the suite stability evaluation; write CSV + JSON summary."""
    out = evaluate_stability(n_replays=n_replays,
                             n_duplicate=n_duplicate, seed=seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stability.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scenario", "family", "change", "truth",
                           "top_trigger", "stability", "verdict",
                           "best_distractor_stability",
                           "minimal_set_stability"])
        writer.writeheader()
        writer.writerows(out["rows"])
    summary = dict(out["metrics"])
    with open(results_dir / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
