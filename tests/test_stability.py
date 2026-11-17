"""Tests for trigger stability scoring (plan_change_attribution/stability)."""

import json
import random

import pytest

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.scenarios import build_suite
from plan_change_attribution.stability import (
    CONFIDENT_THRESHOLD,
    DUPLICATE_KIND,
    STABILITY_KINDS,
    evaluate_stability,
    perturb,
    run_stability_evaluation,
    score_stability,
    stability_verdict,
)


def _suite():
    return {sc.name: sc for sc in build_suite()}


def _report(sc):
    old, new = sc.old_plan(), sc.new_plan()
    return explain_replan(sc.goal, sc.all_observations(), sc.context,
                          old, new, sc.candidates())


# ------------------------------------------------------------- perturbations

def test_perturb_shuffle_preserves_ids_and_subset():
    sc = _suite()["phm-temp-failure-plus-anomaly-0"]
    obs, cand = perturb(sc.all_observations(), sc.candidates(),
                        "shuffle", random.Random(1))
    assert {o.id for o in obs} == {o.id for o in sc.all_observations()}
    assert {o.id for o in cand} <= {o.id for o in obs}


def test_perturb_add_distractor_adds_heartbeat_only():
    from plan_change_attribution.planner import plan as run_planner
    sc = _suite()["phm-cost-spike-0"]
    obs, cand = perturb(sc.all_observations(), sc.candidates(),
                        "add_distractor", random.Random(2))
    added = [o for o in obs if o.id not in {x.id for x in sc.all_observations()}]
    assert 1 <= len(added) <= 2
    assert all(o.kind == "heartbeat" for o in added)
    assert all(o in cand for o in added)
    # The plan must be unchanged: heartbeats are ignored by the planner.
    assert run_planner(sc.goal, obs, sc.context).step_ids() == \
        sc.new_plan().step_ids()


def test_perturb_remove_distractor_drops_heartbeats():
    sc = _suite()["phm-vibration-failure-0"]
    assert any(o.kind == "heartbeat" for o in sc.all_observations())
    obs, cand = perturb(sc.all_observations(), sc.candidates(),
                        "remove_distractor", random.Random(3))
    assert all(o.kind != "heartbeat" for o in obs)
    assert all(o.kind != "heartbeat" for o in cand)
    assert {o.id for o in cand} <= {o.id for o in obs}


def test_perturb_duplicate_copies_candidate_with_new_id():
    sc = _suite()["phm-cost-spike-0"]
    obs, cand = perturb(sc.all_observations(), sc.candidates(),
                        "duplicate", random.Random(4))
    dups = [o for o in obs if o.id.endswith("-dup")]
    assert len(dups) == 1
    src = sc.all_observations()[0]
    assert dups[0].kind == src.kind and dups[0].facts == src.facts
    assert dups[0] in cand


def test_perturb_jitter_t_permutes_timestamps():
    sc = _suite()["phm-temp-failure-plus-anomaly-0"]
    obs, _ = perturb(sc.all_observations(), sc.candidates(),
                     "jitter_t", random.Random(5))
    assert sorted(o.t for o in obs) == sorted(
        o.t for o in sc.all_observations())
    assert {o.id for o in obs} == {o.id for o in sc.all_observations()}


def test_perturb_noop_falls_back_without_breaking_subset():
    # Joint scenario with no heartbeat: remove_distractor is a no-op,
    # so the fallback must still return a valid (subset) trace.
    sc = _suite()["phm-anomaly-plus-cost-0"]
    assert not any(o.kind == "heartbeat" for o in sc.all_observations())
    obs, cand = perturb(sc.all_observations(), sc.candidates(),
                        "remove_distractor", random.Random(6))
    assert {o.id for o in cand} <= {o.id for o in obs}
    assert {o.id for o in obs} == {o.id for o in sc.all_observations()}


def test_unknown_perturbation_kind_raises():
    sc = _suite()["phm-cost-spike-0"]
    with pytest.raises(ValueError, match="unknown perturbation kind"):
        perturb(sc.all_observations(), sc.candidates(),
                "explode", random.Random(0))


# ------------------------------------------------------------ stability core

def test_stability_true_trigger_is_confident():
    sc = _suite()["phm-vibration-failure-0"]
    rep = score_stability(sc.goal, sc.all_observations(), sc.context,
                          sc.old_plan(), sc.new_plan(), sc.candidates())
    assert rep.n_replays == 20
    assert all(ts.stability == 1.0 for ts in rep.per_change)
    assert all(ts.verdict == "confident" for ts in rep.per_change)
    assert all(ts.per_kind[k] == 1.0 for ts in rep.per_change
               for k in STABILITY_KINDS)
    assert rep.minimal_set_stability == 1.0


def test_stability_is_deterministic_for_fixed_seed():
    sc = _suite()["health-contradiction-G1-0"]
    args = (sc.goal, sc.all_observations(), sc.context,
            sc.old_plan(), sc.new_plan(), sc.candidates())
    a = score_stability(*args)
    b = score_stability(*args)
    assert [(t.change_key, t.stability) for t in a.per_change] == \
           [(t.change_key, t.stability) for t in b.per_change]
    assert a.minimal_set_stability == b.minimal_set_stability
    assert a.duplicate_replays == b.duplicate_replays


def test_stability_distractor_never_survives_as_top_trigger():
    sc = _suite()["phm-vibration-failure-0"]
    rep = score_stability(sc.goal, sc.all_observations(), sc.context,
                          sc.old_plan(), sc.new_plan(), sc.candidates())
    for ts in rep.per_change:
        assert ts.candidate_survival.get("dist-0", 0.0) == 0.0


def test_stability_verdict_thresholds():
    assert stability_verdict(1.0) == "confident"
    assert stability_verdict(0.8) == "confident"
    assert stability_verdict(0.79) == "mixed"
    assert stability_verdict(0.5) == "mixed"
    assert stability_verdict(0.49) == "fragile"
    assert stability_verdict(0.0) == "fragile"


def test_duplicating_true_trigger_collapses_verdict():
    # Single-observation ablation cannot separate a cause from its
    # identical copy: removing either copy alone repairs nothing.
    sc = _suite()["phm-vibration-failure-0"]
    old, new = sc.old_plan(), sc.new_plan()
    src = sc.candidates()[0]
    assert src.id == "sf-vibration-0"
    from plan_change_attribution.models import Observation
    dup = Observation(f"{src.id}-dup", src.kind, src.source,
                      src.text, src.t, dict(src.facts))
    obs = sc.all_observations() + [dup]
    cand = sc.candidates() + [dup]
    rep = explain_replan(sc.goal, obs, sc.context, old, new, cand)
    assert rep.verdict == "no trigger found"
    assert all(ca.top_trigger is None for ca in rep.per_change)


def test_duplicating_distractor_leaves_verdict_unchanged():
    sc = _suite()["phm-vibration-failure-0"]
    old, new = sc.old_plan(), sc.new_plan()
    src = [o for o in sc.candidates() if o.kind == "heartbeat"][0]
    from plan_change_attribution.models import Observation
    dup = Observation(f"{src.id}-dup", src.kind, src.source,
                      src.text, src.t, dict(src.facts))
    obs = sc.all_observations() + [dup]
    cand = sc.candidates() + [dup]
    rep = explain_replan(sc.goal, obs, sc.context, old, new, cand)
    assert rep.verdict == "single trigger"
    assert rep.minimal_triggering_set == ["sf-vibration-0"]


# ------------------------------------------------------- suite-level metrics

def test_evaluate_stability_headline_numbers():
    out = evaluate_stability()
    m = out["metrics"]
    assert m["n_scenarios"] == 30
    assert m["n_changes_scored"] == 62
    assert m["mean_stability_true"] == 1.0
    assert m["n_true_triggers"] == 62
    assert m["mean_stability_best_distractor"] == 0.0
    assert m["high_confidence_coverage"] == 1.0
    assert m["high_confidence_accuracy"] == 1.0
    assert m["false_confidence_rate"] == 0.0
    assert m["mean_minimal_set_stability"] == 1.0
    # Duplicate probe: duplicating a true cause collapses the verdict
    # in every case; duplicating a distractor never changes it.
    assert m["duplicate_true_collapse_rate"] == 1.0
    assert m["duplicate_true_n"] == 94
    assert m["duplicate_other_unchanged_rate"] == 1.0
    assert m["duplicate_other_n"] == 16
    assert m["duplicate_no_trigger_rate"] == 0.8933
    assert len(out["rows"]) == 70


def test_run_stability_evaluation_writes_files(tmp_path):
    summary = run_stability_evaluation(tmp_path)
    assert (tmp_path / "stability.csv").exists()
    assert (tmp_path / "stability_summary.json").exists()
    assert json.loads(
        (tmp_path / "stability_summary.json").read_text()) == summary
    assert summary["mean_stability_true"] == 1.0


def test_cli_stability_command(tmp_path, capsys):
    from plan_change_attribution.__main__ import main
    assert main(["stability", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "stability.csv").exists()
    assert "mean_stability_true" in capsys.readouterr().out
