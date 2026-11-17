"""Tests for ablation-based attribution."""

from plan_change_attribution.attribution import (
    explain_replan, greedy_minimal_set, rank_triggers_for_change,
)
from plan_change_attribution.diff import diff_plans
from plan_change_attribution.models import Observation
from plan_change_attribution.planner import plan
from plan_change_attribution.scenarios import build_suite


def _report_for(name):
    sc = {s.name: s for s in build_suite()}[name]
    return sc, explain_replan(
        sc.goal, sc.all_observations(), sc.context,
        sc.old_plan(), sc.new_plan(), sc.candidates())


def test_single_trigger_scenario():
    sc, report = _report_for("phm-vibration-failure-0")
    assert report.verdict == "single trigger"
    assert report.minimal_triggering_set == ["sf-vibration-0"]
    assert len(report.changes) == 4
    for ca in report.per_change:
        assert ca.top_trigger_id() == "sf-vibration-0"


def test_joint_trigger_scenario():
    sc, report = _report_for("phm-temp-failure-plus-anomaly-0")
    assert report.verdict == "joint trigger"
    assert sorted(report.minimal_triggering_set) == [
        "anom-0", "sf-temperature-0"]
    by_step = {ca.change.step_id: ca.top_trigger_id()
               for ca in report.per_change}
    assert by_step["raise-alert"] == "anom-0"
    assert by_step["collect-temperature"] == "sf-temperature-0"


def test_no_trigger_scenario():
    sc, report = _report_for("phm-budget-exhausted-0")
    assert report.verdict == "no trigger found"
    assert report.minimal_triggering_set == []
    assert report.per_change[0].top_trigger is None


def test_distractor_is_never_blamed():
    for sc in build_suite():
        if sc.family != "none":
            continue
        _, report = _report_for(sc.name)
        for ca in report.per_change:
            assert ca.top_trigger is None


def test_ranking_prefers_trigger_over_distractor():
    sc, report = _report_for("phm-vibration-failure-0")
    change = report.changes[0]
    ranked = rank_triggers_for_change(
        sc.goal, sc.all_observations(), sc.context,
        sc.old_plan(), sc.new_plan(), change, sc.candidates())
    assert ranked[0].observation.id == "sf-vibration-0"
    assert ranked[0].score > ranked[-1].score


def test_greedy_minimal_set_single():
    sc = {s.name: s for s in build_suite()}["phm-cost-spike-0"]
    minimal = greedy_minimal_set(
        sc.goal, sc.all_observations(), sc.context,
        sc.old_plan(), sc.new_plan(), sc.candidates())
    assert minimal == ["cost-0"]


def test_greedy_minimal_set_empty_when_no_changes():
    p = plan("phm", [])
    assert greedy_minimal_set("phm", [], {}, p, p, []) == []


def test_report_summary_mentions_verdict():
    _, report = _report_for("health-contradiction-G1-0")
    assert "single trigger" in report.summary()
    assert "contra-G1-0" in report.summary()


def test_healthcare_emergency_attribution():
    sc, report = _report_for("health-emergency-0")
    assert report.verdict == "single trigger"
    assert report.minimal_triggering_set == ["emerg-0"]
    assert {c.step_id for c in report.changes} == {
        "draft-answer", "escalate-clinician"}


def test_all_suite_verdicts_match_expected():
    for sc in build_suite():
        _, report = _report_for(sc.name)
        assert report.verdict == sc.expected_verdict, sc.name
        assert sorted(report.minimal_triggering_set) == sorted(
            sc.expected_minimal_set), sc.name
