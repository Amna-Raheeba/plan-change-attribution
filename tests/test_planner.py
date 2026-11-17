"""Tests for the deterministic mock planner."""

import pytest

from plan_change_attribution.models import Observation
from plan_change_attribution.planner import plan


def _obs(obs_id, kind, t=1, facts=None):
    return Observation(obs_id, kind, "src", "text", t, facts or {})


def test_phm_base_plan():
    p = plan("phm", [])
    assert p.step_ids() == ["collect-vibration", "collect-temperature",
                            "run-rul", "explain", "draft-report"]


def test_healthcare_base_plan():
    p = plan("healthcare", [])
    assert p.step_ids() == ["retrieve-guidelines", "retrieve-trials",
                            "draft-answer", "verify-citations",
                            "add-disclaimer"]


def test_planner_is_deterministic():
    obs = [_obs("a", "cost_spike", 2), _obs("b", "vibration_anomaly", 1)]
    assert plan("phm", obs) == plan("phm", list(reversed(obs)))
    assert plan("phm", obs) == plan("phm", obs)


def test_unknown_goal_raises():
    with pytest.raises(ValueError):
        plan("no-such-domain", [])


def test_sensor_failure_drops_dependent_steps_and_adds_fallback():
    obs = [_obs("sf", "sensor_failure", 1, {"sensor": "vibration"})]
    p = plan("phm", obs)
    assert "collect-vibration" not in p.step_ids()
    assert "run-rul" not in p.step_ids()
    assert "explain" not in p.step_ids()
    assert "collect-temperature" in p.step_ids()
    assert p.step_ids()[0] == "use-fallback-sensor"
    assert p.get("use-fallback-sensor").params["failed"] == ["vibration"]


def test_cost_spike_replaces_model():
    p = plan("phm", [_obs("c", "cost_spike")])
    assert p.get("run-rul").params["model"] == "rule-based"
    assert p.get("run-rul").name.startswith("Run rule-based")


def test_vibration_anomaly_adds_alert():
    p = plan("phm", [_obs("a", "vibration_anomaly")])
    assert "raise-alert" in p.step_ids()
    assert p.get("raise-alert").params["severity"] == "high"


def test_contradiction_drops_guideline_step_and_adds_clarification():
    obs = [_obs("ct", "contradiction", 1, {"guideline": "G2"})]
    p = plan("healthcare", obs)
    assert "retrieve-guidelines" not in p.step_ids()
    assert "request-clarification" in p.step_ids()
    assert p.get("request-clarification").params["guidelines"] == ["G2"]


def test_emergency_escalates_and_drops_draft():
    p = plan("healthcare", [_obs("e", "emergency_flag")])
    assert "draft-answer" not in p.step_ids()
    assert p.step_ids()[0] == "escalate-clinician"


def test_new_evidence_adds_reverification():
    p = plan("healthcare", [_obs("n", "new_evidence")])
    assert "reverify-citations" in p.step_ids()


def test_budget_context_drops_lowest_priority_step():
    p = plan("phm", [], {"step_budget_exhausted": True})
    assert "draft-report" not in p.step_ids()
    assert len(p) == 4
    p2 = plan("healthcare", [], {"step_budget_exhausted": True})
    assert "add-disclaimer" not in p2.step_ids()


def test_unknown_observation_kind_is_ignored():
    p = plan("phm", [_obs("h", "heartbeat")])
    assert p.step_ids() == ["collect-vibration", "collect-temperature",
                            "run-rul", "explain", "draft-report"]
