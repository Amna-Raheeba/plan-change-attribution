"""Scenario suite with known ground truth.

Every scenario is built, not guessed: the builder plants the trigger
observation(s) that cause the replan, so the true cause of each plan
change is known before attribution runs. Three families:

- ``single`` — one new observation causes every change.
- ``joint`` — two new observations, each causing a distinct subset
  of the changes; neither alone restores the original plan.
- ``none`` — the replan is caused by internal state (the step budget
  ran out), not by any observation; 1-2 distractor observations that
  do not affect the plan are planted to make the test honest.

``trigger_map`` maps each changed step id to the observation id that
caused that change, or ``None`` when no observation caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plan_change_attribution.diff import diff_plans
from plan_change_attribution.models import Observation, Plan
from plan_change_attribution.planner import plan as run_planner


@dataclass
class Scenario:
    name: str
    domain: str
    goal: str
    initial_obs: list
    new_obs: list
    context: dict = field(default_factory=dict)
    family: str = "single"  # "single" | "joint" | "none"
    trigger_map: dict = field(default_factory=dict)  # step_id -> obs id | None
    expected_verdict: str = "single trigger"
    expected_minimal_set: list = field(default_factory=list)

    def old_plan(self) -> Plan:
        return run_planner(self.goal, self.initial_obs, {})

    def new_plan(self) -> Plan:
        return run_planner(
            self.goal, self.initial_obs + self.new_obs, self.context)

    def all_observations(self) -> list:
        return self.initial_obs + self.new_obs

    def candidates(self) -> list:
        return list(self.new_obs)

    def changes(self) -> list:
        return diff_plans(self.old_plan(), self.new_plan())


def _obs(obs_id, kind, source, text, t, facts=None):
    return Observation(obs_id, kind, source, text, t, facts or {})


def _distractor(i, t):
    return _obs(f"dist-{i}", "heartbeat", "monitor",
                "Heartbeat OK, all feeds nominal.", t)


# ------------------------------------------------------------- single family

def _single_phm_sensor(sensor, i):
    obs = _obs(f"sf-{sensor}-{i}", "sensor_failure", f"{sensor}-sensor",
               f"{sensor} sensor feed failed.", 10 + i,
               {"sensor": sensor})
    affected = {"vibration": ["collect-vibration", "run-rul", "explain"],
                "temperature": ["collect-temperature", "run-rul", "explain"]}[sensor]
    return Scenario(
        name=f"phm-{sensor}-failure-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[obs, _distractor(i, 11 + i)],
        family="single",
        trigger_map={sid: obs.id for sid in affected + ["use-fallback-sensor"]},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


def _single_phm_cost(i):
    obs = _obs(f"cost-{i}", "cost_spike", "billing-api",
               "LLM inference cost spiked 4x.", 10 + i)
    return Scenario(
        name=f"phm-cost-spike-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[obs],
        family="single", trigger_map={"run-rul": obs.id},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


def _single_phm_anomaly(i):
    obs = _obs(f"anom-{i}", "vibration_anomaly", "vibration-sensor",
               "Vibration amplitude above threshold.", 10 + i)
    return Scenario(
        name=f"phm-vibration-anomaly-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[obs],
        family="single", trigger_map={"raise-alert": obs.id},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


def _single_health_contradiction(guideline, i):
    obs = _obs(f"contra-{guideline}-{i}", "contradiction", "guideline-db",
               f"Guideline {guideline} contradicts the retrieved answer.",
               10 + i, {"guideline": guideline})
    return Scenario(
        name=f"health-contradiction-{guideline}-{i}", domain="healthcare",
        goal="healthcare", initial_obs=[], new_obs=[obs],
        family="single",
        trigger_map={"retrieve-guidelines": obs.id,
                     "request-clarification": obs.id},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


def _single_health_emergency(i):
    obs = _obs(f"emerg-{i}", "emergency_flag", "triage-tool",
               "Red-flag symptoms present.", 10 + i)
    return Scenario(
        name=f"health-emergency-{i}", domain="healthcare", goal="healthcare",
        initial_obs=[], new_obs=[obs],
        family="single",
        trigger_map={"draft-answer": obs.id, "escalate-clinician": obs.id},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


def _single_health_evidence(i):
    obs = _obs(f"evid-{i}", "new_evidence", "trial-db",
               "New trial published after the first retrieval.", 10 + i)
    return Scenario(
        name=f"health-new-evidence-{i}", domain="healthcare",
        goal="healthcare", initial_obs=[], new_obs=[obs],
        family="single", trigger_map={"reverify-citations": obs.id},
        expected_verdict="single trigger", expected_minimal_set=[obs.id])


# -------------------------------------------------------------- joint family

def _joint_phm_temp_anomaly(i):
    sf = _obs(f"sf-temperature-{i}", "sensor_failure", "temperature-sensor",
              "Temperature sensor feed failed.", 10 + i,
              {"sensor": "temperature"})
    an = _obs(f"anom-{i}", "vibration_anomaly", "vibration-sensor",
              "Vibration amplitude above threshold.", 11 + i)
    return Scenario(
        name=f"phm-temp-failure-plus-anomaly-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[sf, an],
        family="joint",
        trigger_map={"collect-temperature": sf.id, "run-rul": sf.id,
                     "explain": sf.id, "use-fallback-sensor": sf.id,
                     "raise-alert": an.id},
        expected_verdict="joint trigger",
        expected_minimal_set=sorted([sf.id, an.id]))


def _joint_phm_anomaly_cost(i):
    an = _obs(f"anom-{i}", "vibration_anomaly", "vibration-sensor",
              "Vibration amplitude above threshold.", 10 + i)
    co = _obs(f"cost-{i}", "cost_spike", "billing-api",
              "LLM inference cost spiked 4x.", 11 + i)
    return Scenario(
        name=f"phm-anomaly-plus-cost-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[an, co],
        family="joint",
        trigger_map={"raise-alert": an.id, "run-rul": co.id},
        expected_verdict="joint trigger",
        expected_minimal_set=sorted([an.id, co.id]))


def _joint_health_contra_evidence(i):
    ct = _obs(f"contra-g2-{i}", "contradiction", "guideline-db",
              "Guideline G2 contradicts the retrieved answer.", 10 + i,
              {"guideline": "G2"})
    ev = _obs(f"evid-{i}", "new_evidence", "trial-db",
              "New trial published after the first retrieval.", 11 + i)
    return Scenario(
        name=f"health-contradiction-plus-evidence-{i}", domain="healthcare",
        goal="healthcare", initial_obs=[], new_obs=[ct, ev],
        family="joint",
        trigger_map={"retrieve-guidelines": ct.id,
                     "request-clarification": ct.id,
                     "reverify-citations": ev.id},
        expected_verdict="joint trigger",
        expected_minimal_set=sorted([ct.id, ev.id]))


def _joint_health_contra_emergency(i):
    ct = _obs(f"contra-g1-{i}", "contradiction", "guideline-db",
              "Guideline G1 contradicts the retrieved answer.", 10 + i,
              {"guideline": "G1"})
    em = _obs(f"emerg-{i}", "emergency_flag", "triage-tool",
              "Red-flag symptoms present.", 11 + i)
    return Scenario(
        name=f"health-contradiction-plus-emergency-{i}", domain="healthcare",
        goal="healthcare", initial_obs=[], new_obs=[ct, em],
        family="joint",
        trigger_map={"retrieve-guidelines": ct.id,
                     "request-clarification": ct.id,
                     "draft-answer": em.id, "escalate-clinician": em.id},
        expected_verdict="joint trigger",
        expected_minimal_set=sorted([ct.id, em.id]))


# --------------------------------------------------------------- none family

def _none_phm(i):
    return Scenario(
        name=f"phm-budget-exhausted-{i}", domain="phm", goal="phm",
        initial_obs=[], new_obs=[_distractor(i, 10 + i)],
        context={"step_budget_exhausted": True},
        family="none", trigger_map={"draft-report": None},
        expected_verdict="no trigger found", expected_minimal_set=[])


def _none_health(i):
    return Scenario(
        name=f"health-budget-exhausted-{i}", domain="healthcare",
        goal="healthcare", initial_obs=[],
        new_obs=[_distractor(i, 10 + i)],
        context={"step_budget_exhausted": True},
        family="none", trigger_map={"add-disclaimer": None},
        expected_verdict="no trigger found", expected_minimal_set=[])


def build_suite() -> list:
    """All 30 scenarios: 12 single, 10 joint, 8 none."""
    suite: list = []
    suite += [_single_phm_sensor("vibration", i) for i in range(2)]
    suite += [_single_phm_sensor("temperature", i) for i in range(2)]
    suite += [_single_phm_cost(i) for i in range(2)]
    suite += [_single_phm_anomaly(i) for i in range(2)]
    suite += [_single_health_contradiction("G1", i) for i in range(2)]
    suite += [_single_health_emergency(0), _single_health_evidence(0)]
    suite += [_joint_phm_temp_anomaly(i) for i in range(3)]
    suite += [_joint_phm_anomaly_cost(i) for i in range(3)]
    suite += [_joint_health_contra_evidence(i) for i in range(2)]
    suite += [_joint_health_contra_emergency(i) for i in range(2)]
    suite += [_none_phm(i) for i in range(4)]
    suite += [_none_health(i) for i in range(4)]
    return suite
