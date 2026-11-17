"""Deterministic mock planner for Plan-and-Execute agents.

The planner is a pure function of ``(goal, observations, context)``:
same inputs always give the same plan. That is what makes ablation
replays meaningful — removing one observation and re-running the
planner is a genuine counterfactual, not a fresh roll of the dice.

Two demo domains are built in:

- ``phm`` — predictive-maintenance agent for IT infrastructure.
- ``healthcare`` — research agent answering a clinical question.

Rules are evaluated in a fixed order, so plan construction is fully
deterministic. ``context`` carries internal state that is NOT an
observation (for example a step budget that ran out); a replan caused
by context alone has no observation trigger, which is exactly the
"no single trigger" case the attributor must report honestly.
"""

from __future__ import annotations

from plan_change_attribution.models import Observation, Plan, Step

# ---------------------------------------------------------------- PHM domain

PHM_BASE = [
    Step("collect-vibration", "Collect vibration data", "collect",
         "vibration-sensor", depends_on=("vibration",)),
    Step("collect-temperature", "Collect temperature data", "collect",
         "temperature-sensor", depends_on=("temperature",)),
    Step("run-rul", "Run LSTM remaining-useful-life model", "predict",
         "rul-model", params={"model": "lstm"}, depends_on=("vibration", "temperature")),
    Step("explain", "Explain prediction with SHAP", "explain",
         "shap", depends_on=("vibration", "temperature")),
    Step("draft-report", "Draft maintenance report", "write",
         "report", depends_on=()),
]


def _plan_phm(observations: list, context: dict) -> Plan:
    obs_by_kind: dict = {}
    for obs in observations:
        obs_by_kind.setdefault(obs.kind, []).append(obs)

    steps: list = []
    failed_sensors = {
        o.facts.get("sensor")
        for o in obs_by_kind.get("sensor_failure", [])
        if o.facts.get("sensor")
    }

    for step in PHM_BASE:
        # A step depending on a failed sensor is dropped...
        if failed_sensors and any(d in failed_sensors for d in step.depends_on):
            continue
        # ...except the report, which is shortened instead of dropped.
        steps.append(step)

    if failed_sensors:
        # Fallback step, inserted where the first dropped step was.
        steps.insert(
            0,
            Step("use-fallback-sensor", "Switch to fallback sensor feed",
                 "switch", "fallback-sensor",
                 params={"failed": sorted(failed_sensors)}),
        )

    if obs_by_kind.get("vibration_anomaly"):
        steps.append(
            Step("raise-alert", "Raise inspection alert", "alert",
                 "maintenance-team",
                 params={"severity": "high"}),
        )

    if obs_by_kind.get("cost_spike"):
        # Premium LSTM replaced by a cheap rule-based model.
        steps = [
            Step("run-rul", "Run rule-based remaining-useful-life model",
                 "predict", "rul-model", params={"model": "rule-based"},
                 depends_on=("vibration", "temperature"))
            if s.id == "run-rul" else s
            for s in steps
        ]

    if context.get("step_budget_exhausted"):
        # Internal cause: the budget ran out, so the lowest-priority
        # step (the report) is dropped. No observation caused this.
        steps = [s for s in steps if s.id != "draft-report"]

    return Plan(goal="phm", steps=tuple(steps))


# ---------------------------------------------------------- healthcare domain

HEALTH_BASE = [
    Step("retrieve-guidelines", "Retrieve clinical guidelines", "retrieve",
         "guideline-db", depends_on=("guideline",)),
    Step("retrieve-trials", "Retrieve trial evidence", "retrieve",
         "trial-db", depends_on=("trial",)),
    Step("draft-answer", "Draft answer with citations", "write",
         "answer", depends_on=("guideline", "trial")),
    Step("verify-citations", "Verify every citation", "verify",
         "citation-checker", depends_on=("guideline", "trial")),
    Step("add-disclaimer", "Add medical disclaimer", "write",
         "disclaimer", depends_on=()),
]


def _plan_healthcare(observations: list, context: dict) -> Plan:
    obs_by_kind: dict = {}
    for obs in observations:
        obs_by_kind.setdefault(obs.kind, []).append(obs)

    steps: list = list(HEALTH_BASE)

    contradictions = obs_by_kind.get("contradiction", [])
    if contradictions:
        # The guideline the contradiction is about can no longer be
        # trusted: drop the step that uses it, ask for clarification.
        bad = {o.facts.get("guideline") for o in contradictions
               if o.facts.get("guideline")}
        steps = [
            s for s in steps
            if not (s.id == "retrieve-guidelines" and bad)
        ]
        steps.insert(
            1,
            Step("request-clarification",
                 "Request clarification on conflicting guidance",
                 "clarify", "clinical-team",
                 params={"guidelines": sorted(bad)}),
        )

    if obs_by_kind.get("new_evidence"):
        steps.append(
            Step("reverify-citations", "Re-verify citations against new evidence",
                 "verify", "citation-checker"),
        )

    if obs_by_kind.get("emergency_flag"):
        # Safety rule: escalate and do not draft an answer.
        steps = [s for s in steps if s.id != "draft-answer"]
        steps.insert(
            0,
            Step("escalate-clinician", "Escalate to clinician immediately",
                 "escalate", "on-call-clinician",
                 params={"severity": "urgent"}),
        )

    if context.get("step_budget_exhausted"):
        steps = [s for s in steps if s.id != "add-disclaimer"]

    return Plan(goal="healthcare", steps=tuple(steps))


PLANNERS = {"phm": _plan_phm, "healthcare": _plan_healthcare}


def plan(goal: str, observations: list, context: dict | None = None) -> Plan:
    """Build a plan deterministically from goal + observations + context.

    ``observations`` may be any iterable; they are sorted by ``t``
    (then id) before planning so input order never changes the result.
    """
    fn = PLANNERS.get(goal)
    if fn is None:
        raise ValueError(f"unknown goal/domain: {goal!r}")
    ordered = sorted(observations, key=lambda o: (o.t, o.id))
    return fn(ordered, context or {})
