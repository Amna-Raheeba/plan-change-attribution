"""Tests for the data models."""

from plan_change_attribution.models import Change, Observation, Plan, Step, Trace


def _step(sid):
    return Step(sid, f"Step {sid}", "act", "target")


def test_plan_step_lookup():
    plan = Plan("phm", (_step("a"), _step("b")))
    assert plan.get("a").name == "Step a"
    assert plan.get("missing") is None
    assert len(plan) == 2
    assert plan.step_ids() == ["a", "b"]


def test_step_defaults():
    step = Step("x", "Do x", "act")
    assert step.target == ""
    assert step.params == {}
    assert step.depends_on == ()


def test_observation_defaults():
    obs = Observation("o1", "heartbeat", "monitor", "OK")
    assert obs.t == 0
    assert obs.facts == {}


def test_change_describe():
    added = Change("added", "s1", None, _step("s1"))
    assert "added step s1" in added.describe()
    removed = Change("removed", "s1", _step("s1"), None)
    assert "removed step s1" in removed.describe()
    modified = Change("modified", "s1", _step("s1"), Step("s1", "New", "act"))
    assert "modified step s1" in modified.describe()
    moved = Change("moved", "s1", _step("s1"), _step("s1"))
    assert "moved step s1" in moved.describe()


def test_change_key_unique_per_kind():
    assert Change("added", "s").key() != Change("removed", "s").key()


def test_trace_visible_at():
    o1 = Observation("o1", "a", "s", "t", 1)
    o2 = Observation("o2", "b", "s", "t", 2)
    plan = Plan("phm", ())
    trace = Trace("phm", [o1, o2],
                  {"initial": (plan, ["o1"]), "replan": (plan, ["o1", "o2"])})
    assert [o.id for o in trace.visible_at("initial")] == ["o1"]
    assert [o.id for o in trace.visible_at("replan")] == ["o1", "o2"]
    assert trace.visible_at("missing") == []
    assert trace.observation("o2") is o2
    assert trace.observation("nope") is None
