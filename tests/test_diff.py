"""Tests for the plan diff."""

from plan_change_attribution.diff import change_is_fixed_by, diff_plans
from plan_change_attribution.models import Plan, Step


def _plan(ids, modified=None):
    steps = []
    for sid in ids:
        name = (modified or {}).get(sid, f"Step {sid}")
        params = {"model": "rule-based"} if sid in (modified or {}) else {}
        steps.append(Step(sid, name, "act", "t", params))
    return Plan("phm", tuple(steps))


def test_empty_diff_for_identical_plans():
    assert diff_plans(_plan(["a", "b"]), _plan(["a", "b"])) == []


def test_added_and_removed():
    changes = diff_plans(_plan(["a", "b"]), _plan(["b", "c"]))
    kinds = {c.key(): c.kind for c in changes}
    assert kinds == {"removed:a": "removed", "added:c": "added"}


def test_modified_step():
    changes = diff_plans(_plan(["a"]), _plan(["a"], {"a": "Step a v2"}))
    assert len(changes) == 1
    assert changes[0].kind == "modified"
    assert changes[0].old_step.name == "Step a"
    assert changes[0].new_step.name == "Step a v2"


def test_reorder_is_a_move():
    changes = diff_plans(_plan(["a", "b", "c"]), _plan(["b", "a", "c"]))
    assert {c.key() for c in changes} == {"moved:a", "moved:b"}


def test_add_remove_does_not_create_spurious_moves():
    # Removing the first step shifts absolute positions, but the
    # relative order of the rest is unchanged: no "moved" changes.
    changes = diff_plans(_plan(["a", "b", "c"]), _plan(["b", "c"]))
    assert [c.kind for c in changes] == ["removed"]
    changes = diff_plans(_plan(["a", "b"]), _plan(["x", "a", "b"]))
    assert [c.kind for c in changes] == ["added"]


def test_change_fixed_by_added():
    old, new = _plan(["a"]), _plan(["a", "b"])
    change = diff_plans(old, new)[0]
    assert change_is_fixed_by(change, old, old) is True
    assert change_is_fixed_by(change, new, old) is False


def test_change_fixed_by_removed():
    old, new = _plan(["a", "b"]), _plan(["b"])
    change = diff_plans(old, new)[0]
    assert change_is_fixed_by(change, old, old) is True
    assert change_is_fixed_by(change, new, old) is False


def test_change_fixed_by_modified():
    old = _plan(["a"])
    new = _plan(["a"], {"a": "Step a v2"})
    change = diff_plans(old, new)[0]
    assert change_is_fixed_by(change, old, old) is True
    assert change_is_fixed_by(change, new, old) is False


def test_change_fixed_by_moved():
    old = _plan(["a", "b", "c"])
    new = _plan(["b", "a", "c"])
    change = [c for c in diff_plans(old, new) if c.step_id == "a"][0]
    assert change_is_fixed_by(change, old, old) is True
    assert change_is_fixed_by(change, new, old) is False
