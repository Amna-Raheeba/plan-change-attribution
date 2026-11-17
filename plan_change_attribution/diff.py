"""Plan diff: what changed between two plans, step by step."""

from __future__ import annotations

from plan_change_attribution.models import Change, Plan


def diff_plans(old: Plan, new: Plan) -> list:
    """Return the list of changes from ``old`` to ``new``.

    Steps are matched by id. A step present in both with different
    action/target/params is "modified"; a step present in both at a
    different position (and otherwise identical) is "moved".
    """
    changes: list = []
    old_ids = old.step_ids()
    new_ids = new.step_ids()
    # "Moved" means the RELATIVE order changed, not the absolute index:
    # adding or removing a step shifts absolute positions of everything
    # after it, and that shift is not a change of its own.
    common = [sid for sid in old_ids if sid in set(new_ids)]
    old_rel = {sid: i for i, sid in enumerate(common)}
    new_common = [sid for sid in new_ids if sid in set(old_ids)]
    new_rel = {sid: i for i, sid in enumerate(new_common)}

    for step in new.steps:
        old_step = old.get(step.id)
        if old_step is None:
            changes.append(Change("added", step.id, None, step))
        elif (old_step.action, old_step.target, old_step.params) != (
            step.action, step.target, step.params,
        ):
            changes.append(Change("modified", step.id, old_step, step))
        elif old_rel[step.id] != new_rel[step.id]:
            changes.append(Change("moved", step.id, old_step, step))

    for step in old.steps:
        if new.get(step.id) is None:
            changes.append(Change("removed", step.id, step, None))

    # Stable order: removals first, then modifications, additions, moves.
    order = {"removed": 0, "modified": 1, "added": 2, "moved": 3}
    return sorted(changes, key=lambda c: (order[c.kind], c.step_id))


def change_is_fixed_by(change: Change, ablated: Plan, old: Plan) -> bool:
    """Did the ablated plan restore the old plan *for this change*?

    - added step: fixed if the step is absent from the ablated plan.
    - removed step: fixed if the step is back, identical to the old one.
    - modified step: fixed if the step matches the old version again.
    - moved step: fixed if the step is back at its old position.
    """
    step = ablated.get(change.step_id)
    if change.kind == "added":
        return step is None
    if step is None:
        return False
    if change.kind == "removed":
        return (step.action, step.target, step.params) == (
            change.old_step.action, change.old_step.target, change.old_step.params)
    if change.kind == "modified":
        return (step.action, step.target, step.params) == (
            change.old_step.action, change.old_step.target, change.old_step.params)
    # moved: back to the same RELATIVE position among the steps the
    # ablated and old plans share.
    common_old = [s for s in old.step_ids() if ablated.get(s) is not None]
    common_new = [s for s in ablated.step_ids() if old.get(s) is not None]
    return (common_old.index(change.step_id)
            == common_new.index(change.step_id))
