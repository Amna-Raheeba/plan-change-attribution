"""Tests for the scenario suite and its ground truth."""

from plan_change_attribution.scenarios import build_suite


def test_suite_size_and_families():
    suite = build_suite()
    assert len(suite) == 30
    families = {}
    for sc in suite:
        families[sc.family] = families.get(sc.family, 0) + 1
    assert families == {"single": 12, "joint": 10, "none": 8}


def test_every_scenario_has_a_real_diff():
    for sc in build_suite():
        assert len(sc.changes()) >= 1, sc.name


def test_trigger_map_covers_every_change():
    for sc in build_suite():
        changed = {c.step_id for c in sc.changes()}
        assert set(sc.trigger_map) == changed, sc.name


def test_trigger_map_ids_are_real_observations():
    for sc in build_suite():
        ids = {o.id for o in sc.all_observations()}
        for trigger in sc.trigger_map.values():
            if trigger is not None:
                assert trigger in ids, sc.name


def test_none_family_has_no_trigger():
    for sc in build_suite():
        if sc.family == "none":
            assert all(v is None for v in sc.trigger_map.values())
            assert sc.expected_verdict == "no trigger found"


def test_joint_family_has_two_triggers():
    for sc in build_suite():
        if sc.family == "joint":
            triggers = {v for v in sc.trigger_map.values() if v}
            assert len(triggers) == 2, sc.name
            assert len(sc.expected_minimal_set) == 2


def test_single_family_has_one_trigger():
    for sc in build_suite():
        if sc.family == "single":
            triggers = {v for v in sc.trigger_map.values() if v}
            assert len(triggers) == 1, sc.name


def test_names_are_unique():
    names = [sc.name for sc in build_suite()]
    assert len(names) == len(set(names))
