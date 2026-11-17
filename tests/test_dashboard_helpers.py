"""Tests for the dashboard helpers (no Streamlit needed)."""

import pytest

from dashboard.helpers import (
    attribution_for, diff_rows, metric_cards, plan_table, scenario_by_name,
    scenario_names, trigger_rows,
)
from plan_change_attribution.evaluation import evaluate_suite

try:
    from streamlit.testing.v1 import AppTest
    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False


def test_scenario_names_cover_suite():
    names = scenario_names()
    assert len(names) == 30
    assert "phm-vibration-failure-0" in names
    assert scenario_by_name("nope") is None


def test_plan_table_rows():
    sc = scenario_by_name("phm-vibration-failure-0")
    report = attribution_for(sc)
    rows = plan_table(report.old_plan)
    assert rows[0] == {"step": "collect-vibration",
                       "name": "Collect vibration data",
                       "action": "collect",
                       "target": "vibration-sensor"}


def test_diff_rows_include_top_trigger():
    sc = scenario_by_name("phm-vibration-failure-0")
    report = attribution_for(sc)
    rows = diff_rows(report)
    assert len(rows) == 4
    assert all(r["top trigger"] == "sf-vibration-0" for r in rows)


def test_trigger_rows_cover_all_candidates():
    sc = scenario_by_name("phm-temp-failure-plus-anomaly-0")
    report = attribution_for(sc)
    rows = trigger_rows(report)
    candidates = {r["candidate"] for r in rows}
    assert candidates == {"sf-temperature-0", "anom-0"}
    best = max(rows, key=lambda r: r["score"])
    assert best["score"] > 0


def test_metric_cards():
    metrics = evaluate_suite()["metrics"]
    cards = dict(metric_cards(metrics))
    assert cards["Top-1 trigger accuracy"] == "100%"
    assert cards["Verdict accuracy"] == "100%"
    assert cards["False-trigger rate"] == "0%"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit AppTest unavailable")
def test_dashboard_runs_without_exceptions():
    # Hardened 2026-11-17 (v0.2.0): AppTest's default timeout can be
    # exceeded on a cold first run under a throttled shared CPU — the
    # suite failed this test once in ~19 runs with no app error. An
    # explicit 120 s timeout plus one fresh re-run covers the cold
    # start; a genuine app exception fails both runs. The helpers the
    # app renders are tested independently above.
    from pathlib import Path
    app = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
    at = AppTest.from_file(app).run(timeout=120)
    if at.exception:
        at = AppTest.from_file(app).run(timeout=120)
    assert not at.exception


def test_stability_rows_cover_changes():
    from dashboard.helpers import stability_for, stability_rows
    sc = scenario_by_name("phm-vibration-failure-0")
    stab = stability_for(sc)
    rows = stability_rows(stab)
    assert len(rows) == 4
    assert all(r["stability"] == "1.00" for r in rows)
    assert all(r["verdict"] == "confident" for r in rows)
    assert all(r["top trigger"] == "sf-vibration-0" for r in rows)
    assert stab.minimal_set_stability == 1.0
    assert stab.duplicate_survival is not None
