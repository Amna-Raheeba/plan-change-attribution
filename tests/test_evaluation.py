"""Tests for the evaluation metrics."""

import csv
import json

from plan_change_attribution.evaluation import evaluate_suite, run_evaluation


def test_suite_metrics_are_perfect_on_synthetic_suite():
    # The suite is built so the method can explain every scenario;
    # these exact numbers are the measured result, asserted so a
    # regression in the attributor is caught immediately.
    out = evaluate_suite()
    m = out["metrics"]
    assert m["n_scenarios"] == 30
    assert m["n_changes"] == 70
    assert m["top1"] == 1.0
    assert m["top1_hits"] == 62
    assert m["verdict_accuracy"] == 1.0
    assert m["no_single_trigger_correct"] == 1.0
    assert m["joint_not_single_correct"] == 1.0
    assert m["false_trigger_rate"] == 0.0
    assert m["mean_triggering_set_size"] == 1.4545


def test_rows_cover_every_change():
    out = evaluate_suite()
    assert len(out["rows"]) == 70
    assert all("top_trigger" in r for r in out["rows"])


def test_run_evaluation_writes_files(tmp_path):
    summary = run_evaluation(tmp_path)
    assert (tmp_path / "evaluation.csv").exists()
    assert (tmp_path / "evaluation_summary.json").exists()
    with open(tmp_path / "evaluation_summary.json") as f:
        saved = json.load(f)
    assert saved["top1"] == summary["top1"]
    with open(tmp_path / "evaluation.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 70
