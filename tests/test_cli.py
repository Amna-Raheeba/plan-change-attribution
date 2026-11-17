"""Tests for the CLI entry points."""

import json
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "plan_change_attribution", *args],
        capture_output=True, text=True, cwd=".")


def test_demo_exits_zero_and_shows_verdicts():
    r = _run("demo")
    assert r.returncode == 0, r.stderr
    assert "single trigger" in r.stdout
    assert "joint trigger" in r.stdout
    assert "no trigger found" in r.stdout


def test_evaluate_writes_summary(tmp_path):
    r = _run("evaluate", "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    assert summary["top1"] == 1.0
    assert (tmp_path / "evaluation.csv").exists()


def test_explain_known_scenario():
    r = _run("explain", "phm-vibration-failure-0")
    assert r.returncode == 0, r.stderr
    assert "sf-vibration-0" in r.stdout


def test_explain_unknown_scenario_fails():
    r = _run("explain", "no-such-scenario")
    assert r.returncode == 2
    assert "unknown scenario" in r.stderr
