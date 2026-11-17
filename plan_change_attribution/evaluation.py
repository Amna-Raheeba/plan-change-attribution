"""Evaluation: run the whole scenario suite and score attribution.

Metrics (all measured on the 30-scenario suite, nothing invented):
- ``top1`` — over changes with a known single trigger (single +
  joint families): fraction whose true trigger is ranked #1.
- ``mean_triggering_set_size`` — mean size of the greedy minimal
  triggering set over single + joint scenarios that were explained
  (verdict single or joint).
- ``no_single_trigger_correct`` — over the ``none`` family: fraction
  correctly reported as "no trigger found".
- ``joint_not_single_correct`` — over the ``joint`` family: fraction
  correctly NOT reported as a single trigger.
- ``verdict_accuracy`` — fraction of scenarios whose verdict equals
  the expected verdict.
- ``false_trigger_rate`` — over single-family changes: fraction where
  the top-ranked trigger exists but is not the true trigger.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.scenarios import build_suite


def evaluate_suite(scenarios=None) -> dict:
    scenarios = scenarios if scenarios is not None else build_suite()
    rows: list = []
    top1_hits = top1_total = 0
    false_triggers = 0
    set_sizes: list = []
    none_correct = none_total = 0
    joint_not_single = joint_total = 0
    verdict_hits = 0

    for sc in scenarios:
        old = sc.old_plan()
        new = sc.new_plan()
        report = explain_replan(
            sc.goal, sc.all_observations(), sc.context,
            old, new, sc.candidates())
        verdict_ok = report.verdict == sc.expected_verdict
        verdict_hits += int(verdict_ok)
        if sc.family == "none":
            none_total += 1
            none_correct += int(report.verdict == "no trigger found")
        if sc.family == "joint":
            joint_total += 1
            joint_not_single += int(report.verdict != "single trigger")
        if sc.family in ("single", "joint") and report.verdict in (
                "single trigger", "joint trigger"):
            set_sizes.append(len(report.minimal_triggering_set))
        for ca in report.per_change:
            truth = sc.trigger_map.get(ca.change.step_id)
            top = ca.top_trigger_id()
            if truth is None:
                hit = top is None
                rows.append({
                    "scenario": sc.name, "family": sc.family,
                    "change": ca.change.key(), "truth": "",
                    "top_trigger": top or "", "verdict": report.verdict,
                    "verdict_expected": sc.expected_verdict,
                    "top1_hit": int(hit),
                    "minimal_set": ";".join(report.minimal_triggering_set),
                })
                continue
            top1_total += 1
            hit = top == truth
            top1_hits += int(hit)
            if top is not None and top != truth:
                false_triggers += 1
            rows.append({
                "scenario": sc.name, "family": sc.family,
                "change": ca.change.key(), "truth": truth,
                "top_trigger": top or "", "verdict": report.verdict,
                "verdict_expected": sc.expected_verdict,
                "top1_hit": int(hit),
                "minimal_set": ";".join(report.minimal_triggering_set),
            })

    single_changes = sum(1 for r in rows if r["family"] == "single"
                         and r["truth"])
    metrics = {
        "n_scenarios": len(scenarios),
        "n_changes": len(rows),
        "top1": round(top1_hits / top1_total, 4) if top1_total else None,
        "top1_hits": top1_hits,
        "top1_total": top1_total,
        "mean_triggering_set_size": (
            round(sum(set_sizes) / len(set_sizes), 4) if set_sizes else None),
        "n_explained_scenarios": len(set_sizes),
        "no_single_trigger_correct": (
            round(none_correct / none_total, 4) if none_total else None),
        "no_single_trigger_n": none_total,
        "joint_not_single_correct": (
            round(joint_not_single / joint_total, 4) if joint_total else None),
        "joint_n": joint_total,
        "verdict_accuracy": round(verdict_hits / len(scenarios), 4),
        "false_trigger_rate": (
            round(false_triggers / single_changes, 4)
            if single_changes else None),
        "false_triggers": false_triggers,
        "single_changes": single_changes,
    }
    return {"metrics": metrics, "rows": rows}


def run_evaluation(results_dir: str | Path = "results") -> dict:
    """Run the suite and write results/evaluation.csv + summary."""
    out = evaluate_suite()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scenario", "family", "change", "truth",
                           "top_trigger", "verdict", "verdict_expected",
                           "top1_hit", "minimal_set"])
        writer.writeheader()
        writer.writerows(out["rows"])
    summary = dict(out["metrics"])
    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
