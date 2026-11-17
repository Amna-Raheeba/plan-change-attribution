"""Dashboard helpers — pure functions, no Streamlit import.

Kept import-free so the test suite can exercise them without a
browser session.
"""

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.scenarios import build_suite
from plan_change_attribution.stability import score_stability


def scenario_names():
    return [sc.name for sc in build_suite()]


def scenario_by_name(name):
    for sc in build_suite():
        if sc.name == name:
            return sc
    return None


def attribution_for(scenario):
    old, new = scenario.old_plan(), scenario.new_plan()
    return explain_replan(
        scenario.goal, scenario.all_observations(), scenario.context,
        old, new, scenario.candidates())


def stability_for(scenario, n_replays=20, n_duplicate=5):
    """Trigger-stability report for one scenario (pure function)."""
    old, new = scenario.old_plan(), scenario.new_plan()
    return score_stability(
        scenario.goal, scenario.all_observations(), scenario.context,
        old, new, scenario.candidates(),
        n_replays=n_replays, n_duplicate=n_duplicate)


def stability_rows(stability_report):
    """One row per change: trigger, survival rate, verdict."""
    return [{
        "change": ts.change_key,
        "detail": ts.change_desc,
        "top trigger": ts.trigger_id or "—",
        "stability": f"{ts.stability:.2f}",
        "verdict": ts.verdict,
    } for ts in stability_report.per_change]


def plan_table(plan):
    """Rows for a plan table: step id, name, action, target."""
    return [{
        "step": s.id,
        "name": s.name,
        "action": s.action,
        "target": s.target,
    } for s in plan.steps]


def diff_rows(report):
    return [{
        "change": c.change.kind,
        "step": c.change.step_id,
        "detail": c.change.describe(),
        "top trigger": c.top_trigger_id() or "—",
    } for c in report.per_change]


def trigger_rows(report):
    """One row per (change, candidate) with its evidence score."""
    rows = []
    for ca in report.per_change:
        for ts in ca.ranked_triggers:
            rows.append({
                "change": ca.change.key(),
                "candidate": ts.observation.id,
                "kind": ts.observation.kind,
                "score": ts.score,
            })
    return rows


def metric_cards(metrics):
    return [
        ("Top-1 trigger accuracy", f"{metrics['top1']:.0%}"
         if metrics["top1"] is not None else "—"),
        ("Verdict accuracy", f"{metrics['verdict_accuracy']:.0%}"),
        ("Mean triggering-set size",
         f"{metrics['mean_triggering_set_size']:.2f}"),
        ("No-trigger correct", f"{metrics['no_single_trigger_correct']:.0%}"),
        ("False-trigger rate", f"{metrics['false_trigger_rate']:.0%}"),
    ]
