"""CLI: ``python -m plan_change_attribution <demo|evaluate|explain|stability>``."""

from __future__ import annotations

import argparse
import json
import sys

from plan_change_attribution.attribution import explain_replan
from plan_change_attribution.evaluation import evaluate_suite, run_evaluation
from plan_change_attribution.scenarios import build_suite
from plan_change_attribution.stability import run_stability_evaluation


def _show_scenario(sc) -> None:
    old, new = sc.old_plan(), sc.new_plan()
    report = explain_replan(
        sc.goal, sc.all_observations(), sc.context, old, new, sc.candidates())
    print(f"== {sc.name} [{sc.family}] ==")
    print("Initial plan:", [s.id for s in old.steps])
    print("Replanned   :", [s.id for s in new.steps])
    print(report.summary())
    print()


def cmd_demo(_args) -> int:
    suite = build_suite()
    for sc in suite:
        if sc.name in ("phm-vibration-failure-0",
                       "health-contradiction-G1-0",
                       "phm-temp-failure-plus-anomaly-0",
                       "phm-budget-exhausted-0"):
            _show_scenario(sc)
    return 0


def cmd_evaluate(args) -> int:
    summary = run_evaluation(args.out)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_stability(args) -> int:
    summary = run_stability_evaluation(
        args.out, n_replays=args.replays, n_duplicate=args.duplicates)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_explain(args) -> int:
    suite = {sc.name: sc for sc in build_suite()}
    sc = suite.get(args.scenario)
    if sc is None:
        print(f"unknown scenario: {args.scenario}", file=sys.stderr)
        print("available:", ", ".join(sorted(suite)), file=sys.stderr)
        return 2
    _show_scenario(sc)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="plan-change-attribution")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="show attribution on 4 example scenarios")
    p_eval = sub.add_parser("evaluate", help="run the 30-scenario suite")
    p_eval.add_argument("--out", default="results")
    p_exp = sub.add_parser("explain", help="attribute one named scenario")
    p_exp.add_argument("scenario")
    p_stab = sub.add_parser(
        "stability", help="score trigger stability over perturbed replays")
    p_stab.add_argument("--out", default="results")
    p_stab.add_argument("--replays", type=int, default=20)
    p_stab.add_argument("--duplicates", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "stability":
        return cmd_stability(args)
    return cmd_explain(args)


if __name__ == "__main__":
    raise SystemExit(main())
