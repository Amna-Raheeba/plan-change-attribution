"""Per-change causal attribution by single-observation ablation.

Method
------
1. Freeze the trace at the replan point: the planner, the goal, the
   observations visible at the replan, and the context are all fixed.
2. Diff the initial plan and the replanned plan.
3. For every candidate observation (the observations that arrived
   between the first plan and the replan), remove it alone, re-run the
   planner, and count how many diff changes that removal repairs.
   That count is the observation's evidence score.
4. Rank candidates by score (ties broken by recency, then id, so the
   ranking is deterministic).
5. Greedy minimal triggering set: repeatedly add the observation whose
   removal repairs the most *remaining* changes, until the ablated
   plan equals the original plan or nothing improves. The set that
   restores the original plan is the minimal triggering set.

Verdicts
---------
- ``single trigger`` — one observation's removal restores the plan.
- ``joint trigger`` — two or more observations are jointly needed.
- ``no trigger found`` — no removal (single or greedy set) restores
  the plan, e.g. the replan was caused by internal state, not by any
  observation. This is an honest verdict, not a failure to look.

Nothing here is a guess: every score is a count of changes repaired
by an actual planner replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plan_change_attribution.diff import change_is_fixed_by, diff_plans
from plan_change_attribution.models import Change, Observation, Plan
from plan_change_attribution.planner import plan as run_planner


@dataclass
class TriggerScore:
    observation: Observation
    score: int  # number of diff changes repaired by removing it
    fixed_keys: list = field(default_factory=list)


@dataclass
class ChangeAttribution:
    change: Change
    ranked_triggers: list  # list[TriggerScore], best first
    top_trigger: Observation | None = None

    def top_trigger_id(self) -> str | None:
        return self.top_trigger.id if self.top_trigger else None


@dataclass
class AttributionReport:
    goal: str
    old_plan: Plan
    new_plan: Plan
    changes: list
    per_change: list  # list[ChangeAttribution]
    minimal_triggering_set: list  # observation ids, greedy order
    verdict: str  # "single trigger" | "joint trigger" | "no trigger found"
    candidates: list = field(default_factory=list)  # observation ids tried

    def summary(self) -> str:
        lines = [f"Plan diff: {len(self.changes)} change(s).",
                 f"Verdict: {self.verdict}."]
        if self.minimal_triggering_set:
            lines.append(
                "Minimal triggering set: "
                + ", ".join(self.minimal_triggering_set))
        for ca in self.per_change:
            top = ca.top_trigger_id() or "none"
            lines.append(f"  - {ca.change.describe()}: trigger {top}")
        return "\n".join(lines)


def _ablated_plan(goal, observations, context, remove_ids) -> Plan:
    kept = [o for o in observations if o.id not in set(remove_ids)]
    return run_planner(goal, kept, context)


def rank_triggers_for_change(goal, observations, context, old_plan,
                             new_plan, change, candidates) -> list:
    """Rank candidate observations by how much they explain one change.

    Score = 1 if removing the observation repairs this change, else 0,
    plus a tie-break bonus for repairing *other* changes too (an
    observation that explains more of the diff is the better trigger).
    """
    scored: list = []
    all_changes = diff_plans(old_plan, new_plan)
    for obs in candidates:
        ablated = _ablated_plan(goal, observations, context, [obs.id])
        fixes_this = change_is_fixed_by(change, ablated, old_plan)
        fixes_others = sum(
            1 for c in all_changes
            if c.key() != change.key()
            and change_is_fixed_by(c, ablated, old_plan)
        )
        score = (1 if fixes_this else 0) * 10 + fixes_others
        scored.append(TriggerScore(obs, score, []))
    # Best score first; ties: most recent observation first, then id.
    scored.sort(key=lambda s: (-s.score, -s.observation.t, s.observation.id))
    return scored


def greedy_minimal_set(goal, observations, context, old_plan,
                       new_plan, candidates) -> list:
    """Greedy smallest set of observations whose removal restores the plan.

    Returns observation ids in the order they were added. Empty list
    means nothing tried restored the plan.
    """
    remaining = list(candidates)
    chosen: list = []
    current_changes = diff_plans(old_plan, new_plan)
    if not current_changes:
        return []
    while remaining:
        best = None
        best_fixed = 0
        best_ablated = None
        for obs in remaining:
            trial_ids = chosen + [obs.id]
            ablated = _ablated_plan(goal, observations, context, trial_ids)
            fixed = sum(
                1 for c in current_changes
                if change_is_fixed_by(c, ablated, old_plan)
            )
            if fixed > best_fixed:
                best, best_fixed, best_ablated = obs, fixed, ablated
        if best is None or best_fixed == 0:
            break
        chosen.append(best.id)
        remaining.remove(best)
        if best_ablated == old_plan:
            return chosen
        # Recompute which changes are still unfixed for the next round.
        current_changes = [
            c for c in current_changes
            if not change_is_fixed_by(c, best_ablated, old_plan)
        ]
        if not current_changes:
            return chosen
    # Only return the set if it actually restores the old plan.
    ablated = _ablated_plan(goal, observations, context, chosen)
    return chosen if ablated == old_plan else []


def explain_replan(goal, observations, context, old_plan, new_plan,
                   candidates) -> AttributionReport:
    """Full attribution for one replan.

    ``observations`` is the full set visible at the replan;
    ``candidates`` is the subset that arrived after the first plan
    (the only ones that can have caused the change).
    """
    changes = diff_plans(old_plan, new_plan)
    per_change: list = []
    for change in changes:
        ranked = rank_triggers_for_change(
            goal, observations, context, old_plan, new_plan, change, candidates)
        top = ranked[0].observation if ranked and ranked[0].score > 0 else None
        per_change.append(ChangeAttribution(change, ranked, top))

    minimal = greedy_minimal_set(
        goal, observations, context, old_plan, new_plan, candidates)
    if len(minimal) == 1:
        verdict = "single trigger"
    elif len(minimal) > 1:
        verdict = "joint trigger"
    else:
        verdict = "no trigger found"
    return AttributionReport(
        goal=goal, old_plan=old_plan, new_plan=new_plan, changes=changes,
        per_change=per_change, minimal_triggering_set=minimal,
        verdict=verdict, candidates=[o.id for o in candidates])
