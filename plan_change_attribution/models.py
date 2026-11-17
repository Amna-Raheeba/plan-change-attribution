"""Data models: steps, plans, observations, traces.

Everything is a plain dataclass with no external dependencies, so a
trace can be serialised to JSON, replayed, and ablated deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """One step of an agent plan.

    ``depends_on`` names the observation kinds this step relies on
    (for example ``"vibration"`` for a vibration-analysis step). The
    planner uses it to decide which steps a sensor failure removes.
    """

    id: str
    name: str
    action: str
    target: str = ""
    params: dict = field(default_factory=dict)
    depends_on: tuple = ()


@dataclass(frozen=True)
class Plan:
    """An ordered list of steps for one goal."""

    goal: str
    steps: tuple = ()

    def step_ids(self) -> list:
        return [s.id for s in self.steps]

    def get(self, step_id: str) -> Step | None:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def __len__(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class Observation:
    """One thing the agent observed.

    ``kind`` is the machine-readable type (``"sensor_failure"``,
    ``"contradiction"``, ``"cost_spike"``, ...). ``source`` names
    where it came from (a sensor, a document, a tool). ``t`` is a
    logical clock value: observations are ordered by ``t``.
    """

    id: str
    kind: str
    source: str
    text: str
    t: int = 0
    # Free-form facts the planner rules read, e.g. {"sensor": "vibration"}.
    facts: dict = field(default_factory=dict)


@dataclass
class Change:
    """One element of a plan diff."""

    kind: str  # "added" | "removed" | "modified" | "moved"
    step_id: str
    old_step: Step | None = None
    new_step: Step | None = None

    def key(self) -> str:
        return f"{self.kind}:{self.step_id}"

    def describe(self) -> str:
        if self.kind == "added":
            return f"added step {self.step_id} ({self.new_step.name})"
        if self.kind == "removed":
            return f"removed step {self.step_id} ({self.old_step.name})"
        if self.kind == "modified":
            return (
                f"modified step {self.step_id} "
                f"({self.old_step.name} -> {self.new_step.name})"
            )
        return f"moved step {self.step_id} ({self.new_step.name})"


@dataclass
class Trace:
    """A Plan-and-Execute run: observations plus the plans made from them.

    ``plans`` maps a plan label (``"initial"``, ``"replan"``) to the
    plan and the observation ids that were visible when it was made.
    ``context`` holds internal (non-observation) state at replan time,
    e.g. a step budget that ran out — the thing a "no trigger" replan
    is caused by.
    """

    goal: str
    observations: list = field(default_factory=list)
    plans: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)

    def observation(self, obs_id: str) -> Observation | None:
        for obs in self.observations:
            if obs.id == obs_id:
                return obs
        return None

    def visible_at(self, plan_label: str) -> list:
        """Observations visible when ``plan_label`` was made."""
        entry = self.plans.get(plan_label)
        if entry is None:
            return []
        visible_ids = set(entry[1])
        return [o for o in self.observations if o.id in visible_ids]
