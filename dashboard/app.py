"""Streamlit dashboard for plan-change attribution.

Run: ``streamlit run dashboard/app.py``
Three tabs: pick a scenario and see the trace, the plan diff, and the
attribution; view the full 30-scenario evaluation; or check whether
each attributed trigger survives perturbed replays (stability).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from dashboard.helpers import (  # noqa: E402
    attribution_for, diff_rows, metric_cards, plan_table, scenario_by_name,
    scenario_names, stability_for, stability_rows, trigger_rows,
)
from plan_change_attribution.evaluation import evaluate_suite  # noqa: E402


st.set_page_config(page_title="Plan-Change Attribution", layout="wide")
st.title("Plan-Change Attribution")
st.write(
    "Why did the agent's plan change? Pick a scenario: the dashboard "
    "replays the planner with each new observation removed in turn and "
    "shows which observation caused each dropped or added step.")

tab_scenario, tab_eval, tab_stab = st.tabs(
    ["Scenario explorer", "Evaluation", "Trigger stability"])

with tab_scenario:
    name = st.selectbox("Scenario", scenario_names())
    scenario = scenario_by_name(name)
    report = attribution_for(scenario)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Initial plan")
        st.table(plan_table(report.old_plan))
    with col2:
        st.subheader("Replanned plan")
        st.table(plan_table(report.new_plan))
    st.subheader(f"Verdict: {report.verdict}")
    if report.minimal_triggering_set:
        st.write("Minimal triggering set: "
                 + ", ".join(report.minimal_triggering_set))
    st.subheader("Changes and their triggers")
    st.table(diff_rows(report))
    st.subheader("Candidate evidence scores")
    st.table(trigger_rows(report))

with tab_eval:
    if st.button("Run full evaluation (30 scenarios)"):
        out = evaluate_suite()
        for label, value in metric_cards(out["metrics"]):
            st.metric(label, value)
        st.table(out["rows"])
    else:
        st.write("Press the button to run the attribution over all "
                 "30 scenarios with known ground truth.")

with tab_stab:
    name_stab = st.selectbox("Scenario (stability)", scenario_names())
    scenario_stab = scenario_by_name(name_stab)
    stab = stability_for(scenario_stab)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Minimal-set stability",
                  f"{stab.minimal_set_stability:.2f}")
    with col2:
        dup = stab.duplicate_survival
        st.metric("Duplicate-probe survival",
                  f"{dup:.2f}" if dup is not None else "—")
    st.write(
        "Each attributed trigger is re-scored over 20 perturbed "
        "replays (shuffled order, added/removed distractors, jittered "
        "timestamps). Stability is the share of replays in which the "
        "trigger is still the top cause: >= 0.80 confident, 0.50-0.80 "
        "mixed, below 0.50 fragile. The duplicate probe (5 replays) "
        "copies one candidate observation: single-observation ablation "
        "cannot separate a cause from its identical copy.")
    st.table(stability_rows(stab))
