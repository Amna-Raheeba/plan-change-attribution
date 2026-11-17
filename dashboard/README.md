# Dashboard

Run: `streamlit run dashboard/app.py` (install with
`pip install -e ".[dashboard]"`).

- **Scenario explorer** — pick one of the 30 scenarios, see the
  initial and replanned plans side by side, the verdict, the minimal
  triggering set, each change with its top trigger, and the evidence
  score of every candidate observation.
- **Evaluation** — run the full 30-scenario suite and see the headline
  metrics and per-change rows.
- **Trigger stability** — pick a scenario and see each attributed
  trigger's survival rate over 20 perturbed replays (confident /
  mixed / fragile), the minimal-set stability, and the
  duplicate-probe survival.

Helpers in `dashboard/helpers.py` are pure functions (no Streamlit
import) and are covered by tests, including an AppTest run of the
whole app with zero exceptions.
