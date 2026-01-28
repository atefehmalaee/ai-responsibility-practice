Safety & Harm Prevention
========================

Goal
----
Prevent harmful outputs and unsafe behavior.

What to build (professional, portfolio-ready)
---------------------------------------------
- Safety policy with risk taxonomy
- Red-team test set and results
- Guardrail evaluation report

Techniques
----------
- Safety policy and red-team prompts
- Output filtering and guardrails
- Scenario-based risk testing

Exercises
---------
- Create a harmful output taxonomy for your use case.
- Run a small red-team test set and report results.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the safety audit:
   - `python safety_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/red_team_cases.csv`
- `reports/guardrail_results.csv`
- `reports/safety_policy.md`

Deliverables
------------
- Safety policy draft
- Red-team results summary
