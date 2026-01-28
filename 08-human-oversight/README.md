Human Oversight & Control
=========================

Goal
----
Design safe human review, escalation, and override flows.

What to build (professional, portfolio-ready)
---------------------------------------------
- Oversight policy with review thresholds
- Escalation and rollback checklist
- Human review queue simulation

Techniques
----------
- Human-in-the-loop review thresholds
- Escalation policies for uncertain outputs
- Override and rollback procedures

Exercises
---------
- Define when humans must review outputs.
- Simulate a bad output and document response steps.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the oversight audit:
   - `python oversight_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/review_queue.csv`
- `reports/oversight_policy.md`
- `reports/escalation_checklist.md`

Deliverables
------------
- Oversight workflow diagram
- Escalation and rollback checklist
