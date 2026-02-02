Human Oversight & Control
=========================

Purpose
-------
Design and demonstrate human-in-the-loop governance with clear review thresholds,
escalation paths, and rollback procedures.

What this module delivers
-------------------------
- Review policy with auto-approve / auto-reject thresholds
- Simulated review queue with decision routing
- Escalation and rollback checklist for incidents

Key techniques
--------------
- Confidence-based routing and human review thresholds
- Escalation policies for low-confidence or high-risk cases
- Override and rollback procedures for safety control

Quick start (top-level)
-----------------------
1. Create environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r 08-human-oversight/requirements.txt`
3. Run from the project root:
   - `python 08-human-oversight/oversight_audit.py --seed 42 --out 08-human-oversight/reports`

Outputs
-------
- `08-human-oversight/reports/review_queue.csv`
- `08-human-oversight/reports/oversight_policy.md`
- `08-human-oversight/reports/escalation_checklist.md`

Deliverables
------------
- Oversight policy and audit trail for routing decisions
- Escalation and rollback checklist for governance readiness
