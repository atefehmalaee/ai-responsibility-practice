Monitoring & Continuous Improvement
====================================

Goal
----
Detect drift, failures, and improve over time.

What to build (professional, portfolio-ready)
---------------------------------------------
- Monitoring metrics and alert thresholds
- Drift simulation report
- Incident response playbook

Techniques
----------
- Monitoring for data drift and performance
- Alert thresholds and incident response
- Feedback loops for retraining

Exercises
---------
- Define 3 monitoring metrics and alert thresholds.
- Simulate drift and document response steps.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the monitoring audit:
   - `python monitoring_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/monitoring_metrics.csv`
- `reports/drift_report.csv`
- `reports/incident_response.md`

Deliverables
------------
- Monitoring plan
- Incident response note
