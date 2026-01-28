Reliability & Quality
=====================

Goal
----
Ensure stable and predictable performance.

What to build (professional, portfolio-ready)
---------------------------------------------
- Reliability metrics table (accuracy + calibration)
- Slice-based error analysis report
- Stress test summary

Techniques
----------
- Calibration and confidence analysis
- Stress tests and edge-case evaluation
- Error analysis by slice

Exercises
---------
- Measure calibration or confidence quality.
- Build an error analysis report by segment.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the audit:
   - `python reliability_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/reliability_metrics.csv`
- `reports/error_slices.csv`
- `reports/stress_test.csv`

Deliverables
------------
- Reliability metrics dashboard (simple table)
- Error analysis notes
