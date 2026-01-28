Fairness
========

Goal
----
Identify and reduce disparate outcomes across groups.

What to build (professional, portfolio-ready)
---------------------------------------------
- A reproducible fairness audit script
- Baseline vs mitigated metrics comparison
- A short technical note on tradeoffs

Techniques
----------
- Group fairness metrics (demographic parity, equalized odds)
- Bias audits across sensitive attributes
- Mitigation (reweighing, thresholding, post-processing)

Exercises
---------
- Measure fairness metrics before and after mitigation.
- Compare tradeoffs between accuracy and fairness.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the audit:
   - `python fairness_audit.py --seed 42 --out reports/fairness_report.csv`

Outputs
-------
- `reports/fairness_report.csv` with per-group metrics
- Console summary of fairness gaps

Latest run (seed 42)
--------------------
Baseline selection rates: group 0 = 0.376, group 1 = 0.404 (gap 0.028)
Mitigated selection rates: group 0 = 0.389, group 1 = 0.390 (gap 0.002)

Accuracy (by group):
- Baseline: group 0 = 0.795, group 1 = 0.808
- Mitigated: group 0 = 0.798, group 1 = 0.799

Deliverables
------------
- Fairness audit table with metrics by group
- Short note on chosen mitigation and tradeoffs

How to show skill in Git
------------------------
- Commit `fairness_audit.py`, `README.md`, and `reports/fairness_report.csv`.
- Add a brief summary in this README: what metrics improved and what accuracy tradeoff you accepted.
