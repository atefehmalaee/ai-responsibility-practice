Fairness
========

Purpose
-------
Measure and reduce disparities across sensitive groups while documenting tradeoffs.

What this module covers
-----------------------
- Auditing: group, intersectional, and counterfactual checks
- Mitigations: pre-, in-, and post-processing techniques
- Reporting: reproducible CSV evidence and summaries

Techniques
----------
- Metrics: demographic parity, equal opportunity, average odds
- Audits: group and intersectional slices, counterfactual flip rate
- Mitigations: reweighing, fairness-regularized training, equalized-odds thresholds
- Advanced: Fairlearn (EG, ThresholdOptimizer) and AIF360 (ROC)

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
2. Run with Census Income (Adult) dataset (default):
   - `python fairness_audit.py --dataset adult --adult-path data/adult.csv --sensitive-attr sex --out reports`
   - To download automatically:
     - `python fairness_audit.py --download-adult --out reports`
3. Synthetic data option removed; this module now uses Census Income (Adult) only.

Outputs
-------
- `reports/fairness_report.csv` (baseline + mitigations)
- `reports/fairness_intersectional.csv` (intersectional slices)
- `reports/counterfactual_summary.csv` (flip-rate check)
- `reports/fairlearn_metrics.csv` and `reports/fairlearn_mitigations.csv` (optional)
- `reports/aif360_metrics.csv`, `reports/aif360_mitigations.csv` (optional)
- `reports/aif360_adversarial.csv` (optional, requires TensorFlow)
- Console summary of fairness gaps by technique

How to present (interview-ready)
--------------------------------
- Show baseline gaps, then improvement per mitigation.
- Highlight accuracy tradeoffs vs fairness gains.
- Explain which fairness constraint best fits the use case.

Deliverables
------------
- Fairness audit tables and plots
- Short note on chosen mitigation and tradeoffs

