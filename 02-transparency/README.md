Transparency & Explainability
=============================

Goal
----
Make system behavior understandable to stakeholders.

What to build (professional, portfolio-ready)
---------------------------------------------
- A reproducible transparency audit script
- Global + local explanation reports
- A concise model card

Techniques
----------
- Model cards and data sheets
- Global and local explanations (e.g., SHAP, LIME)
- Decision logs for key outputs

Exercises
---------
- Create a model card for your baseline model.
- Generate local explanations for 5 examples.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the audit:
   - `python transparency_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/global_feature_importance.csv`
- `reports/local_explanations.csv`
- `reports/model_card.md`

Deliverables
------------
- Model card draft
- Explanation examples with short interpretation
