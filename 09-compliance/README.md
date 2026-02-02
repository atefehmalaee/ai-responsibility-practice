Compliance & Legal Alignment
============================

Purpose
-------
Demonstrate compliance evidence using MLflow metadata attached to model runs.

What this module delivers
-------------------------
- Compliance requirements logged as MLflow tags (`compliance.*`)
- DPIA checks logged as MLflow tags (`dpia.*`)
- Evidence files captured as MLflow artifacts

Key techniques
--------------
- Requirement-to-control mapping via MLflow tags
- DPIA risk documentation with explicit mitigation tags
- Evidence capture using MLflow artifacts

Quick start (top-level)
-----------------------
1. Create environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r 09-compliance/requirements.txt`
3. Run from the project root:
   - `python 09-compliance/compliance_audit.py --evidence-root .`

View results in MLflow UI
-------------------------
- Run `mlflow ui` from the project root, then open `http://127.0.0.1:5000`

Outputs
-------
- MLflow runs under the `responsible-ai-compliance` experiment
- Tags: `compliance.*` and `dpia.*` entries for each requirement/check
- Artifacts: evidence files when paths exist

Deliverables
------------
- Audit-ready compliance metadata in MLflow
