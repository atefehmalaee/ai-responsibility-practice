Documentation & Traceability
============================

Purpose
-------
Track data, experiments, and model versions using MLflow as the single source of truth.

What this module delivers
-------------------------
- Data lineage metadata logged as MLflow tags
- Experiment runs with parameters, metrics, and artifacts
- Registered model versions in the MLflow Model Registry
- Change log stored as an MLflow artifact

Key techniques
--------------
- MLflow experiment grouping with `mlflow.set_experiment()`
- Parameter/metric tracking with `mlflow.log_param` and `mlflow.log_metric`
- Artifact logging with `mlflow.log_artifact`
- Model registration with `mlflow.register_model`

Quick start (top-level)
-----------------------
1. Create environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r 10-documentation-traceability/requirements.txt`
3. Run from the project root:
   - `python 10-documentation-traceability/traceability_audit.py --experiment-name traceability_audit`

View results in MLflow UI
-------------------------
- Run `mlflow ui` from the project root, then open `http://127.0.0.1:5000`

Outputs
-------
- MLflow runs under the `traceability_audit` experiment
- Tags: dataset lineage metadata (source, version, PII flag)
- Artifacts: change log and JSON summaries
- Registered model versions in the MLflow Model Registry

Deliverables
------------
- Traceability evidence in MLflow (runs, tags, artifacts, model registry)
