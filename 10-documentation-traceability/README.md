Documentation & Traceability
============================

Goal
----
Track data, models, and decisions across the lifecycle.

What to build (professional, portfolio-ready)
---------------------------------------------
- Dataset lineage record
- Model version log
- Experiment metadata and change log

Techniques
----------
- Versioning for data and models
- Dataset lineage and provenance notes
- Change logs and experiment tracking

Exercises
---------
- Create a simple lineage record for your dataset.
- Track two model versions with notes.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the documentation audit:
   - `python traceability_audit.py --out reports`

Outputs
-------
- `reports/data_lineage.csv`
- `reports/model_registry.csv`
- `reports/experiment_log.csv`
- `reports/change_log.md`

Deliverables
------------
- Lineage record
- Model version log
