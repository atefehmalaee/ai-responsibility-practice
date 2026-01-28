Compliance & Legal Alignment
============================

Goal
----
Align the system with applicable laws and policies.

What to build (professional, portfolio-ready)
---------------------------------------------
- Compliance matrix mapping requirements to controls
- DPIA-style risk checklist
- Evidence log for approvals

Techniques
----------
- Map requirements to system controls
- DPIA-style privacy impact checklist
- Vendor and licensing review

Exercises
---------
- Build a compliance matrix for your use case.
- Identify any high-risk constraints and mitigations.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the compliance audit:
   - `python compliance_audit.py --out reports`

Outputs
-------
- `reports/compliance_matrix.csv`
- `reports/dpia_checklist.csv`
- `reports/compliance_notes.md`

Deliverables
------------
- Compliance matrix
- Risk summary with required controls
