Privacy & Data Protection
=========================

Goal
----
Protect personal data and minimize exposure.

What to build (professional, portfolio-ready)
---------------------------------------------
- A PII scanner with redaction output
- A feature minimization comparison
- A concise privacy risk note

Techniques
----------
- Data minimization and purpose limitation
- Pseudonymization or anonymization
- Differential privacy (basic mechanisms)

Exercises
---------
- Identify PII and propose removal/redaction.
- Compare model results with and without sensitive fields.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the audit:
   - `python privacy_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/pii_scan.csv`
- `reports/feature_minimization.csv`
- `reports/privacy_risk_note.md`
- `reports/redacted_sample.csv`

Deliverables
------------
- Data handling note (PII map, retention plan)
- Privacy risk assessment summary
