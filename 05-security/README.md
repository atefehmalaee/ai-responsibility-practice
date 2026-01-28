Security & Robustness
=====================

Goal
----
Reduce vulnerabilities in data, model, and pipeline.

What to build (professional, portfolio-ready)
---------------------------------------------
- A lightweight threat model (STRIDE table)
- Input validation and fuzz test report
- Dependency/supply-chain checklist

Techniques
----------
- Threat modeling (STRIDE-style)
- Adversarial testing and input validation
- Dependency and supply chain review

Exercises
---------
- Build a threat model for the data pipeline.
- Test model with perturbed or malformed inputs.

Quick start
-----------
1. Install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run the security audit:
   - `python security_audit.py --seed 42 --out reports`

Outputs
-------
- `reports/threat_model.csv`
- `reports/input_fuzz_report.csv`
- `reports/security_notes.md`

Deliverables
------------
- Threat model diagram or table
- Security test notes and mitigations
