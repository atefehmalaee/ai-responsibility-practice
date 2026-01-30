User Impact & Stakeholder Engagement
====================================

Purpose
-------
Operationalize human-centered governance by combining decision traceability, impact assessment,
and continuous user feedback into audit-ready artifacts.

What this module delivers
-------------------------
- Human-in-the-loop enforcement for low-confidence or high-risk outputs
- Decision logging with model metadata, confidence, and reviewer actions
- Living risk register with incident-driven updates
- User survey pipeline that triggers retraining, policy review, or UX review
- Accessibility checklist plus manual sign-off record

Key techniques
--------------
- Human review routing (risk/threshold-based)
- Structured audit logs (CSV + JSONL)
- Impact assessment with ownership and review cadence
- Incident linkage to risk register updates
- Accessibility compliance checks and approval

Quick start
-----------
1. Create environment (standard library only):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Run the audit:
   - `python user_impact_audit.py --out reports`
3. Optional: simulate a low-confidence decision with review:
   - `python user_impact_audit.py --confidence 0.55 --risk-level high --reviewer-id reviewer-1 --review-decision approve`
4. Optional: append a survey response:
   - `python user_impact_audit.py --survey-response "Confusing output" --survey-rating 2 --survey-user-id user-123`

Outputs
-------
- `reports/impact_assessment.csv` (summary)
- `reports/logs/model_metadata.json`
- `reports/logs/decision_log.csv`
- `reports/logs/decision_log.jsonl`
- `reports/risks/risk_register.csv`
- `reports/risks/incidents.csv` (only when incidents are appended)
- `reports/surveys/survey_responses.csv` (only when surveys are provided)
- `reports/user_survey.md`
- `reports/accessibility_checklist.md`
- `reports/reviews/accessibility_review.csv` (only when review is recorded)

Portfolio notes
---------------
- Demonstrates governance traceability and decision accountability.
- Shows how feedback loops drive risk controls and model review actions.
