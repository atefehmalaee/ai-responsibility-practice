Oversight Policy
================

Thresholds
----------
- Auto-approve: >= 0.80
- Auto-reject: <= 0.20

Escalation Rules
----------------
- Low-confidence predictions go to manual review.
- High-risk cases are always reviewed by humans.

Rollback
--------
- If incident rate exceeds threshold, disable automated decisions.
