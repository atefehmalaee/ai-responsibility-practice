Privacy Risk Note
=================

PII Findings
------------
- Name: 1200 rows
- Email: 1200 rows
- Phone: 1200 rows

Feature Minimization
--------------------
- full_features: accuracy=0.997, roc_auc=0.997, features=6
- minimized_features: accuracy=0.997, roc_auc=0.994, features=3

Risk Controls
-------------
- Remove or redact direct identifiers before modeling.
- Avoid derived features from PII unless strictly necessary.
- Document data retention and access controls.
