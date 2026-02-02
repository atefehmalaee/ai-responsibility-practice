Safety Policy
=============

Scope
-----
This policy defines high-risk categories and expected guardrail behavior.

High-Risk Categories
--------------------
- self_harm
- violence
- hate
- illegal_activity
- privacy_leak

Guardrail Rules
---------------
- Refuse and redirect requests in high-risk categories.
- Block prompts containing explicit unsafe keywords.

Known Limitations
-----------------
- Keyword checks are not robust against paraphrasing.
- Policy requires human review for edge cases.
