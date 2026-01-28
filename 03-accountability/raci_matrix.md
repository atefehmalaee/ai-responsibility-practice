RACI Matrix (Template)
======================

Legend: R = Responsible, A = Accountable, C = Consulted, I = Informed <!-- Role definitions -->

Roles (edit for your org)
-------------------------
- Product Owner <!-- Business accountability -->
- Data Scientist <!-- Model design/validation -->
- ML Engineer <!-- Productionization -->
- Legal/Compliance <!-- Regulatory review -->
- Security <!-- Risk controls -->
- QA/Test <!-- Verification -->

Lifecycle Steps
---------------
| Step | Product Owner | Data Scientist | ML Engineer | Legal/Compliance | Security | QA/Test |
| --- | --- | --- | --- | --- | --- | --- |
| Data sourcing & consent | A | R | I | C | I | I | <!-- Data provenance -->
| Data labeling | A | R | I | C | I | I | <!-- Label quality -->
| Feature engineering | C | R | C | I | I | I | <!-- Feature design -->
| Model training | I | R | C | I | I | I | <!-- Training ownership -->
| Model evaluation | C | R | C | I | I | R | <!-- Validation -->
| Bias/fairness review | A | R | C | C | I | I | <!-- Fairness gate -->
| Security review | I | C | R | I | A | I | <!-- Security gate -->
| Privacy review | I | C | I | A | I | I | <!-- Privacy gate -->
| Release approval | A | C | C | C | C | C | <!-- Go/no-go -->
| Post-launch monitoring | A | R | R | I | I | C | <!-- Operational oversight -->
