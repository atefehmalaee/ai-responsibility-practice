Model Card
==========

Overview
--------
Baseline logistic regression trained on synthetic data for transparency practice.

Intended Use
------------
Educational and interview portfolio demonstration of explainability techniques.

Training Data
-------------
- Samples: 2000
- Features: 8
- Source: `sklearn.datasets.make_classification`

Performance
-----------
- Accuracy: 0.767
- ROC AUC: 0.848

Key Features (Global Importance)
-------------------------------
- feature_4, feature_0, feature_6, feature_2, feature_3

Limitations
-----------
- Synthetic data only; results do not generalize to real-world domains.
- Linear model may underfit complex patterns.

Notes
-----
Global importance is based on absolute logistic regression coefficients.
