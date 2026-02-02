import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import Dict, List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reports
from sklearn.datasets import make_classification  # Synthetic dataset
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score, brier_score_loss  # Reliability metrics
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.preprocessing import StandardScaler  # Feature scaling


@dataclass
class ReliabilityMetrics:
    accuracy: float
    brier: float
    ece: float


def compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(proba[mask]))
        ece += (np.sum(mask) / len(proba)) * abs(bin_acc - bin_conf)  # Calibration gap
    return float(ece)


def error_slices(y_true: np.ndarray, y_pred: np.ndarray, slice_feature: np.ndarray) -> pd.DataFrame:
    rows = []
    for value in sorted(np.unique(slice_feature)):
        mask = slice_feature == value
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        err = 1.0 - acc
        rows.append({"slice": int(value), "n": int(mask.sum()), "accuracy": acc, "error_rate": err})
    return pd.DataFrame(rows)


def stress_tests(X: np.ndarray, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    tests = {}
    tests["noise"] = X + rng.normal(0, 0.3, size=X.shape)  # Add Gaussian noise
    tests["scale"] = X * rng.uniform(0.7, 1.4, size=(1, X.shape[1]))  # Feature scaling drift
    tests["outliers"] = X * (1 + (rng.rand(*X.shape) < 0.02) * 5.0)  # Rare extreme values
    return tests


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> ReliabilityMetrics:
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)  # Default decision threshold
    return ReliabilityMetrics(
        accuracy=float(accuracy_score(y, y_pred)),
        brier=float(brier_score_loss(y, proba)),
        ece=float(compute_ece(y, proba)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability audit with calibration and stress tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)  # Reproducible stress scenarios

    X, y = make_classification(
        n_samples=2500,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=args.seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()  # Normalize features for stable calibration
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)  # Simple baseline for reliability tests
    model.fit(X_train, y_train)

    baseline = evaluate(model, X_test, y_test)  # Baseline reliability metrics

    proba = model.predict_proba(X_test)[:, 1]  # Probabilities for slice analysis
    y_pred = (proba >= 0.5).astype(int)  # Binary predictions

    slice_feature = (X_test[:, 0] > 0).astype(int)  # Simple slice on one feature
    slice_report = error_slices(y_test, y_pred, slice_feature)

    stress_reports = []
    for name, X_stress in stress_tests(X_test, rng).items():
        metrics = evaluate(model, X_stress, y_test)  # Reliability under perturbation
        stress_reports.append({"scenario": name, **metrics.__dict__})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

    pd.DataFrame([baseline.__dict__]).to_csv(out_dir / "reliability_metrics.csv", index=False)
    slice_report.to_csv(out_dir / "error_slices.csv", index=False)
    pd.DataFrame(stress_reports).to_csv(out_dir / "stress_test.csv", index=False)

    print(f"Wrote {out_dir / 'reliability_metrics.csv'}")
    print(f"Wrote {out_dir / 'error_slices.csv'}")
    print(f"Wrote {out_dir / 'stress_test.csv'}")


if __name__ == "__main__":
    main()
