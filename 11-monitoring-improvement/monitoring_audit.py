import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reports
from sklearn.datasets import make_classification  # Synthetic dataset
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score  # Performance metric
from sklearn.model_selection import train_test_split  # Train/test split


@dataclass
class MonitoringMetric:
    metric: str
    value: float
    threshold: float
    status: str


@dataclass
class DriftResult:
    feature: str
    baseline_mean: float
    current_mean: float
    drift: float
    threshold: float
    flagged: bool


def generate_data(seed: int, drift: bool = False) -> tuple:
    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=seed,
    )
    if drift:
        X[:, 0] = X[:, 0] + 0.8  # Shift feature distribution
        X[:, 3] = X[:, 3] * 1.3  # Scale feature distribution
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(X)
    return float(accuracy_score(y, y_pred))


def compute_drift(baseline: np.ndarray, current: np.ndarray, threshold: float = 0.2) -> List[DriftResult]:
    results = []
    for idx in range(baseline.shape[1]):
        base_mean = float(np.mean(baseline[:, idx]))
        curr_mean = float(np.mean(current[:, idx]))
        drift = abs(curr_mean - base_mean)
        results.append(
            DriftResult(
                feature=f"feature_{idx}",
                baseline_mean=base_mean,
                current_mean=curr_mean,
                drift=drift,
                threshold=threshold,
                flagged=drift > threshold,
            )
        )
    return results


def build_monitoring_metrics(acc: float, drift_flags: int) -> List[MonitoringMetric]:
    return [
        MonitoringMetric("accuracy", acc, 0.75, "alert" if acc < 0.75 else "ok"),
        MonitoringMetric("drift_flags", float(drift_flags), 1.0, "alert" if drift_flags >= 1 else "ok"),
        MonitoringMetric("data_volume", 2000.0, 1500.0, "ok"),
    ]


def write_incident_response(path: Path) -> None:
    content = "\n".join(
        [
            "Incident Response",
            "=================",
            "",
            "Triggers",
            "--------",
            "- Accuracy below threshold",
            "- Drift flags above threshold",
            "",
            "Response Steps",
            "--------------",
            "- Page on-call and triage issue",
            "- Disable automation if risk is high",
            "- Collect evidence and decide retraining",
            "- Post-mortem and preventive actions",
            "",
        ]
    )
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring audit with drift simulation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    X_base, y_base = generate_data(args.seed, drift=False)
    X_curr, y_curr = generate_data(args.seed + 1, drift=True)

    model = train_model(X_base, y_base, args.seed)
    acc = evaluate(model, X_curr, y_curr)

    drift_results = compute_drift(X_base, X_curr)
    drift_flags = sum(1 for r in drift_results if r.flagged)

    metrics = build_monitoring_metrics(acc, drift_flags)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([m.__dict__ for m in metrics]).to_csv(out_dir / "monitoring_metrics.csv", index=False)
    pd.DataFrame([d.__dict__ for d in drift_results]).to_csv(out_dir / "drift_report.csv", index=False)
    write_incident_response(out_dir / "incident_response.md")

    print(f"Wrote {out_dir / 'monitoring_metrics.csv'}")
    print(f"Wrote {out_dir / 'drift_report.csv'}")
    print(f"Wrote {out_dir / 'incident_response.md'}")


if __name__ == "__main__":
    main()
