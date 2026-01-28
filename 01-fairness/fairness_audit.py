import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reporting
from sklearn.datasets import make_classification  # Synthetic dataset
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.model_selection import train_test_split  # Train/test split


@dataclass
class GroupMetrics:
    n: int  # Group sample count
    accuracy: float  # Accuracy within group
    selection_rate: float  # Predicted positive rate
    tpr: float  # True positive rate
    fpr: float  # False positive rate


def _safe_rate(num: int, denom: int) -> float:
    if denom == 0:  # Avoid division by zero
        return float("nan")  # Use NaN for undefined rates
    return num / denom  # Standard rate calculation


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict:
    metrics = {}  # Accumulator for per-group metrics
    for value in sorted(np.unique(group)):  # Iterate groups deterministically
        mask = group == value  # Group membership mask
        yt = y_true[mask]  # True labels for group
        yp = y_pred[mask]  # Predicted labels for group

        tp = int(np.sum((yt == 1) & (yp == 1)))  # True positives
        tn = int(np.sum((yt == 0) & (yp == 0)))  # True negatives
        fp = int(np.sum((yt == 0) & (yp == 1)))  # False positives
        fn = int(np.sum((yt == 1) & (yp == 0)))  # False negatives

        metrics[value] = GroupMetrics(  # Store computed metrics
            n=int(mask.sum()),  # Group size
            accuracy=_safe_rate(tp + tn, tp + tn + fp + fn),  # Accuracy
            selection_rate=_safe_rate(tp + fp, tp + tn + fp + fn),  # Positive rate
            tpr=_safe_rate(tp, tp + fn),  # True positive rate
            fpr=_safe_rate(fp, fp + tn),  # False positive rate
        )
    return metrics  # Return per-group metrics map


def fairness_gaps(group_metrics: dict) -> dict:
    values = sorted(group_metrics.keys())  # Group labels
    if len(values) < 2:  # Need at least two groups to compare
        return {"dp_diff": float("nan"), "eo_diff": float("nan"), "avg_odds_diff": float("nan")}

    g0 = group_metrics[values[0]]  # First group
    g1 = group_metrics[values[1]]  # Second group

    dp_diff = abs(g0.selection_rate - g1.selection_rate)  # Demographic parity diff
    eo_diff = abs(g0.tpr - g1.tpr)  # Equal opportunity diff
    avg_odds_diff = 0.5 * (abs(g0.tpr - g1.tpr) + abs(g0.fpr - g1.fpr))  # Avg odds diff
    return {"dp_diff": dp_diff, "eo_diff": eo_diff, "avg_odds_diff": avg_odds_diff}  # Gap summary


def equalize_selection_rate_thresholds(proba: np.ndarray, group: np.ndarray, target_rate: float) -> dict:
    thresholds = {}  # Group-specific thresholds
    for value in sorted(np.unique(group)):  # Iterate groups deterministically
        mask = group == value  # Group membership
        if mask.sum() == 0:  # Handle empty group safely
            thresholds[value] = 0.5  # Default threshold
            continue
        thresholds[value] = float(np.quantile(proba[mask], 1 - target_rate))  # Match rate
    return thresholds  # Thresholds per group


def apply_group_thresholds(proba: np.ndarray, group: np.ndarray, thresholds: dict) -> np.ndarray:
    preds = np.zeros_like(proba, dtype=int)  # Output predictions
    for value, thresh in thresholds.items():  # Apply threshold per group
        mask = group == value  # Group membership
        preds[mask] = (proba[mask] >= thresh).astype(int)  # Thresholded predictions
    return preds  # Final predictions


def build_report(stage: str, metrics: dict) -> pd.DataFrame:
    rows = []  # Collect report rows
    for group_value, gm in metrics.items():  # Iterate per-group metrics
        rows.append(
            {
                "stage": stage,  # Baseline or mitigated
                "group": int(group_value),  # Group label
                "n": gm.n,  # Group sample count
                "accuracy": gm.accuracy,  # Group accuracy
                "selection_rate": gm.selection_rate,  # Group positive rate
                "tpr": gm.tpr,  # Group true positive rate
                "fpr": gm.fpr,  # Group false positive rate
            }
        )
    return pd.DataFrame(rows)  # Report table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness audit with baseline and mitigation.")  # CLI
    parser.add_argument("--seed", type=int, default=42)  # RNG seed
    parser.add_argument("--out", type=str, default="reports/fairness_report.csv")  # Report path
    args = parser.parse_args()  # Parse CLI args

    rng = np.random.RandomState(args.seed)  # Reproducible randomness

    X, y = make_classification(  # Synthetic binary classification data
        n_samples=2500,  # Sample count
        n_features=8,  # Total features
        n_informative=5,  # Predictive features
        n_redundant=1,  # Redundant features
        weights=[0.6, 0.4],  # Class balance
        class_sep=1.0,  # Class separation
        random_state=args.seed,  # Reproducible generation
    )

    sensitive = (X[:, 0] + rng.normal(0, 0.5, size=X.shape[0]) > 0).astype(int)  # Simulated group

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(  # Split data
        X, y, sensitive, test_size=0.3, random_state=args.seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000)  # Baseline model
    model.fit(X_train, y_train)  # Train model

    proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities
    y_pred = (proba >= 0.5).astype(int)  # Default 0.5 threshold predictions

    baseline_metrics = compute_group_metrics(y_test, y_pred, s_test)  # Baseline metrics
    baseline_gaps = fairness_gaps(baseline_metrics)  # Baseline fairness gaps

    target_rate = float(np.mean(y_pred))  # Overall positive rate
    thresholds = equalize_selection_rate_thresholds(proba, s_test, target_rate)  # Per-group thresholds
    y_pred_mitigated = apply_group_thresholds(proba, s_test, thresholds)  # Mitigated predictions

    mitigated_metrics = compute_group_metrics(y_test, y_pred_mitigated, s_test)  # Mitigated metrics
    mitigated_gaps = fairness_gaps(mitigated_metrics)  # Mitigated gaps

    report = pd.concat(  # Combine baseline and mitigated reports
        [
            build_report("baseline", baseline_metrics),
            build_report("mitigated", mitigated_metrics),
        ],
        ignore_index=True,  # Clean index
    )
    report.to_csv(args.out, index=False)  # Save report

    print("Baseline gaps:", baseline_gaps)  # Report baseline gaps
    print("Mitigated gaps:", mitigated_gaps)  # Report mitigated gaps
    print(f"Saved report to {args.out}")  # Report output location


if __name__ == "__main__":
    main()  # Run CLI entrypoint
