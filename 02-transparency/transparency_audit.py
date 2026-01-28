import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reports
from sklearn.datasets import make_classification  # Synthetic dataset
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score, roc_auc_score  # Summary metrics
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.preprocessing import StandardScaler  # Feature standardization


@dataclass
class ModelSummary:
    accuracy: float  # Test accuracy
    roc_auc: float  # Test ROC AUC
    n_samples: int  # Dataset size
    n_features: int  # Feature count


def build_feature_names(n_features: int) -> list:
    return [f"feature_{i}" for i in range(n_features)]  # Stable feature labels


def global_importance(coef: np.ndarray, feature_names: list) -> pd.DataFrame:
    importance = np.abs(coef).ravel()  # Use absolute coefficients
    order = np.argsort(importance)[::-1]  # Sort descending
    return pd.DataFrame(  # Global feature importance table
        {
            "feature": [feature_names[i] for i in order],  # Ranked features
            "importance": importance[order],  # Importance score
        }
    )


def local_explanations(
    X_std: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    feature_names: list,
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_ids: list,
) -> pd.DataFrame:
    rows = []  # Collect per-sample explanations
    weights = coef.ravel()  # Linear model weights
    for row_idx in sample_ids:  # Explain selected samples
        contributions = X_std[row_idx] * weights  # Per-feature contribution
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]  # Top 3 by magnitude
        top_features = ", ".join(  # Compact explanation string
            f"{feature_names[i]}={contributions[i]:.3f}" for i in top_idx
        )
        rows.append(
            {
                "sample_id": row_idx,  # Row identifier
                "true_label": int(y_true[row_idx]),  # Ground truth
                "pred_proba": float(proba[row_idx]),  # Predicted probability
                "intercept": float(intercept),  # Model intercept
                "top_contributions": top_features,  # Top feature impacts
            }
        )
    return pd.DataFrame(rows)  # Local explanation table


def write_model_card(
    path: Path,
    summary: ModelSummary,
    feature_names: list,
    notes: str,
) -> None:
    content = "\n".join(  # Simple markdown model card
        [
            "Model Card",
            "==========",
            "",
            "Overview",
            "--------",
            "Baseline logistic regression trained on synthetic data for transparency practice.",
            "",
            "Intended Use",
            "------------",
            "Educational and interview portfolio demonstration of explainability techniques.",
            "",
            "Training Data",
            "-------------",
            f"- Samples: {summary.n_samples}",
            f"- Features: {summary.n_features}",
            "- Source: `sklearn.datasets.make_classification`",
            "",
            "Performance",
            "-----------",
            f"- Accuracy: {summary.accuracy:.3f}",
            f"- ROC AUC: {summary.roc_auc:.3f}",
            "",
            "Key Features (Global Importance)",
            "-------------------------------",
            "- " + ", ".join(feature_names[:5]),
            "",
            "Limitations",
            "-----------",
            "- Synthetic data only; results do not generalize to real-world domains.",
            "- Linear model may underfit complex patterns.",
            "",
            "Notes",
            "-----",
            notes,
            "",
        ]
    )
    path.write_text(content)  # Write model card to disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Transparency audit with global/local explanations.")  # CLI
    parser.add_argument("--seed", type=int, default=42)  # RNG seed
    parser.add_argument("--out", type=str, default="reports")  # Output directory
    args = parser.parse_args()  # Parse args

    rng = np.random.RandomState(args.seed)  # Reproducible sampling

    X, y = make_classification(  # Synthetic dataset
        n_samples=2000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=args.seed,
    )
    feature_names = build_feature_names(X.shape[1])  # Stable feature labels

    X_train, X_test, y_train, y_test = train_test_split(  # Stratified split
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()  # Normalize features
    X_train_std = scaler.fit_transform(X_train)  # Fit on train
    X_test_std = scaler.transform(X_test)  # Apply to test

    model = LogisticRegression(max_iter=1000)  # Baseline model
    model.fit(X_train_std, y_train)  # Train model

    proba = model.predict_proba(X_test_std)[:, 1]  # Predicted probabilities
    y_pred = (proba >= 0.5).astype(int)  # Thresholded predictions

    summary = ModelSummary(  # Summary metrics
        accuracy=float(accuracy_score(y_test, y_pred)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )

    out_dir = Path(args.out)  # Output directory
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    global_report = global_importance(model.coef_, feature_names)  # Global explanation
    global_report.to_csv(out_dir / "global_feature_importance.csv", index=False)  # Save

    sample_ids = rng.choice(len(y_test), size=5, replace=False).tolist()  # Sample rows
    local_report = local_explanations(  # Local explanations
        X_test_std,
        model.coef_,
        float(model.intercept_[0]),
        feature_names,
        y_test,
        proba,
        sample_ids,
    )
    local_report.to_csv(out_dir / "local_explanations.csv", index=False)  # Save

    write_model_card(  # Generate model card
        out_dir / "model_card.md",
        summary,
        global_report["feature"].tolist(),
        notes="Global importance is based on absolute logistic regression coefficients.",
    )

    print(f"Wrote {out_dir / 'global_feature_importance.csv'}")  # Report outputs
    print(f"Wrote {out_dir / 'local_explanations.csv'}")  # Report outputs
    print(f"Wrote {out_dir / 'model_card.md'}")  # Report outputs


if __name__ == "__main__":
    main()  # CLI entrypoint
