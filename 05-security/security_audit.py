import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reports
from sklearn.datasets import make_classification  # Synthetic data
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score  # Evaluation metric
from sklearn.model_selection import train_test_split  # Train/test split


@dataclass
class ThreatItem:
    asset: str
    threat: str
    impact: str
    mitigation: str


@dataclass
class FuzzResult:
    scenario: str
    accuracy: float
    failure_rate: float


def build_threat_model() -> List[ThreatItem]:
    # Purpose: Define a lightweight STRIDE-style threat model for key assets.
    return [
        ThreatItem(
            asset="Training data",
            threat="Data poisoning (tampered labels or samples)",
            impact="Model integrity compromised",
            mitigation="Data access controls, checksums, sampling audits",
        ),
        ThreatItem(
            asset="Feature pipeline",
            threat="Input validation bypass",
            impact="Unexpected model behavior",
            mitigation="Schema validation, type checks, reject malformed rows",
        ),
        ThreatItem(
            asset="Model artifact",
            threat="Model theft or tampering",
            impact="Intellectual property loss, unsafe outputs",
            mitigation="Artifact signing, access control, integrity checks",
        ),
        ThreatItem(
            asset="Inference API",
            threat="Adversarial inputs / prompt injection",
            impact="Unsafe or incorrect decisions",
            mitigation="Rate limits, input sanitization, monitoring",
        ),
        ThreatItem(
            asset="Dependencies",
            threat="Malicious package in supply chain",
            impact="Runtime compromise",
            mitigation="Pinned versions, SCA scans, restricted registries",
        ),
    ]


def generate_data(seed: int) -> tuple:
    # Purpose: Generate a synthetic dataset for fuzz testing scenarios.
    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=seed,
    )
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    # Purpose: Train a baseline model to evaluate robustness under attacks.
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    # Purpose: Compute a single accuracy score for a given input scenario.
    y_pred = model.predict(X)
    return float(accuracy_score(y, y_pred))


def fuzz_inputs(rng: np.random.RandomState, X: np.ndarray) -> dict:
    # Purpose: Create adversarial-style perturbations of inputs.
    n_samples, n_features = X.shape
    results = {}

    # Scenario 1: additive noise
    noise = rng.normal(0, 0.5, size=X.shape)
    results["noise"] = X + noise

    # Scenario 2: feature scaling attack
    scale = rng.uniform(0.5, 2.0, size=(1, n_features))
    results["scale"] = X * scale

    # Scenario 3: missing values injected
    X_missing = X.copy()
    mask = rng.rand(*X_missing.shape) < 0.1
    X_missing[mask] = np.nan
    results["missing"] = X_missing

    # Scenario 4: extreme outliers
    X_outliers = X.copy()
    idx = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    X_outliers[idx] = X_outliers[idx] * 10.0
    results["outliers"] = X_outliers

    return results


def sanitize_inputs(X: np.ndarray) -> np.ndarray:
    # Purpose: Apply basic input sanitization before inference.
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_clean = np.clip(X_clean, -10.0, 10.0)
    return X_clean


def run_fuzz_tests(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> List[FuzzResult]:
    # Purpose: Measure performance degradation across fuzz scenarios.
    rng = np.random.RandomState(seed)
    scenarios = fuzz_inputs(rng, X_test)

    results = []
    baseline_acc = evaluate(model, X_test, y_test)
    for name, X_scenario in scenarios.items():
        X_clean = sanitize_inputs(X_scenario)
        acc = evaluate(model, X_clean, y_test)
        failure_rate = max(0.0, baseline_acc - acc)
        results.append(FuzzResult(name, acc, failure_rate))
    return results


def write_security_notes(path: Path, fuzz_results: List[FuzzResult]) -> None:
    # Purpose: Persist a concise security summary and mitigation notes.
    lines = [
        "Security Notes",
        "==============",
        "",
        "Summary",
        "-------",
        "This module demonstrates lightweight threat modeling and input fuzzing.",
        "",
        "Fuzz Test Outcomes",
        "------------------",
    ]
    for r in fuzz_results:
        lines.append(f"- {r.scenario}: accuracy={r.accuracy:.3f}, drop={r.failure_rate:.3f}")
    lines.extend(
        [
            "",
            "Mitigations",
            "-----------",
            "- Validate input schemas and types before inference.",
            "- Apply sanity checks and clipping for numeric ranges.",
            "- Monitor error rates for anomalies and abuse patterns.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    # Purpose: Orchestrate threat modeling, fuzz testing, and report outputs.
    parser = argparse.ArgumentParser(description="Security audit with threat model and fuzz testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    X, y = generate_data(args.seed)
    model = train_model(X, y, args.seed)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    fuzz_results = run_fuzz_tests(model, X_test, y_test, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    threat_table = pd.DataFrame([item.__dict__ for item in build_threat_model()])
    threat_table.to_csv(out_dir / "threat_model.csv", index=False)

    fuzz_table = pd.DataFrame([r.__dict__ for r in fuzz_results])
    fuzz_table.to_csv(out_dir / "input_fuzz_report.csv", index=False)

    write_security_notes(out_dir / "security_notes.md", fuzz_results)

    print(f"Wrote {out_dir / 'threat_model.csv'}")
    print(f"Wrote {out_dir / 'input_fuzz_report.csv'}")
    print(f"Wrote {out_dir / 'security_notes.md'}")


if __name__ == "__main__":
    main()
