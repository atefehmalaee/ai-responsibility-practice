import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular data
from sklearn.datasets import make_classification  # Synthetic dataset
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score  # Evaluation metric
from sklearn.model_selection import train_test_split  # Train/test split


@dataclass
class ReviewDecision:
    case_id: int
    proba: float
    model_pred: int
    requires_review: bool
    escalation_level: str


def build_review_policy() -> dict:
    return {
        "auto_approve_threshold": 0.80,
        "auto_reject_threshold": 0.20,
        "escalation_rules": {
            "high_risk": "manual_review",
            "low_confidence": "manual_review",
        },
    }


def assign_review_decision(proba: float, policy: dict) -> ReviewDecision:
    if proba >= policy["auto_approve_threshold"]:
        return ReviewDecision(0, proba, 1, False, "auto_approve")  # Approve without human review
    if proba <= policy["auto_reject_threshold"]:
        return ReviewDecision(0, proba, 0, False, "auto_reject")  # Reject without human review
    return ReviewDecision(0, proba, int(proba >= 0.5), True, "manual_review")  # Route to reviewer


def simulate_review_queue(proba: np.ndarray, policy: dict) -> List[ReviewDecision]:
    decisions = []
    for idx, p in enumerate(proba, start=1):
        decision = assign_review_decision(float(p), policy)
        decision.case_id = idx  # Stable ID for audit trail
        decisions.append(decision)
    return decisions


def write_policy(path: Path, policy: dict) -> None:
    content = "\n".join(
        [
            "Oversight Policy",
            "================",
            "",
            "Thresholds",
            "----------",
            f"- Auto-approve: >= {policy['auto_approve_threshold']:.2f}",
            f"- Auto-reject: <= {policy['auto_reject_threshold']:.2f}",
            "",
            "Escalation Rules",
            "----------------",
            "- Low-confidence predictions go to manual review.",
            "- High-risk cases are always reviewed by humans.",
            "",
            "Rollback",
            "--------",
            "- If incident rate exceeds threshold, disable automated decisions.",
            "",
        ]
    )
    path.write_text(content)  # Human oversight policy artifact


def write_escalation_checklist(path: Path) -> None:
    content = "\n".join(
        [
            "Escalation & Rollback Checklist",
            "===============================",
            "",
            "- [ ] Identify issue and severity",
            "- [ ] Escalate to on-call reviewer",
            "- [ ] Pause automated approvals if needed",
            "- [ ] Notify stakeholders",
            "- [ ] Apply rollback and document outcome",
            "",
        ]
    )
    path.write_text(content)  # Operational checklist for incident response


def main() -> None:
    parser = argparse.ArgumentParser(description="Human oversight audit with review queue simulation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=args.seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000)  # Simple baseline for oversight simulation
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]  # Class-1 confidence scores
    y_pred = (proba >= 0.5).astype(int)  # Default decision threshold
    accuracy = float(accuracy_score(y_test, y_pred))

    policy = build_review_policy()  # Oversight rules and thresholds
    review_queue = simulate_review_queue(proba, policy)  # Assign review outcomes

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

    pd.DataFrame([d.__dict__ for d in review_queue]).to_csv(out_dir / "review_queue.csv", index=False)
    write_policy(out_dir / "oversight_policy.md", policy)
    write_escalation_checklist(out_dir / "escalation_checklist.md")

    print(f"Model accuracy (for context): {accuracy:.3f}")
    print(f"Wrote {out_dir / 'review_queue.csv'}")
    print(f"Wrote {out_dir / 'oversight_policy.md'}")
    print(f"Wrote {out_dir / 'escalation_checklist.md'}")


if __name__ == "__main__":
    main()
