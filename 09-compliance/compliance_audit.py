import argparse  # CLI parsing
# Compliance evidence is attached to model runs as MLflow tags and artifacts for auditability.
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import mlflow  # MLflow tracking APIs


@dataclass
class ComplianceRequirement:
    requirement: str
    control: str
    evidence: str
    status: str


@dataclass
class DpiaCheck:
    category: str
    question: str
    risk_level: str
    mitigation: str


def build_compliance_matrix() -> List[ComplianceRequirement]:
    return [
        ComplianceRequirement(
            requirement="Data minimization",
            control="Remove PII columns before training",
            evidence="privacy_audit.py output + data retention policy",
            status="In place",
        ),
        ComplianceRequirement(
            requirement="Purpose limitation",
            control="Document training purpose in model card",
            evidence="reports/model_card.md",
            status="In place",
        ),
        ComplianceRequirement(
            requirement="Security review",
            control="Threat model and fuzz tests",
            evidence="reports/threat_model.csv + reports/input_fuzz_report.csv",
            status="In place",
        ),
        ComplianceRequirement(
            requirement="Human oversight",
            control="Manual review thresholds",
            evidence="reports/oversight_policy.md",
            status="In place",
        ),
        ComplianceRequirement(
            requirement="Auditability",
            control="Decision log and approvals",
            evidence="03-accountability/audit_log.csv",
            status="In place",
        ),
    ]


def build_dpia_checklist() -> List[DpiaCheck]:
    return [
        DpiaCheck(
            category="Data",
            question="Does the system process personal data or identifiers?",
            risk_level="Medium",
            mitigation="PII scanning, redaction, minimization",
        ),
        DpiaCheck(
            category="Purpose",
            question="Is data reuse clearly limited to a defined purpose?",
            risk_level="Low",
            mitigation="Document purpose and restrict access",
        ),
        DpiaCheck(
            category="Rights",
            question="Can users request deletion or correction?",
            risk_level="Medium",
            mitigation="Provide user request workflow",
        ),
        DpiaCheck(
            category="Security",
            question="Are controls in place against unauthorized access?",
            risk_level="Medium",
            mitigation="Access control, encryption at rest",
        ),
    ]


def _split_evidence_paths(evidence: str) -> List[str]:
    parts = []
    for chunk in evidence.replace(",", "+").split("+"):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)  # Keep user-provided evidence tokens
    return parts


def _log_evidence_artifacts(evidence: str, evidence_root: Path, prefix: str) -> None:
    for idx, item in enumerate(_split_evidence_paths(evidence), start=1):
        candidate = evidence_root / item
        if candidate.exists() and candidate.is_file():
            mlflow.log_artifact(str(candidate), artifact_path=f"evidence/{prefix}")  # Upload evidence file
            mlflow.set_tag(f"{prefix}.evidence_{idx}", str(candidate))  # Store resolved path
        else:
            mlflow.set_tag(f"{prefix}.evidence_{idx}", item)  # Keep reference if file missing


def _set_db_tracking_uri() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance audit using MLflow metadata only.")
    parser.add_argument("--evidence-root", type=str, default=".")
    args = parser.parse_args()

    _set_db_tracking_uri()
    matrix = build_compliance_matrix()
    dpia = build_dpia_checklist()

    evidence_root = Path(args.evidence_root)  # Resolve evidence paths from this base

    mlflow.set_experiment("responsible-ai-compliance")  # Single experiment for compliance runs

    with mlflow.start_run(run_name="compliance_audit") as run:
        for idx, req in enumerate(matrix, start=1):
            prefix = f"compliance.{idx}"
            mlflow.set_tag(f"{prefix}.requirement", req.requirement)  # Requirement statement
            mlflow.set_tag(f"{prefix}.control", req.control)  # Implemented control
            mlflow.set_tag(f"{prefix}.status", req.status)  # Compliance status
            mlflow.set_tag(f"{prefix}.evidence", req.evidence)  # Evidence references
            _log_evidence_artifacts(req.evidence, evidence_root, prefix)

        for idx, check in enumerate(dpia, start=1):
            prefix = f"dpia.{idx}"
            mlflow.set_tag(f"{prefix}.category", check.category)  # DPIA category
            mlflow.set_tag(f"{prefix}.risk_level", check.risk_level)  # Risk rating
            mlflow.set_tag(f"{prefix}.mitigation", check.mitigation)  # Mitigation plan
            mlflow.set_tag(f"{prefix}.question", check.question)  # DPIA prompt

        print(f"Logged compliance audit to MLflow: {run.info.run_id}")


if __name__ == "__main__":
    main()
