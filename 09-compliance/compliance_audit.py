import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import pandas as pd  # Tabular data


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


def write_notes(path: Path, matrix: List[ComplianceRequirement]) -> None:
    open_items = [m for m in matrix if m.status.lower() != "in place"]
    content = "\n".join(
        [
            "Compliance Notes",
            "================",
            "",
            f"Total requirements: {len(matrix)}",
            f"Open items: {len(open_items)}",
            "",
            "Next Steps",
            "----------",
            "- Review any items not marked 'In place'.",
            "- Attach evidence links or documents for each control.",
            "",
        ]
    )
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance audit with matrix and DPIA checklist.")
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    matrix = build_compliance_matrix()
    dpia = build_dpia_checklist()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([m.__dict__ for m in matrix]).to_csv(out_dir / "compliance_matrix.csv", index=False)
    pd.DataFrame([d.__dict__ for d in dpia]).to_csv(out_dir / "dpia_checklist.csv", index=False)
    write_notes(out_dir / "compliance_notes.md", matrix)

    print(f"Wrote {out_dir / 'compliance_matrix.csv'}")
    print(f"Wrote {out_dir / 'dpia_checklist.csv'}")
    print(f"Wrote {out_dir / 'compliance_notes.md'}")


if __name__ == "__main__":
    main()
