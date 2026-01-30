import argparse  # CLI parsing
import csv  # CSV outputs
import hashlib  # Anonymized IDs
import json  # JSON outputs
import uuid  # Stable identifiers
from dataclasses import dataclass  # Lightweight data container
from datetime import datetime, timezone  # Timestamps
from pathlib import Path  # Path utilities
from typing import Dict, List, Optional  # Type hints


@dataclass
class ModelMetadata:
    model_name: str
    model_version: str
    owner: str
    intended_use: str


@dataclass
class DecisionLogEntry:
    decision_id: str
    timestamp: str
    input_summary: str
    model_output: str
    confidence: float
    explanation: str
    risk_level: str
    requires_review: bool
    reviewer_id: str
    review_decision: str
    review_notes: str


@dataclass
class RiskEntry:
    risk_id: str
    description: str
    severity: str
    likelihood: str
    mitigation: str
    status: str
    version: str
    owner: str
    last_reviewed: str
    incident_triggered_count: int
    mitigation_status: str


@dataclass
class IncidentEntry:
    incident_id: str
    risk_id: str
    timestamp: str
    description: str
    severity: str
    status: str
    resolved_at: str


@dataclass
class SurveyResponse:
    response_id: str
    timestamp: str
    anonymized_user_id: str
    rating: int
    response_text: str
    retrain_flag: bool
    policy_review_flag: bool
    ux_review_flag: bool


@dataclass
class AccessibilityReview:
    reviewer: str
    review_date: str
    status: str
    notes: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # Consistent UTC timestamps for audit trails


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    paths = {  # Standardized governance artifact folders
        "base": base,
        "logs": base / "logs",
        "reviews": base / "reviews",
        "risks": base / "risks",
        "surveys": base / "surveys",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)  # Create folders if missing
    return paths


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return  # Skip empty writes to keep files clean
    write_header = not path.exists()  # Only add header on first write
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")  # Append one event per line


def build_survey_template() -> str:
    return "\n".join(  # Human-centered survey prompts
        [
            "User Survey",
            "===========",
            "",
            "1. What task are you trying to complete with this system?",
            "2. How often do you use it (daily/weekly/monthly)?",
            "3. What did the system do well?",
            "4. What confused you or slowed you down?",
            "5. Did you receive any incorrect or harmful output?",
            "6. How confident are you in the results (1-5)?",
            "7. What would you change to make it more trustworthy?",
            "",
        ]
    )


def build_accessibility_checklist() -> str:
    return "\n".join(  # WCAG-aligned checks
        [
            "Accessibility Checklist",
            "======================",
            "",
            "- [ ] Keyboard navigation works for all actions",
            "- [ ] Color contrast meets WCAG AA",
            "- [ ] Forms have labels and error messages",
            "- [ ] Screen reader testing completed",
            "- [ ] Captions or transcripts for media",
            "",
        ]
    )


def build_impact_assessment() -> List[RiskEntry]:
    now = _utc_now()  # Shared timestamp for baseline register
    return [
        RiskEntry(
            risk_id="R-001",
            description="Incorrect decision harms user outcome",
            severity="High",
            likelihood="Medium",
            mitigation="Human review for low-confidence outputs",
            status="Open",
            version="1.0",
            owner="Responsible AI",
            last_reviewed=now,
            incident_triggered_count=0,
            mitigation_status="planned",
        ),
        RiskEntry(
            risk_id="R-002",
            description="Confusing UI leads to misuse",
            severity="Medium",
            likelihood="Medium",
            mitigation="Usability testing and clear explanations",
            status="Open",
            version="1.0",
            owner="Product",
            last_reviewed=now,
            incident_triggered_count=0,
            mitigation_status="planned",
        ),
        RiskEntry(
            risk_id="R-003",
            description="Accessibility barriers for screen readers",
            severity="Medium",
            likelihood="Low",
            mitigation="WCAG checks and keyboard navigation support",
            status="Open",
            version="1.0",
            owner="UX",
            last_reviewed=now,
            incident_triggered_count=0,
            mitigation_status="planned",
        ),
    ]


def write_risk_register(path: Path, risks: List[RiskEntry]) -> None:
    rows = [r.__dict__ for r in risks]  # Serialize dataclasses to rows
    if path.exists():
        path.unlink()  # Replace to avoid stale entries
    _write_csv(path, rows)


def append_incident(path: Path, incident: IncidentEntry) -> None:
    _write_csv(path, [incident.__dict__])  # Append incident as a new row


def update_risk_on_incident(path: Path, risk_id: str) -> None:
    if not path.exists():
        return  # No register to update
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        if row["risk_id"] == risk_id:
            row["incident_triggered_count"] = str(int(row["incident_triggered_count"]) + 1)  # Increment count
            row["last_reviewed"] = _utc_now()  # Keep evidence of review cadence
            row["mitigation_status"] = row["mitigation_status"] or "planned"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def anonymize_user_id(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()  # Irreversible user ID


def requires_human_review(confidence: float, risk_level: str, threshold: float) -> bool:
    if risk_level.lower() == "high":
        return True  # Always route high-risk decisions to humans
    return confidence < threshold  # Low-confidence routing


def write_model_metadata(path: Path, metadata: ModelMetadata) -> None:
    path.write_text(json.dumps(metadata.__dict__, indent=2))  # Audit-ready metadata record


def write_accessibility_review(path: Path, review: AccessibilityReview) -> None:
    _write_csv(path, [review.__dict__])  # Persist manual sign-off


def append_survey_response(path: Path, response: SurveyResponse) -> None:
    _write_csv(path, [response.__dict__])  # Store structured feedback


def append_decision_logs(logs_dir: Path, entry: DecisionLogEntry) -> None:
    _write_csv(logs_dir / "decision_log.csv", [entry.__dict__])  # Tabular audit log
    _write_jsonl(logs_dir / "decision_log.jsonl", [entry.__dict__])  # Event stream for tooling


def main() -> None:
    parser = argparse.ArgumentParser(description="User impact audit with human-centered governance.")
    parser.add_argument("--out", type=str, default="reports")
    parser.add_argument("--model-name", type=str, default="baseline-model")
    parser.add_argument("--model-version", type=str, default="1.0.0")
    parser.add_argument("--model-owner", type=str, default="Responsible AI")
    parser.add_argument("--intended-use", type=str, default="Demonstration and governance testing")
    parser.add_argument("--confidence", type=float, default=0.62)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--risk-level", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--input-summary", type=str, default="Sample input summary")
    parser.add_argument("--model-output", type=str, default="Sample model output")
    parser.add_argument("--explanation", type=str, default="Placeholder explanation (SHAP/LIME/LLM rationale)")
    parser.add_argument("--reviewer-id", type=str, default="")
    parser.add_argument("--review-decision", type=str, default="", choices=["", "approve", "reject", "edit"])
    parser.add_argument("--review-notes", type=str, default="")
    parser.add_argument("--survey-response", type=str, default="")
    parser.add_argument("--survey-rating", type=int, default=3)
    parser.add_argument("--survey-user-id", type=str, default="")
    parser.add_argument("--append-incident", action="store_true")
    parser.add_argument("--incident-risk-id", type=str, default="R-001")
    parser.add_argument("--incident-description", type=str, default="User-reported harm")
    parser.add_argument("--incident-severity", type=str, default="Medium")
    parser.add_argument("--incident-status", type=str, default="Open")
    parser.add_argument("--accessibility-reviewed", action="store_true")
    parser.add_argument("--accessibility-reviewer", type=str, default="")
    parser.add_argument("--accessibility-notes", type=str, default="")
    args = parser.parse_args()  # CLI inputs drive governance workflow

    out_dir = Path(args.out)
    paths = _ensure_dirs(out_dir)  # Create audit-ready folder structure

    # --- Governance metadata ---
    metadata = ModelMetadata(
        model_name=args.model_name,
        model_version=args.model_version,
        owner=args.model_owner,
        intended_use=args.intended_use,
    )
    write_model_metadata(paths["logs"] / "model_metadata.json", metadata)  # Persist model registry info

    # --- Risk register (living artifact) ---
    risk_register = build_impact_assessment()
    write_risk_register(paths["risks"] / "risk_register.csv", risk_register)
    _write_csv(paths["base"] / "impact_assessment.csv", [r.__dict__ for r in risk_register])  # Summary copy

    # --- Human-in-the-loop enforcement ---
    needs_review = requires_human_review(args.confidence, args.risk_level, args.confidence_threshold)
    reviewer_id = args.reviewer_id or ("pending" if needs_review else "system")  # Placeholder for audit trail
    review_decision = args.review_decision or ("pending" if needs_review else "auto_approve")
    decision_entry = DecisionLogEntry(
        decision_id=str(uuid.uuid4()),
        timestamp=_utc_now(),
        input_summary=args.input_summary,
        model_output=args.model_output,
        confidence=args.confidence,
        explanation=args.explanation,
        risk_level=args.risk_level,
        requires_review=needs_review,
        reviewer_id=reviewer_id,
        review_decision=review_decision,
        review_notes=args.review_notes,
    )
    append_decision_logs(paths["logs"], decision_entry)  # Persist decision traceability

    # --- Surveys and feedback loop ---
    Path(paths["base"] / "user_survey.md").write_text(build_survey_template())  # Publish template
    if args.survey_response:
        anonymized = anonymize_user_id(args.survey_user_id or str(uuid.uuid4()), "survey-salt")
        retrain_flag = args.survey_rating <= 2  # Low ratings trigger model review
        policy_review_flag = "harm" in args.survey_response.lower()  # Keyword-based escalation
        ux_review_flag = args.survey_rating <= 3  # Usability check for low satisfaction
        survey = SurveyResponse(
            response_id=str(uuid.uuid4()),
            timestamp=_utc_now(),
            anonymized_user_id=anonymized,
            rating=args.survey_rating,
            response_text=args.survey_response,
            retrain_flag=retrain_flag,
            policy_review_flag=policy_review_flag,
            ux_review_flag=ux_review_flag,
        )
        append_survey_response(paths["surveys"] / "survey_responses.csv", survey)
        if policy_review_flag:
            incident = IncidentEntry(
                incident_id=str(uuid.uuid4()),
                risk_id=args.incident_risk_id,
                timestamp=_utc_now(),
                description="Survey-triggered policy review",
                severity="High",
                status="Open",
                resolved_at="",
            )
            append_incident(paths["risks"] / "incidents.csv", incident)
            update_risk_on_incident(paths["risks"] / "risk_register.csv", args.incident_risk_id)  # Update register

    # --- Incident tracking ---
    if args.append_incident:
        incident = IncidentEntry(
            incident_id=str(uuid.uuid4()),
            risk_id=args.incident_risk_id,
            timestamp=_utc_now(),
            description=args.incident_description,
            severity=args.incident_severity,
            status=args.incident_status,
            resolved_at="",
        )
        append_incident(paths["risks"] / "incidents.csv", incident)
        update_risk_on_incident(paths["risks"] / "risk_register.csv", args.incident_risk_id)  # Keep risk stats in sync

    # --- Accessibility checks ---
    Path(paths["base"] / "accessibility_checklist.md").write_text(build_accessibility_checklist())  # Publish checklist
    if args.accessibility_reviewed:
        review = AccessibilityReview(
            reviewer=args.accessibility_reviewer or "Reviewer",
            review_date=_utc_now(),
            status="approved",
            notes=args.accessibility_notes,
        )
        write_accessibility_review(paths["reviews"] / "accessibility_review.csv", review)  # Record sign-off

    print(f"Wrote {paths['base'] / 'user_survey.md'}")
    print(f"Wrote {paths['base'] / 'impact_assessment.csv'}")
    print(f"Wrote {paths['base'] / 'accessibility_checklist.md'}")
    print(f"Wrote decision logs to {paths['logs']}")
    print(f"Wrote survey data to {paths['surveys']}")
    print(f"Wrote risk data to {paths['risks']}")


if __name__ == "__main__":
    main()
