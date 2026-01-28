import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import pandas as pd  # Tabular data


@dataclass
class ImpactItem:
    impact: str
    severity: str
    likelihood: str
    mitigation: str


def build_survey() -> str:
    return "\n".join(
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


def build_impact_assessment() -> List[ImpactItem]:
    return [
        ImpactItem(
            impact="Incorrect decision harms user outcome",
            severity="High",
            likelihood="Medium",
            mitigation="Human review for low-confidence outputs",
        ),
        ImpactItem(
            impact="Confusing UI leads to misuse",
            severity="Medium",
            likelihood="Medium",
            mitigation="Usability testing and clear explanations",
        ),
        ImpactItem(
            impact="Accessibility barriers for screen readers",
            severity="Medium",
            likelihood="Low",
            mitigation="WCAG checks and keyboard navigation support",
        ),
    ]


def build_accessibility_checklist() -> str:
    return "\n".join(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="User impact audit with survey and impact assessment.")
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Path(out_dir / "user_survey.md").write_text(build_survey())
    pd.DataFrame([i.__dict__ for i in build_impact_assessment()]).to_csv(
        out_dir / "impact_assessment.csv", index=False
    )
    Path(out_dir / "accessibility_checklist.md").write_text(build_accessibility_checklist())

    print(f"Wrote {out_dir / 'user_survey.md'}")
    print(f"Wrote {out_dir / 'impact_assessment.csv'}")
    print(f"Wrote {out_dir / 'accessibility_checklist.md'}")


if __name__ == "__main__":
    main()
