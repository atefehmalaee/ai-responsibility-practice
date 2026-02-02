import argparse  # CLI parsing
import csv  # CSV writing
from datetime import date  # Default date
from pathlib import Path  # Path utilities


def main() -> None:
    # Purpose: Append a governance event to the accountability audit log.
    parser = argparse.ArgumentParser(description="Append an entry to audit_log.csv.")
    parser.add_argument("--date", type=str, default=date.today().isoformat())
    parser.add_argument("--decision", type=str, required=True)
    parser.add_argument("--owner", type=str, required=True)
    parser.add_argument("--impact", type=str, default="")
    parser.add_argument("--risk-level", type=str, default="")
    parser.add_argument("--approved-by", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--comment", type=str, default="")
    args = parser.parse_args()

    log_path = Path(__file__).with_name("audit_log.csv")
    write_header = not log_path.exists()

    row = {
        "date": args.date,
        "decision": args.decision,
        "owner": args.owner,
        "impact": args.impact,
        "risk_level": args.risk_level,
        "approved_by": args.approved_by,
        "notes": args.notes,
        "comment": args.comment,
    }

    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended entry to {log_path}")


if __name__ == "__main__":
    main()
