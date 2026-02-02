import argparse  # CLI parsing
import re  # PII pattern matching
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import Dict, List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular data
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.metrics import accuracy_score, roc_auc_score  # Summary metrics
from sklearn.model_selection import train_test_split  # Train/test split


@dataclass
class PiiFinding:
    column: str
    pii_type: str
    match_count: int
    example: str


@dataclass
class ModelMetrics:
    variant: str
    accuracy: float
    roc_auc: float
    features_used: int


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")  # Email pattern
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")  # Phone pattern (simple)


def generate_synthetic_dataset(seed: int, n_samples: int = 1200) -> pd.DataFrame:
    rng = np.random.RandomState(seed)  # Reproducible randomness
    first_names = np.array(["Alex", "Sam", "Taylor", "Jordan", "Riley", "Casey"])
    last_names = np.array(["Lee", "Patel", "Garcia", "Kim", "Smith", "Nguyen"])
    domains = np.array(["example.com", "mail.test", "demo.org"])

    first = rng.choice(first_names, size=n_samples)
    last = rng.choice(last_names, size=n_samples)
    name = np.char.add(np.char.add(first, " "), last)
    email = np.char.add(np.char.add(np.char.lower(first), "."), np.char.add(np.char.lower(last), "@"))
    email = np.char.add(email, rng.choice(domains, size=n_samples))
    phone = np.array([f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}" for _ in range(n_samples)])

    age = rng.randint(18, 70, size=n_samples)
    zipcode = rng.randint(10000, 99999, size=n_samples)
    income = rng.normal(60000, 15000, size=n_samples).round(0).astype(int)

    # Construct a label with some dependence on income/age (non-PII signal)
    logit = 0.00004 * income + 0.02 * (age - 40) + rng.normal(0, 0.5, size=n_samples)
    y = (logit > 0.0).astype(int)

    return pd.DataFrame(
        {
            "name": name,
            "email": email,
            "phone": phone,
            "age": age,
            "zipcode": zipcode,
            "income": income,
            "label": y,
        }
    )


def scan_pii(df: pd.DataFrame) -> List[PiiFinding]:
    findings = []
    for col in df.columns:
        if df[col].dtype != object:
            continue  # Skip non-text fields
        series = df[col].astype(str)
        email_matches = series.str.contains(EMAIL_RE, regex=True)
        phone_matches = series.str.contains(PHONE_RE, regex=True)
        if email_matches.any():
            example = series[email_matches].iloc[0]
            findings.append(PiiFinding(col, "email", int(email_matches.sum()), example))
        if phone_matches.any():
            example = series[phone_matches].iloc[0]
            findings.append(PiiFinding(col, "phone", int(phone_matches.sum()), example))
        if col == "name":  # Simple heuristic for name column
            findings.append(PiiFinding(col, "name", int(series.notna().sum()), series.iloc[0]))
    return findings


def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    redacted = df.copy()
    redacted["name"] = "[REDACTED]"
    redacted["email"] = "[REDACTED]"
    redacted["phone"] = "[REDACTED]"
    return redacted


def build_features(df: pd.DataFrame, include_pii: bool) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "age": df["age"].astype(float),
            "zipcode": df["zipcode"].astype(float),
            "income": df["income"].astype(float),
        }
    )
    if include_pii:
        # Simple derived features from PII fields to demonstrate leakage risk
        features["name_length"] = df["name"].astype(str).str.len().astype(float)
        features["email_domain_len"] = df["email"].astype(str).str.split("@").str[-1].str.len().astype(float)
        features["phone_last4"] = df["phone"].astype(str).str[-4:].astype(float)
    return features


def train_and_eval(X: pd.DataFrame, y: np.ndarray, seed: int, variant: str) -> ModelMetrics:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return ModelMetrics(
        variant=variant,
        accuracy=float(accuracy_score(y_test, y_pred)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        features_used=int(X.shape[1]),
    )


def write_privacy_note(path: Path, pii_counts: Dict[str, int], metrics: List[ModelMetrics]) -> None:
    lines = [
        "Privacy Risk Note",
        "=================",
        "",
        "PII Findings",
        "------------",
        f"- Name: {pii_counts.get('name', 0)} rows",
        f"- Email: {pii_counts.get('email', 0)} rows",
        f"- Phone: {pii_counts.get('phone', 0)} rows",
        "",
        "Feature Minimization",
        "--------------------",
    ]
    for m in metrics:
        lines.append(
            f"- {m.variant}: accuracy={m.accuracy:.3f}, roc_auc={m.roc_auc:.3f}, features={m.features_used}"
        )
    lines.extend(
        [
            "",
            "Risk Controls",
            "-------------",
            "- Remove or redact direct identifiers before modeling.",
            "- Avoid derived features from PII unless strictly necessary.",
            "- Document data retention and access controls.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy audit with PII scan and feature minimization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports")
    parser.add_argument("--save-synthetic", action="store_true")
    args = parser.parse_args()

    df = generate_synthetic_dataset(args.seed)

    findings = scan_pii(df)
    pii_scan = pd.DataFrame([finding.__dict__ for finding in findings])
    pii_counts = pii_scan.groupby("pii_type")["match_count"].sum().to_dict()

    redacted = redact_pii(df)

    X_full = build_features(df, include_pii=True)
    X_min = build_features(redacted, include_pii=False)
    y = df["label"].to_numpy()

    metrics_full = train_and_eval(X_full, y, args.seed, variant="full_features")
    metrics_min = train_and_eval(X_min, y, args.seed, variant="minimized_features")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_synthetic:
        df.to_csv(out_dir / "synthetic_dataset.csv", index=False)
    pii_scan.to_csv(out_dir / "pii_scan.csv", index=False)
    pd.DataFrame([metrics_full.__dict__, metrics_min.__dict__]).to_csv(
        out_dir / "feature_minimization.csv", index=False
    )
    redacted.head(10).to_csv(out_dir / "redacted_sample.csv", index=False)
    write_privacy_note(out_dir / "privacy_risk_note.md", pii_counts, [metrics_full, metrics_min])

    print(f"Wrote {out_dir / 'pii_scan.csv'}")
    print(f"Wrote {out_dir / 'feature_minimization.csv'}")
    print(f"Wrote {out_dir / 'redacted_sample.csv'}")
    print(f"Wrote {out_dir / 'privacy_risk_note.md'}")
    if args.save_synthetic:
        print(f"Wrote {out_dir / 'synthetic_dataset.csv'}")


if __name__ == "__main__":
    main()
