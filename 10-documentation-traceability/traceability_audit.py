import argparse  # CLI parsing
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from typing import List  # Type hints

import numpy as np  # Numerical ops
import pandas as pd  # Tabular data


@dataclass
class DataLineage:
    dataset_name: str
    source: str
    version: str
    created_by: str
    retention_policy: str
    pii_present: bool


@dataclass
class ModelRegistryEntry:
    model_name: str
    version: str
    owner: str
    metrics: str
    training_data_version: str
    notes: str


@dataclass
class ExperimentLog:
    run_id: str
    params: str
    metrics: str
    artifacts: str
    timestamp: str


def build_lineage() -> List[DataLineage]:
    return [
        DataLineage(
            dataset_name="synthetic_classification",
            source="sklearn.datasets.make_classification",
            version="v1.0",
            created_by="Data Scientist",
            retention_policy="Delete after 90 days",
            pii_present=False,
        )
    ]


def build_model_registry() -> List[ModelRegistryEntry]:
    return [
        ModelRegistryEntry(
            model_name="logreg_baseline",
            version="1.0.0",
            owner="ML Engineer",
            metrics="accuracy=0.80, roc_auc=0.86",
            training_data_version="v1.0",
            notes="Baseline model for responsibility practice",
        ),
        ModelRegistryEntry(
            model_name="logreg_baseline",
            version="1.1.0",
            owner="ML Engineer",
            metrics="accuracy=0.81, roc_auc=0.86",
            training_data_version="v1.0",
            notes="Hyperparameter tuning update",
        ),
    ]


def build_experiment_log() -> List[ExperimentLog]:
    return [
        ExperimentLog(
            run_id="exp_001",
            params="max_iter=1000, C=1.0",
            metrics="accuracy=0.80, roc_auc=0.86",
            artifacts="model.pkl, report.csv",
            timestamp="2026-01-28T00:00:00Z",
        ),
        ExperimentLog(
            run_id="exp_002",
            params="max_iter=1000, C=0.8",
            metrics="accuracy=0.81, roc_auc=0.86",
            artifacts="model.pkl, report.csv",
            timestamp="2026-01-28T01:00:00Z",
        ),
    ]


def write_change_log(path: Path) -> None:
    content = "\n".join(
        [
            "Change Log",
            "==========",
            "",
            "- 2026-01-28: Created dataset lineage record and model registry entries.",
            "- 2026-01-28: Logged two experiment runs with parameters and metrics.",
            "",
        ]
    )
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Documentation audit with lineage and registry logs.")
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([l.__dict__ for l in build_lineage()]).to_csv(out_dir / "data_lineage.csv", index=False)
    pd.DataFrame([m.__dict__ for m in build_model_registry()]).to_csv(
        out_dir / "model_registry.csv", index=False
    )
    pd.DataFrame([e.__dict__ for e in build_experiment_log()]).to_csv(
        out_dir / "experiment_log.csv", index=False
    )
    write_change_log(out_dir / "change_log.md")

    print(f"Wrote {out_dir / 'data_lineage.csv'}")
    print(f"Wrote {out_dir / 'model_registry.csv'}")
    print(f"Wrote {out_dir / 'experiment_log.csv'}")
    print(f"Wrote {out_dir / 'change_log.md'}")


if __name__ == "__main__":
    main()
