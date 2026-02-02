import argparse  # CLI parsing
import hashlib  # Dataset fingerprint
import json  # Structured artifacts
import tempfile  # Temporary artifact files
from dataclasses import dataclass  # Lightweight data container
from datetime import datetime, timezone  # Timestamps
from pathlib import Path  # Path utilities
from typing import Dict, List  # Type hints

import mlflow  # MLflow tracking APIs
import mlflow.sklearn  # MLflow sklearn model logging
from mlflow.tracking import MlflowClient  # Registry tagging
import numpy as np  # Numerical ops
from sklearn.datasets import make_classification  # Synthetic data
from sklearn.linear_model import LogisticRegression  # Baseline model


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # Consistent timestamps for audit logs


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


def build_change_log() -> str:
    return "\n".join(
        [
            "Change Log",
            "==========",
            "",
            "- 2026-01-28: Created dataset lineage record and model registry entries.",
            "- 2026-01-28: Logged two experiment runs with parameters and metrics.",
            "",
        ]
    )


def _parse_kv_pairs(raw: str) -> Dict[str, float]:
    pairs = {}  # Simple key=value parser for params/metrics strings
    for item in raw.split(","):
        key, value = item.strip().split("=")
        pairs[key.strip()] = float(value)
    return pairs


def _log_artifact_text(name: str, content: str, artifact_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{name}", delete=False) as handle:
        handle.write(content)  # Write content to a temp file for MLflow
        temp_path = handle.name
    mlflow.log_artifact(temp_path, artifact_path=artifact_path)  # Persist artifact in run


def _dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    payload = np.concatenate([X.astype(np.float32).ravel(), y.astype(np.float32).ravel()])
    return hashlib.sha256(payload.tobytes()).hexdigest()


def build_model_card(owner: str, intended_use: str, dataset_version: str) -> str:
    return "\n".join(
        [
            "Model Card (Summary)",
            "====================",
            "",
            f"Owner: {owner}",
            f"Intended use: {intended_use}",
            f"Training data version: {dataset_version}",
            "",
        ]
    )


def _set_db_tracking_uri() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Documentation audit using MLflow tracking only.")
    parser.add_argument("--experiment-name", type=str, default="traceability_audit")
    parser.add_argument("--run-prefix", type=str, default="")
    parser.add_argument("--owner", type=str, default="Responsible AI")
    parser.add_argument("--risk-level", type=str, default="Medium")
    parser.add_argument("--data-sensitivity", type=str, default="Low")
    parser.add_argument("--approval-status", type=str, default="Approved")
    parser.add_argument("--intended-use", type=str, default="Demonstration and governance testing")
    args = parser.parse_args()

    _set_db_tracking_uri()
    mlflow.set_experiment(args.experiment_name)  # Group related runs

    run_prefix = args.run_prefix or args.experiment_name
    lineage = build_lineage()[0]  # One lineage record for this demo
    registry_entries = build_model_registry()
    experiment_logs = build_experiment_log()
    common_tags = {
        "owner": args.owner,
        "risk_level": args.risk_level,
        "data_sensitivity": args.data_sensitivity,
        "approval_status": args.approval_status,
    }

    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=42,
    )
    dataset_hash = _dataset_fingerprint(X, y)

    # --- Documentation run: lineage and change log ---
    with mlflow.start_run(run_name=f"{run_prefix}/documentation") as run:
        mlflow.set_tag("run_type", "documentation")
        mlflow.set_tag("dataset_name", lineage.dataset_name)
        mlflow.set_tag("dataset_source", lineage.source)
        mlflow.set_tag("dataset_version", lineage.version)
        mlflow.set_tag("dataset_created_by", lineage.created_by)
        mlflow.set_tag("dataset_retention_policy", lineage.retention_policy)
        mlflow.set_tag("dataset_pii_present", str(lineage.pii_present))
        mlflow.set_tag("dataset_hash", dataset_hash)
        for key, value in common_tags.items():
            mlflow.set_tag(key, value)

        mlflow.log_param("lineage_dataset_name", lineage.dataset_name)
        mlflow.log_param("lineage_version", lineage.version)
        mlflow.log_param("lineage_pii_present", lineage.pii_present)
        mlflow.log_param("intended_use", args.intended_use)
        mlflow.log_metric("compliance_pass", 1.0)

        _log_artifact_text("change_log.md", build_change_log(), "artifacts/lineage")
        _log_artifact_text("data_lineage.json", json.dumps(lineage.__dict__, indent=2), "artifacts/lineage")
        _log_artifact_text(
            "model_registry.json",
            json.dumps([m.__dict__ for m in registry_entries], indent=2),
            "artifacts/registry",
        )
        _log_artifact_text(
            "experiment_log.json",
            json.dumps([e.__dict__ for e in experiment_logs], indent=2),
            "artifacts/experiments",
        )
        _log_artifact_text(
            "model_card.md",
            build_model_card(args.owner, args.intended_use, lineage.version),
            "artifacts/registry",
        )

        print(f"Logged documentation run to MLflow: {run.info.run_id}")

    # --- Experiment runs: params, metrics, artifacts ---
    for exp in experiment_logs:
        params = _parse_kv_pairs(exp.params)
        metrics = _parse_kv_pairs(exp.metrics)
        with mlflow.start_run(run_name=f"{run_prefix}/{exp.run_id}") as run:
            mlflow.set_tag("run_type", "experiment")
            mlflow.set_tag("dataset_name", lineage.dataset_name)
            mlflow.set_tag("dataset_version", lineage.version)
            mlflow.set_tag("dataset_pii_present", str(lineage.pii_present))
            mlflow.set_tag("dataset_hash", dataset_hash)
            mlflow.set_tag("artifact_list", exp.artifacts)
            mlflow.set_tag("timestamp", exp.timestamp)
            for key, value in common_tags.items():
                mlflow.set_tag(key, value)

            for key, value in params.items():
                mlflow.log_param(key, value)  # Track hyperparameters
            for key, value in metrics.items():
                mlflow.log_metric(key, value)  # Track evaluation metrics

            _log_artifact_text(
                "run_summary.json",
                json.dumps(
                    {
                        "run_id": exp.run_id,
                        "params": params,
                        "metrics": metrics,
                        "artifacts": exp.artifacts,
                        "timestamp": exp.timestamp,
                    },
                    indent=2,
                ),
                "artifacts/experiments",
            )

            print(f"Logged experiment run to MLflow: {run.info.run_id}")

    # --- Model registry: register two versions from tracked runs ---
    for entry in registry_entries:
        params = _parse_kv_pairs("max_iter=1000, C=1.0")
        model = LogisticRegression(max_iter=int(params["max_iter"]), C=params["C"])
        model.fit(X, y)  # Train a simple model for registry demo

        with mlflow.start_run(run_name=f"{run_prefix}/registry_{entry.version}") as run:
            mlflow.set_tag("run_type", "registry")
            mlflow.set_tag("dataset_name", lineage.dataset_name)
            mlflow.set_tag("dataset_version", lineage.version)
            mlflow.set_tag("dataset_pii_present", str(lineage.pii_present))
            mlflow.set_tag("dataset_hash", dataset_hash)
            mlflow.set_tag("model_owner", entry.owner)
            mlflow.set_tag("training_data_version", entry.training_data_version)
            mlflow.set_tag("registry_notes", entry.notes)
            mlflow.set_tag("registry_version_hint", entry.version)
            for key, value in common_tags.items():
                mlflow.set_tag(key, value)

            mlflow.log_param("model_name", entry.model_name)
            mlflow.log_param("training_data_version", entry.training_data_version)
            mlflow.log_param("owner", entry.owner)

            for key, value in _parse_kv_pairs(entry.metrics).items():
                mlflow.log_metric(key, value)  # Store registry metrics

            mlflow.sklearn.log_model(model, artifact_path="model")  # Log model artifact
            model_uri = f"runs:/{run.info.run_id}/model"
            model_version = mlflow.register_model(model_uri, entry.model_name)  # Create registered model version
            client = MlflowClient()
            client.set_model_version_tag(entry.model_name, model_version.version, "dataset_hash", dataset_hash)
            client.set_model_version_tag(entry.model_name, model_version.version, "owner", args.owner)
            client.set_model_version_tag(entry.model_name, model_version.version, "risk_level", args.risk_level)
            client.set_model_version_tag(
                entry.model_name, model_version.version, "data_sensitivity", args.data_sensitivity
            )
            client.set_model_version_tag(
                entry.model_name, model_version.version, "approval_status", args.approval_status
            )

            print(f"Registered model version from run: {run.info.run_id}")

    # View results: run `mlflow ui` in this folder and open http://127.0.0.1:5000


if __name__ == "__main__":
    main()
