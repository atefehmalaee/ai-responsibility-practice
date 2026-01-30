import argparse  # CLI parsing
import sys  # Exit for missing data
from dataclasses import dataclass  # Lightweight data container
from pathlib import Path  # Path utilities
from urllib.request import urlretrieve  # Dataset download

import numpy as np  # Numerical ops
import pandas as pd  # Tabular reporting
from sklearn.linear_model import LogisticRegression  # Baseline classifier
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.utils import resample  # Over/under sampling

# Optional dependencies (kept at top for clarity)
try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    from fairlearn.postprocessing import ThresholdOptimizer

    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR
    from aif360.algorithms.postprocessing import RejectOptionClassification

    AIF360_AVAILABLE = True
except Exception:
    AIF360_AVAILABLE = False

try:
    import tensorflow as tf
    from aif360.algorithms.inprocessing import AdversarialDebiasing

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


@dataclass
class GroupMetrics:
    n: int  # Group sample count
    accuracy: float  # Accuracy within group
    selection_rate: float  # Predicted positive rate
    tpr: float  # True positive rate
    fpr: float  # False positive rate


@dataclass
class CounterfactualSummary: #how often predictions flip
    stage: str
    flip_rate: float


def _safe_rate(num: int, denom: int) -> float:
    if denom == 0:  # Avoid division by zero
        return float("nan")  # Use NaN for undefined rates
    return num / denom  # Standard rate calculation


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# Technique: Census Income (Adult) dataset loader
# Description: Loads tabular data, encodes features, and defines sensitive attribute.
def load_adult_dataset(csv_path: str, sensitive_attr: str) -> tuple:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = pd.read_csv(csv_path, names=columns, na_values="?", skipinitialspace=True)
    df = df.dropna()

    df["label"] = (df["income"].str.strip() == ">50K").astype(int)
    if sensitive_attr == "sex":
        df["sensitive"] = (df["sex"].str.strip() == "Male").astype(int)
    elif sensitive_attr == "race":
        df["sensitive"] = (df["race"].str.strip() == "White").astype(int)
    else:
        raise ValueError("sensitive_attr must be 'sex' or 'race'")

    drop_cols = ["income", "label", "sensitive"]
    X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
    X = X.astype(float)  # Ensure numeric dtype for downstream math
    y = df["label"].to_numpy()
    sensitive = df["sensitive"].to_numpy()
    return X.to_numpy(), y, sensitive


# Technique: Dataset downloader (utility)
# Description: Downloads the Census Income (Adult) dataset if missing.
def download_adult_dataset(dest_path: str) -> None:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urlretrieve(url, dest_path)


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict:
    metrics = {}  # Accumulator for per-group metrics
    for value in sorted(np.unique(group)):  # Iterate groups deterministically
        mask = group == value  # Group membership mask
        yt = y_true[mask]  # True labels for group
        yp = y_pred[mask]  # Predicted labels for group

        tp = int(np.sum((yt == 1) & (yp == 1)))  # True positives
        tn = int(np.sum((yt == 0) & (yp == 0)))  # True negatives
        fp = int(np.sum((yt == 0) & (yp == 1)))  # False positives
        fn = int(np.sum((yt == 1) & (yp == 0)))  # False negatives

        metrics[value] = GroupMetrics(  # Store computed metrics
            n=int(mask.sum()),  # Group size
            accuracy=_safe_rate(tp + tn, tp + tn + fp + fn),  # Accuracy
            selection_rate=_safe_rate(tp + fp, tp + tn + fp + fn),  # Positive rate
            tpr=_safe_rate(tp, tp + fn),  # True positive rate
            fpr=_safe_rate(fp, fp + tn),  # False positive rate
        )
    return metrics  # Return per-group metrics map


def fairness_gaps(group_metrics: dict) -> dict:
    values = sorted(group_metrics.keys())  # Group labels
    if len(values) < 2:  # Need at least two groups to compare
        return {"dp_diff": float("nan"), "eo_diff": float("nan"), "avg_odds_diff": float("nan")}

    g0 = group_metrics[values[0]]  # First group
    g1 = group_metrics[values[1]]  # Second group

    dp_diff = abs(g0.selection_rate - g1.selection_rate)  # Demographic parity diff, approval rate gap
    eo_diff = abs(g0.tpr - g1.tpr)  # Equal opportunity diff, true positive rate gap
    avg_odds_diff = 0.5 * (abs(g0.tpr - g1.tpr) + abs(g0.fpr - g1.fpr))  # Avg odds diff, average of true positive rate and false positive rate gaps
    return {"dp_diff": dp_diff, "eo_diff": eo_diff, "avg_odds_diff": avg_odds_diff}  # Gap summary


# === Pre-processing techniques ===
# Technique: Reweighing (pre-processing)
# Description: Reweighing assigns larger weights to underrepresented group–label combinations so the model learns fairly from all groups.
def reweighing_weights(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(y, dtype=float)
    for g in np.unique(group):
        for label in np.unique(y):
            mask = (group == g) & (y == label)
            if mask.sum() == 0:
                continue
            weights[mask] = 1.0 / mask.sum()
    weights *= len(y) / weights.sum()
    return weights


# Technique: Over/under sampling (pre-processing)
# Description: Balances group representation by resampling.
def resample_by_group(
    X: np.ndarray, y: np.ndarray, group: np.ndarray, group_b: np.ndarray, strategy: str
) -> tuple:
    indices = []
    group_values = np.unique(group)
    counts = {g: int(np.sum(group == g)) for g in group_values}
    target = max(counts.values()) if strategy == "oversample" else min(counts.values())
    for g in group_values:
        group_idx = np.where(group == g)[0]
        if len(group_idx) == 0:
            continue
        sampled = resample(
            group_idx,
            replace=(strategy == "oversample"),
            n_samples=target,
            random_state=42,
        )
        indices.extend(sampled.tolist())
    indices = np.array(indices)
    return X[indices], y[indices], group[indices], group_b[indices]


# === In-processing techniques ===
# Technique: Fairness-aware regularization (in-processing)
# Description: Adds a demographic-parity penalty to the loss during training.
def train_logreg_fairness(
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    seed: int,
    lr: float = 0.1,
    epochs: int = 500,
    dp_lambda: float = 2.0, # strength of fairness penalty
) -> tuple:
    rng = np.random.RandomState(seed)  # Reproducible init
    w = rng.normal(0, 0.1, size=X.shape[1])
    b = 0.0

    for _ in range(epochs):
        logits = X @ w + b
        proba = _sigmoid(logits)
        error = proba - y

        # Demographic parity penalty: difference in mean predictions
        p0 = proba[group == 0].mean()  # Mean prediction for group 0
        p1 = proba[group == 1].mean()  # Mean prediction for group 1
        dp_grad = (p0 - p1)  # Demographic parity gap (prediction mean diff)
        grad_dp = np.zeros_like(proba)  # Per-sample DP gradient term
        grad_dp[group == 0] = dp_grad / max(1, (group == 0).sum())  # Push group 0 toward parity
        grad_dp[group == 1] = -dp_grad / max(1, (group == 1).sum())  # Push group 1 toward parity

        grad_w = (X.T @ (error + dp_lambda * grad_dp)) / len(y)
        grad_b = float(np.mean(error + dp_lambda * grad_dp))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict_proba_custom(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(X @ w + b)


# === Post-processing techniques ===
# Technique: Equalized selection rates (post-processing)
# Description: Finds group thresholds to match a target selection rate.
def equalize_selection_rate_thresholds(proba: np.ndarray, group: np.ndarray, target_rate: float) -> dict:
    thresholds = {}  # Group-specific thresholds
    for value in sorted(np.unique(group)):  # Iterate groups deterministically
        mask = group == value  # Group membership
        if mask.sum() == 0:  # Handle empty group safely
            thresholds[value] = 0.5  # Default threshold
            continue
        thresholds[value] = float(np.quantile(proba[mask], 1 - target_rate))  # Match rate
    return thresholds  # Thresholds per group


def apply_group_thresholds(proba: np.ndarray, group: np.ndarray, thresholds: dict) -> np.ndarray:
    preds = np.zeros_like(proba, dtype=int)  # Output predictions
    for value, thresh in thresholds.items():  # Apply threshold per group
        mask = group == value  # Group membership
        preds[mask] = (proba[mask] >= thresh).astype(int)  # Thresholded predictions
    return preds  # Final predictions


# Technique: Equalized odds threshold search (post-processing)
# Description: Searches thresholds that minimize average odds difference.
def equalized_odds_thresholds(
    proba: np.ndarray, y_true: np.ndarray, group: np.ndarray, grid: np.ndarray
) -> dict:
    best = {}
    best_score = float("inf")
    for t0 in grid:
        for t1 in grid:
            thresholds = {0: t0, 1: t1}
            y_pred = apply_group_thresholds(proba, group, thresholds)
            metrics = compute_group_metrics(y_true, y_pred, group)
            gaps = fairness_gaps(metrics)
            score = gaps["avg_odds_diff"]
            if score < best_score:
                best_score = score
                best = thresholds
    return best


def build_report(stage: str, metrics: dict) -> pd.DataFrame:
    rows = []  # Collect report rows
    for group_value, gm in metrics.items():  # Iterate per-group metrics
        rows.append(
            {
                "stage": stage,  # Baseline or mitigated
                "group": int(group_value),  # Group label
                "n": gm.n,  # Group sample count
                "accuracy": gm.accuracy,  # Group accuracy
                "selection_rate": gm.selection_rate,  # Group positive rate
                "tpr": gm.tpr,  # Group true positive rate
                "fpr": gm.fpr,  # Group false positive rate
            }
        )
    return pd.DataFrame(rows)  # Report table


# Technique: Intersectional audit (evaluation)
# Description: Computes metrics on combined sensitive attributes.
def intersectional_report(
    y_true: np.ndarray, y_pred: np.ndarray, group_a: np.ndarray, group_b: np.ndarray, stage: str
) -> pd.DataFrame:
    combo = group_a.astype(int) * 2 + group_b.astype(int)  # Encode pair (a,b) into single group id
    metrics = compute_group_metrics(y_true, y_pred, combo)  # Compute metrics per intersection
    report = build_report(stage, metrics)  # Standardize report format
    report["group_a"] = report["group"] // 2  # Decode group_a from combined id
    report["group_b"] = report["group"] % 2  # Decode group_b from combined id
    report["intersection"] = report["group_a"].astype(str) + "_" + report["group_b"].astype(str)  # Label pair
    return report


# Technique: Counterfactual flip rate (evaluation)
# Description: Measures decision changes when sensitive attribute flips.
def counterfactual_flip_rate(proba: np.ndarray, group: np.ndarray, thresholds: dict) -> float:
    y_pred = apply_group_thresholds(proba, group, thresholds)
    flipped = apply_group_thresholds(proba, 1 - group, thresholds)
    return float(np.mean(y_pred != flipped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness audit with baseline and mitigation.")  # CLI
    parser.add_argument("--seed", type=int, default=42)  # RNG seed
    parser.add_argument("--out", type=str, default="reports")  # Output directory
    parser.add_argument("--adult-path", type=str, default="data/adult.csv")
    parser.add_argument("--sensitive-attr", type=str, default="sex", choices=["sex", "race"])
    parser.add_argument("--download-adult", action="store_true")
    args = parser.parse_args()  # Parse CLI args

    if args.download_adult:
        download_adult_dataset(args.adult_path)
    if not Path(args.adult_path).exists():
        print(
            "Adult dataset not found. Provide --adult-path or use --download-adult to fetch it."
        )
        sys.exit(1)
    X, y, sensitive = load_adult_dataset(args.adult_path, args.sensitive_attr)
    sensitive_b = (X[:, 0] > np.median(X[:, 0])).astype(int)

    X_train, X_test, y_train, y_test, s_train, s_test, s2_train, s2_test = train_test_split(  # Split data
        X, y, sensitive, sensitive_b, test_size=0.3, random_state=args.seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000)  # Baseline model
    model.fit(X_train, y_train)  # Train model

    proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities
    y_pred = (proba >= 0.5).astype(int)  # Default 0.5 threshold predictions

    baseline_metrics = compute_group_metrics(y_test, y_pred, s_test)  # Baseline metrics
    baseline_gaps = fairness_gaps(baseline_metrics)  # Baseline fairness gaps

    # === Pre-processing techniques ===
    # Technique: Reweighing
    sample_weights = reweighing_weights(y_train, s_train)
    model_reweight = LogisticRegression(max_iter=1000)
    model_reweight.fit(X_train, y_train, sample_weight=sample_weights)
    proba_reweight = model_reweight.predict_proba(X_test)[:, 1]
    y_pred_reweight = (proba_reweight >= 0.5).astype(int)
    reweight_metrics = compute_group_metrics(y_test, y_pred_reweight, s_test)
    reweight_gaps = fairness_gaps(reweight_metrics)

    # Technique: Oversampling
    X_over, y_over, s_over, s2_over = resample_by_group(X_train, y_train, s_train, s2_train, "oversample")
    model_over = LogisticRegression(max_iter=1000)
    model_over.fit(X_over, y_over)
    y_pred_over = (model_over.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    oversample_metrics = compute_group_metrics(y_test, y_pred_over, s_test)
    oversample_gaps = fairness_gaps(oversample_metrics)

    # Technique: Undersampling
    X_under, y_under, s_under, s2_under = resample_by_group(X_train, y_train, s_train, s2_train, "undersample")
    model_under = LogisticRegression(max_iter=1000)
    model_under.fit(X_under, y_under)
    y_pred_under = (model_under.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    undersample_metrics = compute_group_metrics(y_test, y_pred_under, s_test)
    undersample_gaps = fairness_gaps(undersample_metrics)

    # === In-processing techniques ===
    # Technique: Fairness-aware regularization
    w_fair, b_fair = train_logreg_fairness(X_train, y_train, s_train, args.seed)
    proba_fair = predict_proba_custom(X_test, w_fair, b_fair)
    y_pred_fair = (proba_fair >= 0.5).astype(int)
    inproc_metrics = compute_group_metrics(y_test, y_pred_fair, s_test)
    inproc_gaps = fairness_gaps(inproc_metrics)

    # === Post-processing techniques ===
    # Technique: Equalized selection rates
    target_rate = float(np.mean(y_pred))  # Overall positive rate
    thresholds = equalize_selection_rate_thresholds(proba, s_test, target_rate)  # Per-group thresholds
    y_pred_mitigated = apply_group_thresholds(proba, s_test, thresholds)  # Mitigated predictions

    mitigated_metrics = compute_group_metrics(y_test, y_pred_mitigated, s_test)  # Mitigated metrics
    mitigated_gaps = fairness_gaps(mitigated_metrics)  # Mitigated gaps

    # Technique: Equalized odds thresholds
    eo_thresholds = equalized_odds_thresholds(proba, y_test, s_test, grid=np.linspace(0.1, 0.9, 17))
    y_pred_eo = apply_group_thresholds(proba, s_test, eo_thresholds)
    eo_metrics = compute_group_metrics(y_test, y_pred_eo, s_test)
    eo_gaps = fairness_gaps(eo_metrics)

    report = pd.concat(  # Combine reports
        [
            build_report("baseline", baseline_metrics),
            build_report("reweighing", reweight_metrics),
            build_report("oversampling", oversample_metrics),
            build_report("undersampling", undersample_metrics),
            build_report("in_processing", inproc_metrics),
            build_report("mitigated", mitigated_metrics),
            build_report("equalized_odds", eo_metrics),
        ],
        ignore_index=True,  # Clean index
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "fairness_report.csv", index=False)  # Save report

    intersection = pd.concat(
        [
            intersectional_report(y_test, y_pred, s_test, s2_test, "baseline"),
            intersectional_report(y_test, y_pred_eo, s_test, s2_test, "equalized_odds"),
        ],
        ignore_index=True,
    )
    intersection.to_csv(out_dir / "fairness_intersectional.csv", index=False)

    counterfactual = [
        CounterfactualSummary("mitigated", counterfactual_flip_rate(proba, s_test, thresholds)),
        CounterfactualSummary("equalized_odds", counterfactual_flip_rate(proba, s_test, eo_thresholds)),
    ]
    pd.DataFrame([c.__dict__ for c in counterfactual]).to_csv(
        out_dir / "counterfactual_summary.csv", index=False
    )

    print("Baseline gaps:", baseline_gaps)  # Report baseline gaps
    print("Reweighing gaps:", reweight_gaps)
    print("Oversampling gaps:", oversample_gaps)
    print("Undersampling gaps:", undersample_gaps)
    print("In-processing gaps:", inproc_gaps)
    print("Mitigated gaps:", mitigated_gaps)  # Report mitigated gaps
    print("Equalized odds gaps:", eo_gaps)
    print(f"Saved reports to {out_dir}")  # Report output location

    # Optional: Fairlearn-based evaluation and mitigation
    if FAIRLEARN_AVAILABLE:
        fairlearn_metrics = MetricFrame(
            metrics={
                "accuracy": lambda yt, yp: float(np.mean(yt == yp)),
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=s_test,
        )
        fairlearn_summary = fairlearn_metrics.by_group.reset_index()
        fairlearn_summary["dp_diff"] = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
        fairlearn_summary["eo_diff"] = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)
        fairlearn_summary.to_csv(out_dir / "fairlearn_metrics.csv", index=False)

        eg = ExponentiatedGradient(LogisticRegression(max_iter=1000), DemographicParity())
        eg.fit(X_train, y_train, sensitive_features=s_train)
        eg_pred = eg.predict(X_test)
        eg_metrics = MetricFrame(
            metrics={"accuracy": lambda yt, yp: float(np.mean(yt == yp)), "selection_rate": selection_rate},
            y_true=y_test,
            y_pred=eg_pred,
            sensitive_features=s_test,
        ).by_group.reset_index()
        eg_metrics["stage"] = "exponentiated_gradient"

        to = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",
            predict_method="predict_proba",
        )
        to.fit(X_train, y_train, sensitive_features=s_train)
        to_pred = to.predict(X_test, sensitive_features=s_test)
        to_metrics = MetricFrame(
            metrics={"accuracy": lambda yt, yp: float(np.mean(yt == yp)), "selection_rate": selection_rate},
            y_true=y_test,
            y_pred=to_pred,
            sensitive_features=s_test,
        ).by_group.reset_index()
        to_metrics["stage"] = "threshold_optimizer"

        pd.concat([eg_metrics, to_metrics], ignore_index=True).to_csv(
            out_dir / "fairlearn_mitigations.csv", index=False
        )
        print("Saved Fairlearn reports.")
    else:
        print("Fairlearn not available or failed: missing dependency.")

    # === AIF360: pre- and post-processing ===
    # Technique: DisparateImpactRemover (pre-processing)
    # Technique: Learning Fair Representations (pre-processing)
    # Technique: Reweighing (pre-processing)
    # Technique: RejectOptionClassification (post-processing)
    if AIF360_AVAILABLE:
        def build_aif_dataset(X_arr: np.ndarray, y_arr: np.ndarray, s_arr: np.ndarray) -> BinaryLabelDataset:
            df_local = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
            df_local["label"] = y_arr
            df_local["sensitive"] = s_arr
            return BinaryLabelDataset(
                df=df_local,
                label_names=["label"],
                protected_attribute_names=["sensitive"],
            )

        def metric_row(stage: str, dataset_true: BinaryLabelDataset, dataset_pred: BinaryLabelDataset) -> dict:
            metric = ClassificationMetric(
                dataset_true,
                dataset_pred,
                unprivileged_groups=[{"sensitive": 0}],
                privileged_groups=[{"sensitive": 1}],
            )
            return {
                "stage": stage,
                "statistical_parity_diff": metric.statistical_parity_difference(),
                "disparate_impact": metric.disparate_impact(),
                "equal_opportunity_diff": metric.equal_opportunity_difference(),
                "average_odds_diff": metric.average_odds_difference(),
            }

        train_ds = build_aif_dataset(X_train, y_train, s_train)
        test_ds = build_aif_dataset(X_test, y_test, s_test)
        pred_ds = build_aif_dataset(X_test, y_pred, s_test)

        baseline_metrics = pd.DataFrame([metric_row("baseline", test_ds, pred_ds)])
        baseline_metrics.to_csv(out_dir / "aif360_metrics.csv", index=False)

        aif_rows = []

        # Reweighing
        rw = Reweighing(unprivileged_groups=[{"sensitive": 0}], privileged_groups=[{"sensitive": 1}])
        rw.fit(train_ds)
        train_rw = rw.transform(train_ds)
        model_rw = LogisticRegression(max_iter=1000)
        model_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)
        rw_pred = (model_rw.predict_proba(test_ds.features)[:, 1] >= 0.5).astype(int)
        rw_pred_ds = build_aif_dataset(test_ds.features, rw_pred, s_test)
        aif_rows.append(metric_row("reweighing", test_ds, rw_pred_ds))

        # Disparate Impact Remover
        dir_algo = DisparateImpactRemover(repair_level=1.0)
        train_dir = dir_algo.fit_transform(train_ds)
        test_dir = dir_algo.fit_transform(test_ds)
        model_dir = LogisticRegression(max_iter=1000)
        model_dir.fit(train_dir.features, train_dir.labels.ravel())
        dir_pred = (model_dir.predict_proba(test_dir.features)[:, 1] >= 0.5).astype(int)
        dir_pred_ds = build_aif_dataset(test_dir.features, dir_pred, s_test)
        aif_rows.append(metric_row("disparate_impact_remover", test_ds, dir_pred_ds))

        # Learning Fair Representations
        lfr = LFR(
            unprivileged_groups=[{"sensitive": 0}],
            privileged_groups=[{"sensitive": 1}],
            k=5,
            Ax=0.01,
            Ay=1.0,
            Az=2.0,
        )
        lfr.fit(train_ds)
        train_lfr = lfr.transform(train_ds)
        test_lfr = lfr.transform(test_ds)
        model_lfr = LogisticRegression(max_iter=1000)
        model_lfr.fit(train_lfr.features, train_lfr.labels.ravel())
        lfr_pred = (model_lfr.predict_proba(test_lfr.features)[:, 1] >= 0.5).astype(int)
        lfr_pred_ds = build_aif_dataset(test_lfr.features, lfr_pred, s_test)
        aif_rows.append(metric_row("learning_fair_representations", test_ds, lfr_pred_ds))

        # Reject Option Classification
        roc = RejectOptionClassification(
            unprivileged_groups=[{"sensitive": 0}],
            privileged_groups=[{"sensitive": 1}],
            low_class_thresh=0.2,
            high_class_thresh=0.8,
            num_class_thresh=50,
            num_ROC_margin=50,
            metric_name="Statistical parity difference",
            metric_ub=0.01,
            metric_lb=-0.01,
        )
        roc = roc.fit(test_ds, pred_ds)
        roc_pred = roc.predict(pred_ds)
        aif_rows.append(metric_row("reject_option_classification", test_ds, roc_pred))

        pd.DataFrame(aif_rows).to_csv(out_dir / "aif360_mitigations.csv", index=False)
        print("Saved AIF360 reports.")
    else:
        print("AIF360 not available or failed: missing dependency.")

    # === AIF360: adversarial debiasing (in-processing) ===
    # Technique: AdversarialDebiasing (requires TensorFlow)
    if TF_AVAILABLE and AIF360_AVAILABLE:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        df_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
        df_train["label"] = y_train
        df_train["sensitive"] = s_train
        train_ds = BinaryLabelDataset(
            df=df_train,
            label_names=["label"],
            protected_attribute_names=["sensitive"],
        )
        adv = AdversarialDebiasing(
            privileged_groups=[{"sensitive": 1}],
            unprivileged_groups=[{"sensitive": 0}],
            scope_name="adv_debiasing",
            sess=sess,
        )
        adv.fit(train_ds)
        adv_pred = adv.predict(train_ds)
        adv_metric = ClassificationMetric(
            train_ds,
            adv_pred,
            unprivileged_groups=[{"sensitive": 0}],
            privileged_groups=[{"sensitive": 1}],
        )
        pd.DataFrame(
            [
                {
                    "stage": "adversarial_debiasing",
                    "statistical_parity_diff": adv_metric.statistical_parity_difference(),
                    "disparate_impact": adv_metric.disparate_impact(),
                    "equal_opportunity_diff": adv_metric.equal_opportunity_difference(),
                    "average_odds_diff": adv_metric.average_odds_difference(),
                }
            ]
        ).to_csv(out_dir / "aif360_adversarial.csv", index=False)
        print("Saved AIF360 adversarial report.")
    else:
        print("AIF360 adversarial debiasing not available or failed: missing dependency.")


if __name__ == "__main__":
    main()  # Run CLI entrypoint
