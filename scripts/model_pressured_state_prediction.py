"""
Model 5: Pressured-State Prediction

Research question:
    Can physiological features help predict whether a survivor is in a pressured state?

Binary label:
    safe:
        healthy, escaped

    pressured:
        injured, downed, ballooned, chaired, eliminated

Input:
    data/processed/master_dataset_with_stress.csv

Output:
    outputs/modeling/pressured_state_prediction/

Main design:
    - Survivor panels only.
    - Group-level cross-validation by clip_id.
    - Logistic Regression with class_weight='balanced'.
    - Compare:
        1. context_baseline: game_phase + player_panel
        2. physiology_full: heart_rate + stress_z + game_phase + player_panel
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset_with_stress.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "pressured_state_prediction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

SAFE_STATES = [
    "healthy",
    "escaped",
]

PRESSURED_STATES = [
    "injured",
    "downed",
    "ballooned",
    "chaired",
    "eliminated",
]

ALL_STATES = SAFE_STATES + PRESSURED_STATES

N_SPLITS = 5
SEED = 42
MAX_SPLIT_TRIALS = 500

TARGET_COL = "pressured_state"

# Model feature sets
FEATURE_SETS = {
    "context_baseline": {
        "numeric": [],
        "categorical": ["game_phase", "player_panel"],
    },
    "physiology_full": {
        "numeric": ["heart_rate", "stress_z"],
        "categorical": ["game_phase", "player_panel"],
    },
}


# ============================================================
# Utility functions
# ============================================================

def normalize_status_name(x):
    """
    Supports:
        healthy
        0_healthy
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if "_" in x and x.split("_", 1)[0].isdigit():
        return x.split("_", 1)[1]

    return x


def make_onehot_encoder():
    """
    sklearn changed sparse -> sparse_output in newer versions.
    This helper supports both.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def load_dataset():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {INPUT_PATH}\n"
            "Please run:\n"
            "python scripts\\model_stress_index.py"
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("Loaded dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    required_cols = [
        "clip_id",
        "timestamp",
        "player_panel",
        "heart_rate",
        "stress_z",
        "game_phase",
        "survivor_status",
        "role",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_dataset(df):
    df = df.copy()

    df["survivor_status"] = df["survivor_status"].apply(normalize_status_name)

    df["role"] = df["role"].astype(str).str.lower().str.strip()
    df["game_phase"] = df["game_phase"].astype(str).str.lower().str.strip()

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["stress_z"] = pd.to_numeric(df["stress_z"], errors="coerce")

    # Survivor status is meaningful only for survivor panels.
    df = df[df["role"] == "survivor"].copy()

    df = df[df["survivor_status"].isin(ALL_STATES)].copy()

    df = df.dropna(
        subset=[
            "clip_id",
            "player_panel",
            "game_phase",
            "heart_rate",
            "stress_z",
            "survivor_status",
        ]
    ).copy()

    df[TARGET_COL] = df["survivor_status"].apply(
        lambda x: 0 if x in SAFE_STATES else 1
    )

    # Use categorical strings for one-hot encoding.
    df["player_panel"] = df["player_panel"].astype(int).astype(str)
    df["clip_id"] = df["clip_id"].astype(str)

    print("\nAfter preparing prediction dataset:")
    print(f"Rows: {len(df)}")
    print("Target counts:")
    print(df[TARGET_COL].value_counts())
    print("Status counts:")
    print(df["survivor_status"].value_counts())

    return df


def find_valid_group_splits(df):
    """
    Find StratifiedGroupKFold splits where every train and validation fold
    contains both binary classes.

    This avoids invalid folds for ROC-AUC and classification metrics.
    """
    X_dummy = np.zeros(len(df))
    y = df[TARGET_COL].values
    groups = df["clip_id"].values

    for trial in range(MAX_SPLIT_TRIALS):
        seed = SEED + trial

        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        splits = list(sgkf.split(X_dummy, y, groups))

        valid = True
        split_log = []

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            y_train = y[train_idx]
            y_val = y[val_idx]

            train_classes = sorted(np.unique(y_train).tolist())
            val_classes = sorted(np.unique(y_val).tolist())

            row = {
                "fold": fold_id,
                "train_classes": ",".join(map(str, train_classes)),
                "val_classes": ",".join(map(str, val_classes)),
                "train_clips": ",".join(sorted(df.iloc[train_idx]["clip_id"].unique())),
                "val_clips": ",".join(sorted(df.iloc[val_idx]["clip_id"].unique())),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
            }

            split_log.append(row)

            if len(train_classes) < 2 or len(val_classes) < 2:
                valid = False
                break

        if valid:
            print(f"\nFound valid StratifiedGroupKFold split with random_state={seed}")

            split_log_df = pd.DataFrame(split_log)
            split_log_df.to_csv(
                OUTPUT_DIR / "valid_split_log.csv",
                index=False,
                encoding="utf-8"
            )

            return splits, seed

    raise RuntimeError(
        f"Could not find valid {N_SPLITS}-fold splits after {MAX_SPLIT_TRIALS} trials.\n"
        "Try reducing N_SPLITS to 3."
    )


def build_model(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                make_onehot_encoder(),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def get_feature_names(model, numeric_features, categorical_features):
    """
    Get feature names after preprocessing.
    """
    names = []

    if numeric_features:
        names.extend(numeric_features)

    if categorical_features:
        encoder = model.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = encoder.get_feature_names_out(categorical_features)
        names.extend(cat_names.tolist())

    return names


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def evaluate_feature_set(df, splits, feature_set_name, feature_config):
    print("\n" + "=" * 90)
    print(f"Evaluating feature set: {feature_set_name}")
    print("=" * 90)

    out_dir = OUTPUT_DIR / feature_set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = feature_config["numeric"]
    categorical_features = feature_config["categorical"]
    all_features = numeric_features + categorical_features

    fold_rows = []
    all_true = []
    all_pred = []
    all_prob = []
    pred_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        X_train = train_df[all_features]
        y_train = train_df[TARGET_COL].values

        X_val = val_df[all_features]
        y_val = val_df[TARGET_COL].values

        model = build_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_prob)

        row = {
            "feature_set": feature_set_name,
            "fold": fold_id,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_clips": train_df["clip_id"].nunique(),
            "val_clips": val_df["clip_id"].nunique(),
            **metrics,
        }

        fold_rows.append(row)

        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())
        all_prob.extend(y_prob.tolist())

        fold_pred_df = val_df[
            [
                "clip_id",
                "timestamp",
                "player_panel",
                "survivor_status",
                "heart_rate",
                "stress_z",
                "game_phase",
                TARGET_COL,
            ]
        ].copy()

        fold_pred_df["fold"] = fold_id
        fold_pred_df["pred_pressured"] = y_pred
        fold_pred_df["prob_pressured"] = y_prob
        fold_pred_df["correct"] = fold_pred_df[TARGET_COL] == fold_pred_df["pred_pressured"]

        pred_rows.append(fold_pred_df)

        print(
            f"Fold {fold_id}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)

    fold_metrics_path = out_dir / "cv_metrics_by_fold.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8")

    pooled_true = np.array(all_true)
    pooled_pred = np.array(all_pred)
    pooled_prob = np.array(all_prob)

    pooled_metrics = compute_metrics(pooled_true, pooled_pred, pooled_prob)

    summary_rows = []

    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
        summary_rows.append({
            "feature_set": feature_set_name,
            "metric": metric,
            "fold_mean": fold_metrics[metric].mean(),
            "fold_std": fold_metrics[metric].std(ddof=1),
            "pooled": pooled_metrics[metric],
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = out_dir / "cv_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    pred_df = pd.concat(pred_rows, ignore_index=True)

    pred_path = out_dir / "cv_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    # Classification report
    report = classification_report(
        pooled_true,
        pooled_pred,
        target_names=["safe", "pressured"],
        digits=4,
        zero_division=0,
    )

    report_path = out_dir / "pooled_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(pooled_true, pooled_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["safe", "pressured"],
    )
    disp.plot(
        ax=ax,
        values_format="d",
        cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix: {feature_set_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(pooled_true, pooled_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {pooled_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {feature_set_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Train final version of this feature set on all data
    final_model = build_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    final_model.fit(
        df[all_features],
        df[TARGET_COL].values,
    )

    model_path = out_dir / "final_logistic_model.joblib"
    joblib.dump(final_model, model_path)

    # Coefficients
    feature_names = get_feature_names(
        final_model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    coef = final_model.named_steps["clf"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    })

    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    coef_path = out_dir / "logistic_coefficients.csv"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8")

    # Coefficient plot
    top_coef = coef_df.head(20).sort_values("coefficient")

    plt.figure(figsize=(9, 7))
    plt.barh(top_coef["feature"], top_coef["coefficient"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top Logistic Coefficients: {feature_set_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "logistic_coefficients_top20.png", dpi=300)
    plt.close()

    result = {
        "feature_set": feature_set_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "pooled_metrics": pooled_metrics,
        "fold_metrics_mean": {
            metric: float(fold_metrics[metric].mean())
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "fold_metrics_std": {
            metric: float(fold_metrics[metric].std(ddof=1))
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "paths": {
            "fold_metrics": str(fold_metrics_path),
            "summary": str(summary_path),
            "predictions": str(pred_path),
            "classification_report": str(report_path),
            "coefficients": str(coef_path),
            "final_model": str(model_path),
        },
    }

    with open(out_dir / "model_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def compare_results(results):
    rows = []

    for res in results:
        pooled = res["pooled_metrics"]

        row = {
            "feature_set": res["feature_set"],
            "accuracy": pooled["accuracy"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "precision": pooled["precision"],
            "recall": pooled["recall"],
            "f1": pooled["f1"],
            "roc_auc": pooled["roc_auc"],
        }

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")

    # Bar plot: AUC and F1
    plot_df = comparison_df.set_index("feature_set")[["balanced_accuracy", "f1", "roc_auc"]]

    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Pressured-State Prediction Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
    plt.close()

    print("\nModel comparison:")
    print(comparison_df)

    return comparison_df


def save_dataset_summary(df):
    summary = {
        "input_path": str(INPUT_PATH),
        "n_rows": int(len(df)),
        "n_clips": int(df["clip_id"].nunique()),
        "target_counts": df[TARGET_COL].value_counts().to_dict(),
        "status_counts": df["survivor_status"].value_counts().to_dict(),
        "game_phase_counts": df["game_phase"].value_counts().to_dict(),
        "safe_states": SAFE_STATES,
        "pressured_states": PRESSURED_STATES,
        "note": (
            "The target is derived from CNN-predicted survivor_status. "
            "Therefore, this model predicts CNN-derived pressured state, "
            "not manually verified ground truth."
        ),
    }

    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


# ============================================================
# Main
# ============================================================

def main():
    df = load_dataset()
    df = prepare_dataset(df)

    save_dataset_summary(df)

    if df["clip_id"].nunique() < N_SPLITS:
        raise ValueError(
            f"Not enough clips for {N_SPLITS}-fold group CV. "
            f"Found {df['clip_id'].nunique()} clips."
        )

    splits, used_seed = find_valid_group_splits(df)

    print(f"\nUsing group CV seed: {used_seed}")

    all_results = []

    for feature_set_name, feature_config in FEATURE_SETS.items():
        result = evaluate_feature_set(
            df=df,
            splits=splits,
            feature_set_name=feature_set_name,
            feature_config=feature_config,
        )

        all_results.append(result)

    comparison_df = compare_results(all_results)

    final_summary = {
        "used_cv_seed": used_seed,
        "n_splits": N_SPLITS,
        "model_comparison": comparison_df.to_dict(orient="records"),
        "recommendation": (
            "Use physiology_full as the main Model 5 result if it improves "
            "balanced accuracy, F1, or ROC-AUC over context_baseline."
        ),
    }

    with open(OUTPUT_DIR / "model5_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, default=str)

    print("\n=== Model 5 Complete: Pressured-State Prediction ===")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nMain files to check:")
    print(f"1. {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"2. {OUTPUT_DIR / 'physiology_full' / 'cv_metrics_summary.csv'}")
    print(f"3. {OUTPUT_DIR / 'physiology_full' / 'pooled_classification_report.txt'}")
    print(f"4. {OUTPUT_DIR / 'physiology_full' / 'logistic_coefficients.csv'}")
    print(f"5. {OUTPUT_DIR / 'physiology_full' / 'confusion_matrix.png'}")
    print(f"6. {OUTPUT_DIR / 'physiology_full' / 'roc_curve.png'}")


if __name__ == "__main__":
    main()