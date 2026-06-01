"""
Modeling Direction 1: Stress Index

This script builds a normalized stress index from master_dataset.csv.

Input:
    data/processed/master_dataset.csv

Output:
    data/processed/master_dataset_with_stress.csv
    outputs/modeling/stress_index/*.csv
    outputs/modeling/stress_index/*.png
    outputs/modeling/stress_index/stress_index_summary.json

Definition:
    stress_z = (heart_rate - mean_hr_of_same_clip_and_panel) / std_hr_of_same_clip_and_panel

The normalization is done within each (clip_id, player_panel) group.
This reduces baseline differences between players, panels, and clips.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset_with_stress.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "stress_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Orders
# ============================================================

STATUS_ORDER = [
    "healthy",
    "injured",
    "downed",
    "ballooned",
    "chaired",
    "eliminated",
    "escaped",
]

PHASE_ORDER = [
    "early",
    "mid",
    "late",
]

ROLE_ORDER = [
    "survivor",
    "hunter",
]


# ============================================================
# Utility functions
# ============================================================

def load_master_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        sys.exit(
            f"Cannot find {INPUT_PATH}\n"
            "Please run the pipeline first, for example:\n"
            "python scripts\\run_pipeline.py --all --fps 0.5"
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded master dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    required_cols = ["clip_id", "timestamp", "player_panel", "heart_rate"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            "Stress index requires clip_id, timestamp, player_panel, and heart_rate."
        )

    return df


def add_role_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    In the current project setup:
        player_panel 0-3 = survivors
        player_panel 4 = hunter
    """
    df = df.copy()

    def map_role(panel):
        try:
            panel = int(panel)
        except Exception:
            return "unknown"

        if panel in [0, 1, 2, 3]:
            return "survivor"
        if panel == 4:
            return "hunter"
        return "unknown"

    df["role"] = df["player_panel"].apply(map_role)
    return df


def compute_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized stress index within each clip-player series.

    stress_z = (heart_rate - group_mean) / group_std

    Groups with std = 0 are protected by replacing std with NaN first.
    Their stress_z will become 0 after filling, because no variation exists
    inside that group.
    """
    df = df.copy()

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["heart_rate", "clip_id", "player_panel"]).copy()
    after = len(df)

    print(f"Rows before HR filtering: {before}")
    print(f"Rows after HR filtering:  {after}")

    group_cols = ["clip_id", "player_panel"]

    baseline = (
        df.groupby(group_cols)["heart_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "baseline_hr_mean",
            "std": "baseline_hr_std",
            "count": "baseline_hr_count",
        })
    )

    df = df.merge(baseline, on=group_cols, how="left")

    df["baseline_hr_std_safe"] = df["baseline_hr_std"].replace(0, np.nan)

    df["stress_z"] = (
        (df["heart_rate"] - df["baseline_hr_mean"])
        / df["baseline_hr_std_safe"]
    )

    # If a group has no variance or only one row, stress_z is undefined.
    # For analysis, set it to 0 because it has no within-group deviation.
    df["stress_z"] = df["stress_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Helpful binary flags for descriptive analysis.
    df["high_stress_z1"] = (df["stress_z"] >= 1.0).astype(int)
    df["high_stress_z15"] = (df["stress_z"] >= 1.5).astype(int)

    df = add_role_column(df)

    return df


def summarize_group(df: pd.DataFrame, group_col: str, order=None) -> pd.DataFrame:
    """
    Summarize stress_z and raw HR by a categorical column.
    """
    data = df.dropna(subset=["stress_z", "heart_rate", group_col]).copy()

    if order is not None:
        data = data[data[group_col].isin(order)]

    summary = (
        data.groupby(group_col)
        .agg(
            n=("stress_z", "size"),
            mean_stress_z=("stress_z", "mean"),
            std_stress_z=("stress_z", "std"),
            median_stress_z=("stress_z", "median"),
            mean_hr=("heart_rate", "mean"),
            std_hr=("heart_rate", "std"),
            high_stress_z1_ratio=("high_stress_z1", "mean"),
            high_stress_z15_ratio=("high_stress_z15", "mean"),
        )
        .reset_index()
    )

    if order is not None:
        summary[group_col] = pd.Categorical(summary[group_col], categories=order, ordered=True)
        summary = summary.sort_values(group_col)

    numeric_cols = [
        "mean_stress_z",
        "std_stress_z",
        "median_stress_z",
        "mean_hr",
        "std_hr",
        "high_stress_z1_ratio",
        "high_stress_z15_ratio",
    ]

    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(4)

    return summary


def kruskal_test(df: pd.DataFrame, group_col: str, value_col: str = "stress_z"):
    """
    Non-parametric group comparison.
    """
    data = df.dropna(subset=[group_col, value_col]).copy()

    groups = []
    labels = []

    for label, sub in data.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) > 1:
            groups.append(vals)
            labels.append(label)

    if len(groups) < 2:
        return None

    stat, p_value = stats.kruskal(*groups)

    return {
        "group_col": group_col,
        "value_col": value_col,
        "groups": labels,
        "statistic": float(stat),
        "p_value": float(p_value),
    }


def mann_whitney_role_test(df: pd.DataFrame):
    """
    Compare stress_z between hunter and survivor.
    """
    data = df.dropna(subset=["role", "stress_z"]).copy()
    data = data[data["role"].isin(["hunter", "survivor"])]

    hunter = data[data["role"] == "hunter"]["stress_z"].values
    survivor = data[data["role"] == "survivor"]["stress_z"].values

    if len(hunter) < 2 or len(survivor) < 2:
        return None

    stat, p_value = stats.mannwhitneyu(hunter, survivor, alternative="two-sided")

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(hunter) - 1) * np.var(hunter, ddof=1)
         + (len(survivor) - 1) * np.var(survivor, ddof=1))
        / (len(hunter) + len(survivor) - 2)
    )

    if pooled_std == 0:
        cohen_d = 0.0
    else:
        cohen_d = (np.mean(hunter) - np.mean(survivor)) / pooled_std

    return {
        "hunter_n": int(len(hunter)),
        "survivor_n": int(len(survivor)),
        "hunter_mean_stress_z": float(np.mean(hunter)),
        "survivor_mean_stress_z": float(np.mean(survivor)),
        "mann_whitney_u": float(stat),
        "p_value": float(p_value),
        "cohens_d_hunter_minus_survivor": float(cohen_d),
    }


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ============================================================
# Plotting
# ============================================================

def save_fig(fig, filename: str):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_stress_by_status(df: pd.DataFrame):
    if "survivor_status" not in df.columns:
        print("No survivor_status column. Skipping stress_by_status plot.")
        return

    data = df.dropna(subset=["survivor_status", "stress_z"]).copy()
    data = data[data["survivor_status"].isin(STATUS_ORDER)]

    if data.empty:
        print("No status data for stress plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=data,
        x="survivor_status",
        y="stress_z",
        order=[s for s in STATUS_ORDER if s in data["survivor_status"].unique()],
        inner="quartile",
        ax=ax,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Survivor Status")
    ax.set_ylabel("Normalized Stress (z-score)")
    ax.set_title("Normalized Stress by Survivor Status")
    plt.xticks(rotation=30, ha="right")
    save_fig(fig, "stress_by_status.png")


def plot_stress_by_phase(df: pd.DataFrame):
    if "game_phase" not in df.columns:
        print("No game_phase column. Skipping stress_by_phase plot.")
        return

    data = df.dropna(subset=["game_phase", "stress_z"]).copy()
    data = data[data["game_phase"].isin(PHASE_ORDER)]

    if data.empty:
        print("No game phase data for stress plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=data,
        x="game_phase",
        y="stress_z",
        order=[p for p in PHASE_ORDER if p in data["game_phase"].unique()],
        ax=ax,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Normalized Stress (z-score)")
    ax.set_title("Normalized Stress by Game Phase")
    save_fig(fig, "stress_by_phase.png")


def plot_stress_by_role(df: pd.DataFrame):
    data = df.dropna(subset=["role", "stress_z"]).copy()
    data = data[data["role"].isin(ROLE_ORDER)]

    if data.empty:
        print("No role data for stress plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=data,
        x="role",
        y="stress_z",
        order=[r for r in ROLE_ORDER if r in data["role"].unique()],
        inner="quartile",
        ax=ax,
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Role")
    ax.set_ylabel("Normalized Stress (z-score)")
    ax.set_title("Normalized Stress: Hunter vs Survivors")
    save_fig(fig, "stress_by_role.png")


def plot_example_stress_timeline(df: pd.DataFrame):
    """
    Plot one example clip with both raw HR and stress_z.
    """
    data = df.dropna(subset=["clip_id", "timestamp", "player_panel", "heart_rate", "stress_z"]).copy()

    if data.empty:
        print("No data for stress timeline plot.")
        return

    # Choose the clip with the most observations.
    clip_id = data["clip_id"].value_counts().index[0]
    clip_data = data[data["clip_id"] == clip_id].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    for panel in sorted(clip_data["player_panel"].unique()):
        panel_data = clip_data[clip_data["player_panel"] == panel].sort_values("timestamp")
        ax.plot(
            panel_data["timestamp"],
            panel_data["stress_z"],
            label=f"Panel {panel}",
            linewidth=1.4,
            alpha=0.85,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Video Timestamp (seconds)")
    ax.set_ylabel("Normalized Stress (z-score)")
    ax.set_title(f"Stress Timeline Example — Clip {clip_id}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_fig(fig, "stress_timeline_example.png")


# ============================================================
# Main
# ============================================================

def main():
    sns.set_theme(style="whitegrid", font_scale=1.05)

    df = load_master_dataset()
    df_stress = compute_stress_index(df)

    # Save enhanced master dataset
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_stress.to_csv(OUTPUT_DATA_PATH, index=False, encoding="utf-8")
    print(f"\nSaved enhanced dataset with stress_z:")
    print(OUTPUT_DATA_PATH)

    # Save baseline by clip and panel
    baseline_cols = [
        "clip_id",
        "player_panel",
        "baseline_hr_mean",
        "baseline_hr_std",
        "baseline_hr_count",
    ]

    baseline_df = (
        df_stress[baseline_cols]
        .drop_duplicates()
        .sort_values(["clip_id", "player_panel"])
    )
    baseline_df.to_csv(OUTPUT_DIR / "stress_by_clip_panel.csv", index=False, encoding="utf-8")

    # Summaries
    summaries = {}

    if "survivor_status" in df_stress.columns:
        stress_by_status = summarize_group(df_stress, "survivor_status", STATUS_ORDER)
        stress_by_status.to_csv(OUTPUT_DIR / "stress_by_status.csv", index=False, encoding="utf-8")
        summaries["by_status"] = stress_by_status.to_dict(orient="records")

    if "game_phase" in df_stress.columns:
        stress_by_phase = summarize_group(df_stress, "game_phase", PHASE_ORDER)
        stress_by_phase.to_csv(OUTPUT_DIR / "stress_by_phase.csv", index=False, encoding="utf-8")
        summaries["by_phase"] = stress_by_phase.to_dict(orient="records")

    stress_by_role = summarize_group(df_stress, "role", ROLE_ORDER)
    stress_by_role.to_csv(OUTPUT_DIR / "stress_by_role.csv", index=False, encoding="utf-8")
    summaries["by_role"] = stress_by_role.to_dict(orient="records")

    # Statistical tests
    tests = {}

    if "survivor_status" in df_stress.columns:
        tests["kruskal_status_stress_z"] = kruskal_test(df_stress, "survivor_status", "stress_z")

    if "game_phase" in df_stress.columns:
        tests["kruskal_phase_stress_z"] = kruskal_test(df_stress, "game_phase", "stress_z")

    tests["hunter_vs_survivor_stress_z"] = mann_whitney_role_test(df_stress)

    # Overall summary
    overall = {
        "input_path": str(INPUT_PATH),
        "output_dataset_path": str(OUTPUT_DATA_PATH),
        "num_rows": int(len(df_stress)),
        "num_clips": int(df_stress["clip_id"].nunique()),
        "num_panels": int(df_stress["player_panel"].nunique()),
        "stress_z_mean": float(df_stress["stress_z"].mean()),
        "stress_z_std": float(df_stress["stress_z"].std()),
        "high_stress_z1_ratio": float(df_stress["high_stress_z1"].mean()),
        "high_stress_z15_ratio": float(df_stress["high_stress_z15"].mean()),
        "summaries": summaries,
        "tests": tests,
    }

    save_json(overall, OUTPUT_DIR / "stress_index_summary.json")

    # Figures
    plot_stress_by_status(df_stress)
    plot_stress_by_phase(df_stress)
    plot_stress_by_role(df_stress)
    plot_example_stress_timeline(df_stress)

    print("\n=== Stress Index Analysis Complete ===")
    print(f"Enhanced dataset: {OUTPUT_DATA_PATH}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nMain files to check:")
    print(f"1. {OUTPUT_DATA_PATH}")
    print(f"2. {OUTPUT_DIR / 'stress_by_status.csv'}")
    print(f"3. {OUTPUT_DIR / 'stress_by_phase.csv'}")
    print(f"4. {OUTPUT_DIR / 'stress_by_role.csv'}")
    print(f"5. {OUTPUT_DIR / 'stress_index_summary.json'}")


if __name__ == "__main__":
    main()