"""
Model 2: State Progression Analysis

Research question:
    Does heart rate / normalized stress increase as survivor state becomes worse?

Progression:
    healthy -> injured -> downed -> chaired

Input:
    data/processed/master_dataset_with_stress.csv

Outputs:
    outputs/modeling/state_progression/
        state_progression_frame_level.csv
        state_progression_series_level.csv
        adjacent_pair_tests.csv
        state_progression_tests.json
        state_progression_hr.png
        state_progression_stress_z.png
        state_progression_hr_boxplot.png
        state_progression_stress_z_boxplot.png
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset_with_stress.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "state_progression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

STATE_ORDER = [
    "healthy",
    "injured",
    "downed",
    "chaired",
]

STATE_SCORE = {
    "healthy": 0,
    "injured": 1,
    "downed": 2,
    "chaired": 3,
}

ADJACENT_PAIRS = [
    ("healthy", "injured"),
    ("injured", "downed"),
    ("downed", "chaired"),
]


# ============================================================
# Helper functions
# ============================================================

def normalize_status_name(x):
    """
    Make status names robust.

    Handles:
        healthy
        0_healthy
        1_injured
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if "_" in x and x.split("_", 1)[0].isdigit():
        return x.split("_", 1)[1]

    return x


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
        "survivor_status",
        "stress_z",
        "role",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_dataset(df):
    df = df.copy()

    df["survivor_status"] = df["survivor_status"].apply(normalize_status_name)

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["stress_z"] = pd.to_numeric(df["stress_z"], errors="coerce")

    # Only survivor panels are meaningful for survivor status progression.
    df = df[df["role"] == "survivor"].copy()

    df = df[df["survivor_status"].isin(STATE_ORDER)].copy()

    df = df.dropna(
        subset=[
            "clip_id",
            "player_panel",
            "survivor_status",
            "heart_rate",
            "stress_z",
        ]
    )

    df["state_score"] = df["survivor_status"].map(STATE_SCORE)

    df["survivor_status"] = pd.Categorical(
        df["survivor_status"],
        categories=STATE_ORDER,
        ordered=True,
    )

    print("\nAfter cleaning:")
    print(f"Rows: {len(df)}")
    print("State counts:")
    print(df["survivor_status"].value_counts().sort_index())

    return df


def summarize_frame_level(df):
    """
    Frame-level summary.
    This is descriptive, but not ideal for statistical tests because
    consecutive frames from the same player and clip are correlated.
    """
    summary = (
        df.groupby("survivor_status", observed=False)
        .agg(
            n_frames=("heart_rate", "size"),
            mean_hr=("heart_rate", "mean"),
            std_hr=("heart_rate", "std"),
            median_hr=("heart_rate", "median"),
            mean_stress_z=("stress_z", "mean"),
            std_stress_z=("stress_z", "std"),
            median_stress_z=("stress_z", "median"),
        )
        .reset_index()
    )

    numeric_cols = [
        "mean_hr",
        "std_hr",
        "median_hr",
        "mean_stress_z",
        "std_stress_z",
        "median_stress_z",
    ]

    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def build_series_level_dataset(df):
    """
    Aggregate repeated frames within the same clip-panel-status.

    This reduces the problem that many consecutive frames are not independent.
    Each row becomes:
        one clip + one player panel + one state
    """
    series_df = (
        df.groupby(
            ["clip_id", "player_panel", "survivor_status"],
            observed=False
        )
        .agg(
            n_frames=("heart_rate", "size"),
            mean_hr=("heart_rate", "mean"),
            mean_stress_z=("stress_z", "mean"),
            median_hr=("heart_rate", "median"),
            median_stress_z=("stress_z", "median"),
        )
        .reset_index()
    )

    # Drop empty category combinations created by categorical grouping.
    series_df = series_df[series_df["n_frames"] > 0].copy()

    series_df["state_score"] = series_df["survivor_status"].map(STATE_SCORE)

    print("\nSeries-level dataset:")
    print(f"Rows: {len(series_df)}")
    print("State counts:")
    print(series_df["survivor_status"].value_counts().sort_index())

    return series_df


def summarize_series_level(series_df):
    """
    Main summary for reporting.
    """
    summary = (
        series_df.groupby("survivor_status", observed=False)
        .agg(
            n_series=("mean_hr", "size"),
            total_frames=("n_frames", "sum"),
            mean_hr=("mean_hr", "mean"),
            std_hr=("mean_hr", "std"),
            sem_hr=("mean_hr", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mean_stress_z=("mean_stress_z", "mean"),
            std_stress_z=("mean_stress_z", "std"),
            sem_stress_z=("mean_stress_z", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
        )
        .reset_index()
    )

    numeric_cols = [
        "mean_hr",
        "std_hr",
        "sem_hr",
        "mean_stress_z",
        "std_stress_z",
        "sem_stress_z",
    ]

    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def cohens_d(x, y):
    """
    Cohen's d for two independent groups.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx = len(x)
    ny = len(y)

    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(y) - np.mean(x)) / pooled_std


def run_tests(series_df):
    """
    Statistical tests on series-level data.

    Tests:
    1. Kruskal-Wallis across all progression states.
    2. Spearman correlation between state_score and HR / stress_z.
    3. Adjacent Mann-Whitney U tests.
    """
    results = {}

    # ----------------------------------------
    # Kruskal-Wallis across all states
    # ----------------------------------------
    groups_hr = []
    groups_stress = []

    for state in STATE_ORDER:
        sub = series_df[series_df["survivor_status"] == state]
        if len(sub) > 1:
            groups_hr.append(sub["mean_hr"].values)
            groups_stress.append(sub["mean_stress_z"].values)

    if len(groups_hr) >= 2:
        h_hr, p_hr = stats.kruskal(*groups_hr)
        results["kruskal_hr"] = {
            "statistic": float(h_hr),
            "p_value": float(p_hr),
        }

    if len(groups_stress) >= 2:
        h_stress, p_stress = stats.kruskal(*groups_stress)
        results["kruskal_stress_z"] = {
            "statistic": float(h_stress),
            "p_value": float(p_stress),
        }

    # ----------------------------------------
    # Spearman trend test
    # ----------------------------------------
    rho_hr, p_spear_hr = stats.spearmanr(
        series_df["state_score"],
        series_df["mean_hr"]
    )

    rho_stress, p_spear_stress = stats.spearmanr(
        series_df["state_score"],
        series_df["mean_stress_z"]
    )

    results["spearman_state_score_hr"] = {
        "rho": float(rho_hr),
        "p_value": float(p_spear_hr),
    }

    results["spearman_state_score_stress_z"] = {
        "rho": float(rho_stress),
        "p_value": float(p_spear_stress),
    }

    # ----------------------------------------
    # Adjacent pair tests
    # ----------------------------------------
    pair_rows = []

    for src, dst in ADJACENT_PAIRS:
        a = series_df[series_df["survivor_status"] == src]
        b = series_df[series_df["survivor_status"] == dst]

        row = {
            "comparison": f"{src}_vs_{dst}",
            "src": src,
            "dst": dst,
            "n_src": len(a),
            "n_dst": len(b),
        }

        if len(a) > 1 and len(b) > 1:
            u_hr, p_hr = stats.mannwhitneyu(
                a["mean_hr"],
                b["mean_hr"],
                alternative="two-sided"
            )

            u_stress, p_stress = stats.mannwhitneyu(
                a["mean_stress_z"],
                b["mean_stress_z"],
                alternative="two-sided"
            )

            row.update({
                "src_mean_hr": a["mean_hr"].mean(),
                "dst_mean_hr": b["mean_hr"].mean(),
                "delta_hr_dst_minus_src": b["mean_hr"].mean() - a["mean_hr"].mean(),
                "p_hr": p_hr,
                "cohens_d_hr_dst_minus_src": cohens_d(a["mean_hr"], b["mean_hr"]),

                "src_mean_stress_z": a["mean_stress_z"].mean(),
                "dst_mean_stress_z": b["mean_stress_z"].mean(),
                "delta_stress_z_dst_minus_src": b["mean_stress_z"].mean() - a["mean_stress_z"].mean(),
                "p_stress_z": p_stress,
                "cohens_d_stress_z_dst_minus_src": cohens_d(a["mean_stress_z"], b["mean_stress_z"]),
            })

        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    return results, pair_df


# ============================================================
# Plotting
# ============================================================

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


def plot_progression_line(summary_df, value_col, sem_col, ylabel, title, filename):
    x = np.arange(len(summary_df))

    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x,
        summary_df[value_col],
        yerr=summary_df[sem_col],
        marker="o",
        capsize=5,
        linewidth=2,
    )

    plt.xticks(x, summary_df["survivor_status"])
    plt.xlabel("Survivor State Progression")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    save_plot(OUTPUT_DIR / filename)


def plot_boxplot(series_df, value_col, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=series_df,
        x="survivor_status",
        y=value_col,
        order=STATE_ORDER,
    )

    sns.stripplot(
        data=series_df,
        x="survivor_status",
        y=value_col,
        order=STATE_ORDER,
        color="black",
        alpha=0.35,
        size=3,
    )

    plt.xlabel("Survivor State")
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(OUTPUT_DIR / filename)


# ============================================================
# Main
# ============================================================

def main():
    sns.set_theme(style="whitegrid", font_scale=1.05)

    df = load_dataset()
    df = clean_dataset(df)

    if df.empty:
        raise ValueError("No usable survivor progression data found.")

    # --------------------------------------------------------
    # Frame-level descriptive summary
    # --------------------------------------------------------
    frame_summary = summarize_frame_level(df)

    frame_summary_path = OUTPUT_DIR / "state_progression_frame_level.csv"
    frame_summary.to_csv(frame_summary_path, index=False, encoding="utf-8")
    print(f"Saved: {frame_summary_path}")

    # --------------------------------------------------------
    # Series-level dataset and summary
    # --------------------------------------------------------
    series_df = build_series_level_dataset(df)

    series_data_path = OUTPUT_DIR / "state_progression_series_data.csv"
    series_df.to_csv(series_data_path, index=False, encoding="utf-8")
    print(f"Saved: {series_data_path}")

    series_summary = summarize_series_level(series_df)

    series_summary_path = OUTPUT_DIR / "state_progression_series_level.csv"
    series_summary.to_csv(series_summary_path, index=False, encoding="utf-8")
    print(f"Saved: {series_summary_path}")

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------
    tests, pair_df = run_tests(series_df)

    tests_path = OUTPUT_DIR / "state_progression_tests.json"
    with open(tests_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)
    print(f"Saved: {tests_path}")

    pair_path = OUTPUT_DIR / "adjacent_pair_tests.csv"
    pair_df.to_csv(pair_path, index=False, encoding="utf-8")
    print(f"Saved: {pair_path}")

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_progression_line(
        summary_df=series_summary,
        value_col="mean_hr",
        sem_col="sem_hr",
        ylabel="Mean Heart Rate (bpm)",
        title="Heart Rate Across Survivor State Progression",
        filename="state_progression_hr.png",
    )

    plot_progression_line(
        summary_df=series_summary,
        value_col="mean_stress_z",
        sem_col="sem_stress_z",
        ylabel="Mean Normalized Stress (z-score)",
        title="Normalized Stress Across Survivor State Progression",
        filename="state_progression_stress_z.png",
    )

    plot_boxplot(
        series_df=series_df,
        value_col="mean_hr",
        ylabel="Mean Heart Rate (bpm)",
        title="Heart Rate by Survivor State",
        filename="state_progression_hr_boxplot.png",
    )

    plot_boxplot(
        series_df=series_df,
        value_col="mean_stress_z",
        ylabel="Mean Normalized Stress (z-score)",
        title="Normalized Stress by Survivor State",
        filename="state_progression_stress_z_boxplot.png",
    )

    print("\n=== State Progression Analysis Complete ===")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nMain files to check:")
    print(f"1. {series_summary_path}")
    print(f"2. {pair_path}")
    print(f"3. {tests_path}")
    print(f"4. {OUTPUT_DIR / 'state_progression_hr.png'}")
    print(f"5. {OUTPUT_DIR / 'state_progression_stress_z.png'}")


if __name__ == "__main__":
    main()