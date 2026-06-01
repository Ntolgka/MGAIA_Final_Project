"""
Model 4: Stress Index by State and Phase

Research questions:
    1. Which survivor state has the highest normalized stress?
    2. Which game phase has the highest normalized stress?
    3. Which state-phase combination has the highest normalized stress?

Input:
    data/processed/master_dataset_with_stress.csv

Output:
    outputs/modeling/stress_state_phase/
"""

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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "stress_state_phase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
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
# Helper functions
# ============================================================

def normalize_status_name(x):
    """
    Support both:
        healthy
        0_healthy
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
        "stress_z",
        "game_phase",
        "role",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_dataset(df):
    df = df.copy()

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["stress_z"] = pd.to_numeric(df["stress_z"], errors="coerce")

    df["game_phase"] = df["game_phase"].astype(str).str.lower().str.strip()
    df["role"] = df["role"].astype(str).str.lower().str.strip()

    if "survivor_status" in df.columns:
        df["survivor_status"] = df["survivor_status"].apply(normalize_status_name)

    df = df.dropna(
        subset=[
            "clip_id",
            "player_panel",
            "heart_rate",
            "stress_z",
            "game_phase",
            "role",
        ]
    ).copy()

    df = df[df["game_phase"].isin(PHASE_ORDER)].copy()
    df = df[df["role"].isin(ROLE_ORDER)].copy()

    df["high_stress_z1"] = (df["stress_z"] >= 1.0).astype(int)
    df["high_stress_z15"] = (df["stress_z"] >= 1.5).astype(int)

    print("\nAfter basic cleaning:")
    print(f"Rows: {len(df)}")
    print("Role counts:")
    print(df["role"].value_counts())
    print("Phase counts:")
    print(df["game_phase"].value_counts())

    return df


def build_series_level(df, group_cols):
    """
    Aggregate frame-level rows into series-level rows.

    This reduces temporal dependence between consecutive frames.

    Example group_cols:
        ["clip_id", "player_panel", "survivor_status"]
        ["clip_id", "player_panel", "game_phase"]
        ["clip_id", "player_panel", "survivor_status", "game_phase"]
    """
    series_df = (
        df.groupby(group_cols)
        .agg(
            n_frames=("heart_rate", "size"),
            mean_hr=("heart_rate", "mean"),
            median_hr=("heart_rate", "median"),
            mean_stress_z=("stress_z", "mean"),
            median_stress_z=("stress_z", "median"),
            mean_abs_stress_z=("stress_z", lambda x: np.mean(np.abs(x))),
            p90_stress_z=("stress_z", lambda x: np.percentile(x, 90)),
            high_stress_z1_ratio=("high_stress_z1", "mean"),
            high_stress_z15_ratio=("high_stress_z15", "mean"),
        )
        .reset_index()
    )

    return series_df


def summarize_group(series_df, group_col, order=None):
    """
    Summarize one grouping variable at series level.
    """
    summary = (
        series_df.groupby(group_col)
        .agg(
            n_series=("mean_stress_z", "size"),
            total_frames=("n_frames", "sum"),
            mean_hr=("mean_hr", "mean"),
            std_hr=("mean_hr", "std"),
            sem_hr=("mean_hr", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mean_stress_z=("mean_stress_z", "mean"),
            std_stress_z=("mean_stress_z", "std"),
            sem_stress_z=("mean_stress_z", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mean_abs_stress_z=("mean_abs_stress_z", "mean"),
            p90_stress_z=("p90_stress_z", "mean"),
            high_stress_z1_ratio=("high_stress_z1_ratio", "mean"),
            high_stress_z15_ratio=("high_stress_z15_ratio", "mean"),
        )
        .reset_index()
    )

    if order is not None:
        summary[group_col] = pd.Categorical(
            summary[group_col],
            categories=order,
            ordered=True
        )
        summary = summary.sort_values(group_col)

    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def kruskal_test(series_df, group_col, value_col):
    """
    Kruskal-Wallis test across groups.
    """
    groups = []

    for _, sub in series_df.groupby(group_col):
        values = sub[value_col].dropna().values
        if len(values) > 1:
            groups.append(values)

    if len(groups) < 2:
        return None

    h, p = stats.kruskal(*groups)

    return {
        "group_col": group_col,
        "value_col": value_col,
        "statistic": float(h),
        "p_value": float(p),
    }


def holm_adjust(p_values):
    """
    Holm-Bonferroni correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n)

    for rank, idx in enumerate(order):
        adjusted[idx] = min((n - rank) * p_values[idx], 1.0)

    # enforce monotonicity
    sorted_adjusted = adjusted[order]
    for i in range(1, n):
        sorted_adjusted[i] = max(sorted_adjusted[i], sorted_adjusted[i - 1])
    adjusted[order] = sorted_adjusted

    return adjusted


def cohens_d(x, y):
    """
    Cohen's d, direction y - x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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


def pairwise_mannwhitney(series_df, group_col, value_col, order):
    """
    Pairwise Mann-Whitney U tests with Holm correction.
    """
    rows = []

    groups = [g for g in order if g in set(series_df[group_col].dropna().unique())]

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]

            x = series_df[series_df[group_col] == g1][value_col].dropna().values
            y = series_df[series_df[group_col] == g2][value_col].dropna().values

            if len(x) < 2 or len(y) < 2:
                continue

            u, p = stats.mannwhitneyu(x, y, alternative="two-sided")

            rows.append({
                "group_col": group_col,
                "value_col": value_col,
                "group_1": g1,
                "group_2": g2,
                "n_group_1": len(x),
                "n_group_2": len(y),
                "mean_group_1": float(np.mean(x)),
                "mean_group_2": float(np.mean(y)),
                "delta_group2_minus_group1": float(np.mean(y) - np.mean(x)),
                "mannwhitney_u": float(u),
                "p_value": float(p),
                "cohens_d_group2_minus_group1": float(cohens_d(x, y)),
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["p_value_holm"] = holm_adjust(out["p_value"].values)

    return out


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ============================================================
# Plotting
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


def plot_bar_with_sem(summary_df, x_col, y_col, sem_col, order, ylabel, title, filename):
    plot_df = summary_df.copy()
    plot_df = plot_df[plot_df[x_col].isin(order)].copy()

    plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=order, ordered=True)
    plot_df = plot_df.sort_values(x_col)

    x = np.arange(len(plot_df))

    plt.figure(figsize=(8, 5))

    plt.bar(
        x,
        plot_df[y_col],
        yerr=plot_df[sem_col],
        capsize=5,
    )

    plt.xticks(x, plot_df[x_col], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.title(title)
    plt.axhline(0, linestyle="--", linewidth=1)

    save_plot(filename)


def plot_boxplot(series_df, x_col, y_col, order, ylabel, title, filename):
    plot_df = series_df[series_df[x_col].isin(order)].copy()

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        order=order,
    )

    sns.stripplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        order=order,
        color="black",
        alpha=0.35,
        size=3,
    )

    plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.title(title)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right")

    save_plot(filename)


def plot_heatmap(status_phase_summary):
    """
    Heatmap of mean stress_z for status x phase.
    """
    heatmap_df = status_phase_summary.pivot(
        index="survivor_status",
        columns="game_phase",
        values="mean_stress_z"
    )

    # enforce order
    heatmap_df = heatmap_df.reindex(index=STATUS_ORDER, columns=PHASE_ORDER)

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean stress_z"},
    )

    plt.xlabel("Game Phase")
    plt.ylabel("Survivor Status")
    plt.title("Normalized Stress by Status and Game Phase")

    save_plot("stress_status_phase_heatmap.png")


# ============================================================
# Main
# ============================================================

def main():
    sns.set_theme(style="whitegrid", font_scale=1.05)

    df = load_dataset()
    df = clean_dataset(df)

    # --------------------------------------------------------
    # Survivor-only data for status analysis
    # --------------------------------------------------------
    if "survivor_status" not in df.columns:
        raise ValueError("Column survivor_status not found.")

    survivor_df = df[df["role"] == "survivor"].copy()
    survivor_df["survivor_status"] = survivor_df["survivor_status"].apply(normalize_status_name)
    survivor_df = survivor_df[survivor_df["survivor_status"].isin(STATUS_ORDER)].copy()

    print("\nSurvivor-only status data:")
    print(f"Rows: {len(survivor_df)}")
    print(survivor_df["survivor_status"].value_counts())

    # --------------------------------------------------------
    # Status-level analysis
    # one row = clip + panel + status
    # --------------------------------------------------------
    status_series = build_series_level(
        survivor_df,
        ["clip_id", "player_panel", "survivor_status"]
    )

    status_summary = summarize_group(
        status_series,
        "survivor_status",
        STATUS_ORDER
    )

    status_summary_path = OUTPUT_DIR / "stress_by_status.csv"
    status_summary.to_csv(status_summary_path, index=False, encoding="utf-8")
    print(f"Saved: {status_summary_path}")

    status_tests = {
        "kruskal_status_mean_stress_z": kruskal_test(
            status_series,
            "survivor_status",
            "mean_stress_z"
        ),
        "kruskal_status_mean_hr": kruskal_test(
            status_series,
            "survivor_status",
            "mean_hr"
        ),
    }

    status_pairwise = pairwise_mannwhitney(
        status_series,
        group_col="survivor_status",
        value_col="mean_stress_z",
        order=STATUS_ORDER
    )

    status_pairwise_path = OUTPUT_DIR / "status_pairwise_tests.csv"
    status_pairwise.to_csv(status_pairwise_path, index=False, encoding="utf-8")
    print(f"Saved: {status_pairwise_path}")

    # --------------------------------------------------------
    # Phase-level analysis for survivors only
    # one row = clip + panel + phase
    # --------------------------------------------------------
    phase_survivor_series = build_series_level(
        survivor_df,
        ["clip_id", "player_panel", "game_phase"]
    )

    phase_survivor_summary = summarize_group(
        phase_survivor_series,
        "game_phase",
        PHASE_ORDER
    )

    phase_survivor_path = OUTPUT_DIR / "stress_by_phase_survivors.csv"
    phase_survivor_summary.to_csv(phase_survivor_path, index=False, encoding="utf-8")
    print(f"Saved: {phase_survivor_path}")

    phase_tests = {
        "kruskal_phase_survivor_mean_stress_z": kruskal_test(
            phase_survivor_series,
            "game_phase",
            "mean_stress_z"
        ),
        "kruskal_phase_survivor_mean_hr": kruskal_test(
            phase_survivor_series,
            "game_phase",
            "mean_hr"
        ),
    }

    phase_pairwise = pairwise_mannwhitney(
        phase_survivor_series,
        group_col="game_phase",
        value_col="mean_stress_z",
        order=PHASE_ORDER
    )

    phase_pairwise_path = OUTPUT_DIR / "phase_pairwise_tests.csv"
    phase_pairwise.to_csv(phase_pairwise_path, index=False, encoding="utf-8")
    print(f"Saved: {phase_pairwise_path}")

    # --------------------------------------------------------
    # Phase-level analysis for all roles
    # one row = clip + panel + role + phase
    # --------------------------------------------------------
    phase_all_series = build_series_level(
        df,
        ["clip_id", "player_panel", "role", "game_phase"]
    )

    phase_all_summary = summarize_group(
        phase_all_series,
        "game_phase",
        PHASE_ORDER
    )

    phase_all_path = OUTPUT_DIR / "stress_by_phase_all_roles.csv"
    phase_all_summary.to_csv(phase_all_path, index=False, encoding="utf-8")
    print(f"Saved: {phase_all_path}")

    # --------------------------------------------------------
    # Status x phase analysis
    # one row = clip + panel + status + phase
    # --------------------------------------------------------
    status_phase_series = build_series_level(
        survivor_df,
        ["clip_id", "player_panel", "survivor_status", "game_phase"]
    )

    status_phase_summary = (
        status_phase_series
        .groupby(["survivor_status", "game_phase"])
        .agg(
            n_series=("mean_stress_z", "size"),
            total_frames=("n_frames", "sum"),
            mean_hr=("mean_hr", "mean"),
            mean_stress_z=("mean_stress_z", "mean"),
            mean_abs_stress_z=("mean_abs_stress_z", "mean"),
            high_stress_z1_ratio=("high_stress_z1_ratio", "mean"),
        )
        .reset_index()
    )

    status_phase_summary["survivor_status"] = pd.Categorical(
        status_phase_summary["survivor_status"],
        categories=STATUS_ORDER,
        ordered=True
    )

    status_phase_summary["game_phase"] = pd.Categorical(
        status_phase_summary["game_phase"],
        categories=PHASE_ORDER,
        ordered=True
    )

    status_phase_summary = status_phase_summary.sort_values(
        ["survivor_status", "game_phase"]
    )

    numeric_cols = status_phase_summary.select_dtypes(include=[np.number]).columns
    status_phase_summary[numeric_cols] = status_phase_summary[numeric_cols].round(4)

    status_phase_path = OUTPUT_DIR / "stress_by_status_phase.csv"
    status_phase_summary.to_csv(status_phase_path, index=False, encoding="utf-8")
    print(f"Saved: {status_phase_path}")

    # --------------------------------------------------------
    # Identify top stress categories
    # --------------------------------------------------------
    top_status = status_summary.sort_values("mean_stress_z", ascending=False).head(1)
    top_phase_survivor = phase_survivor_summary.sort_values("mean_stress_z", ascending=False).head(1)
    top_status_phase = status_phase_summary.sort_values("mean_stress_z", ascending=False).head(5)

    model_summary = {
        "input_path": str(INPUT_PATH),
        "n_rows_total": int(len(df)),
        "n_rows_survivor_status": int(len(survivor_df)),
        "top_status_by_mean_stress_z": top_status.to_dict(orient="records"),
        "top_phase_survivors_by_mean_stress_z": top_phase_survivor.to_dict(orient="records"),
        "top_status_phase_by_mean_stress_z": top_status_phase.to_dict(orient="records"),
        "status_tests": status_tests,
        "phase_tests": phase_tests,
        "note": (
            "Status analysis uses survivor panels only. "
            "Phase analysis is reported for both survivor-only and all-role data. "
            "Series-level aggregation is used to reduce temporal dependence between consecutive frames."
        ),
    }

    summary_path = OUTPUT_DIR / "model4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_bar_with_sem(
        summary_df=status_summary,
        x_col="survivor_status",
        y_col="mean_stress_z",
        sem_col="sem_stress_z",
        order=STATUS_ORDER,
        ylabel="Mean normalized stress (z-score)",
        title="Normalized Stress by Survivor Status",
        filename="stress_by_status_bar.png",
    )

    plot_boxplot(
        series_df=status_series,
        x_col="survivor_status",
        y_col="mean_stress_z",
        order=STATUS_ORDER,
        ylabel="Mean normalized stress (z-score)",
        title="Normalized Stress by Survivor Status",
        filename="stress_by_status_boxplot.png",
    )

    plot_bar_with_sem(
        summary_df=phase_survivor_summary,
        x_col="game_phase",
        y_col="mean_stress_z",
        sem_col="sem_stress_z",
        order=PHASE_ORDER,
        ylabel="Mean normalized stress (z-score)",
        title="Normalized Stress by Game Phase: Survivors",
        filename="stress_by_phase_bar.png",
    )

    plot_boxplot(
        series_df=phase_survivor_series,
        x_col="game_phase",
        y_col="mean_stress_z",
        order=PHASE_ORDER,
        ylabel="Mean normalized stress (z-score)",
        title="Normalized Stress by Game Phase: Survivors",
        filename="stress_by_phase_boxplot.png",
    )

    plot_heatmap(status_phase_summary)

    print("\n=== Model 4 Complete: Stress Index by State and Phase ===")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nMain files to check:")
    print(f"1. {status_summary_path}")
    print(f"2. {phase_survivor_path}")
    print(f"3. {status_phase_path}")
    print(f"4. {status_pairwise_path}")
    print(f"5. {phase_pairwise_path}")
    print(f"6. {summary_path}")
    print(f"7. {OUTPUT_DIR / 'stress_by_status_bar.png'}")
    print(f"8. {OUTPUT_DIR / 'stress_status_phase_heatmap.png'}")


if __name__ == "__main__":
    main()