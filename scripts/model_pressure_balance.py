"""
Model 3: Pressure Balance Analysis

Research question:
    Is physiological pressure symmetric between hunter and survivors?

Input:
    data/processed/master_dataset_with_stress.csv

Main ideas:
    1. Compare raw heart-rate level between hunter and survivors.
    2. Compare normalized stress exposure instead of raw mean stress_z.
       Because stress_z is normalized within each clip-player panel,
       its mean is expected to be around zero for each panel.
       Therefore, meaningful normalized stress metrics are:
           - mean_abs_stress_z
           - std_stress_z
           - p90_stress_z
           - high_stress_z1_ratio
           - high_stress_z15_ratio

Outputs:
    outputs/modeling/pressure_balance/
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "pressure_balance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

ROLE_ORDER = ["survivor", "hunter"]

METRICS_TO_COMPARE = [
    "mean_hr",
    "median_hr",
    "std_hr",
    "mean_abs_stress_z",
    "std_stress_z",
    "p90_stress_z",
    "high_stress_z1_ratio",
    "high_stress_z15_ratio",
]


# ============================================================
# Helper functions
# ============================================================

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
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def add_role_if_missing(df):
    """
    If role column already exists, use it.
    Otherwise map:
        player_panel 0-3 -> survivor
        player_panel 4   -> hunter
    """
    df = df.copy()

    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.lower().str.strip()
        return df

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


def clean_dataset(df):
    df = df.copy()
    df = add_role_if_missing(df)

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["stress_z"] = pd.to_numeric(df["stress_z"], errors="coerce")
    df["player_panel"] = pd.to_numeric(df["player_panel"], errors="coerce")

    df = df.dropna(
        subset=[
            "clip_id",
            "player_panel",
            "heart_rate",
            "stress_z",
            "role",
        ]
    ).copy()

    df = df[df["role"].isin(ROLE_ORDER)].copy()

    # Safety flags
    df["abs_stress_z"] = df["stress_z"].abs()
    df["high_stress_z1"] = (df["stress_z"] >= 1.0).astype(int)
    df["high_stress_z15"] = (df["stress_z"] >= 1.5).astype(int)

    print("\nAfter cleaning:")
    print(f"Rows: {len(df)}")
    print("Role counts:")
    print(df["role"].value_counts())

    return df


def build_panel_series_metrics(df):
    """
    Aggregate repeated frames into one row per clip_id + player_panel.

    This avoids treating every frame as an independent sample.
    """
    series_df = (
        df.groupby(["clip_id", "player_panel", "role"])
        .agg(
            n_frames=("heart_rate", "size"),
            mean_hr=("heart_rate", "mean"),
            median_hr=("heart_rate", "median"),
            std_hr=("heart_rate", "std"),
            min_hr=("heart_rate", "min"),
            max_hr=("heart_rate", "max"),
            mean_stress_z=("stress_z", "mean"),
            median_stress_z=("stress_z", "median"),
            std_stress_z=("stress_z", "std"),
            mean_abs_stress_z=("abs_stress_z", "mean"),
            p90_stress_z=("stress_z", lambda x: np.percentile(x, 90)),
            p95_stress_z=("stress_z", lambda x: np.percentile(x, 95)),
            high_stress_z1_ratio=("high_stress_z1", "mean"),
            high_stress_z15_ratio=("high_stress_z15", "mean"),
        )
        .reset_index()
    )

    # Replace NaN std caused by single-frame groups.
    for col in ["std_hr", "std_stress_z"]:
        series_df[col] = series_df[col].fillna(0.0)

    print("\nPanel-series metrics:")
    print(f"Rows: {len(series_df)}")
    print(series_df["role"].value_counts())

    return series_df


def summarize_by_role(series_df):
    """
    Role summary using panel-series rows.
    """
    rows = []

    for role in ROLE_ORDER:
        sub = series_df[series_df["role"] == role]

        row = {
            "role": role,
            "n_series": len(sub),
            "total_frames": int(sub["n_frames"].sum()),
        }

        for metric in METRICS_TO_COMPARE:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std(ddof=1)
            row[f"{metric}_median"] = sub[metric].median()

        rows.append(row)

    summary = pd.DataFrame(rows)

    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def cohens_d_independent(x, y):
    """
    Cohen's d for independent groups.
    Direction:
        hunter - survivor
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


def cohens_d_paired(delta):
    """
    Cohen's dz for paired differences.
    Direction:
        hunter - survivor
    """
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]

    if len(delta) < 2:
        return np.nan

    sd = np.std(delta, ddof=1)

    if sd == 0:
        return 0.0

    return np.mean(delta) / sd


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=42):
    """
    Simple bootstrap confidence interval for the mean.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)

    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def run_independent_role_tests(series_df):
    """
    Mann-Whitney U tests on panel-series rows.

    Useful but not the main result, because there are more survivor panels
    than hunter panels.
    """
    test_rows = []

    survivor = series_df[series_df["role"] == "survivor"]
    hunter = series_df[series_df["role"] == "hunter"]

    for metric in METRICS_TO_COMPARE:
        x = survivor[metric].dropna().values
        y = hunter[metric].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue

        u, p = stats.mannwhitneyu(
            y,
            x,
            alternative="two-sided"
        )

        d = cohens_d_independent(x, y)

        row = {
            "test_type": "panel_series_mann_whitney",
            "metric": metric,
            "survivor_n": len(x),
            "hunter_n": len(y),
            "survivor_mean": float(np.mean(x)),
            "hunter_mean": float(np.mean(y)),
            "hunter_minus_survivor": float(np.mean(y) - np.mean(x)),
            "mann_whitney_u": float(u),
            "p_value": float(p),
            "cohens_d_hunter_minus_survivor": float(d),
        }

        test_rows.append(row)

    return pd.DataFrame(test_rows)


def build_clip_level_balance(series_df):
    """
    Build one row per clip.

    For each clip:
        hunter_metric = metric for hunter panel
        survivor_metric = mean metric across survivor panels

    This makes hunter vs survivor comparison paired within the same clip.
    This is the strongest pressure-balance analysis.
    """
    clip_rows = []

    for clip_id, g in series_df.groupby("clip_id"):
        hunter = g[g["role"] == "hunter"]
        survivors = g[g["role"] == "survivor"]

        if hunter.empty or survivors.empty:
            continue

        row = {
            "clip_id": clip_id,
            "n_survivor_panels": len(survivors),
            "n_hunter_panels": len(hunter),
        }

        for metric in METRICS_TO_COMPARE:
            hunter_value = hunter[metric].mean()
            survivor_value = survivors[metric].mean()

            row[f"hunter_{metric}"] = hunter_value
            row[f"survivor_{metric}"] = survivor_value
            row[f"delta_{metric}_hunter_minus_survivor"] = hunter_value - survivor_value

        clip_rows.append(row)

    clip_df = pd.DataFrame(clip_rows)

    print("\nClip-level balance rows:", len(clip_df))

    return clip_df


def run_clip_level_paired_tests(clip_df):
    """
    Paired tests across clips.

    Main test:
        For each clip, compare hunter metric to average survivor metric.
    """
    test_rows = []

    for metric in METRICS_TO_COMPARE:
        delta_col = f"delta_{metric}_hunter_minus_survivor"

        if delta_col not in clip_df.columns:
            continue

        delta = clip_df[delta_col].dropna().values

        if len(delta) < 2:
            continue

        try:
            w, p_wilcoxon = stats.wilcoxon(delta)
        except Exception:
            w, p_wilcoxon = np.nan, np.nan

        try:
            t, p_ttest = stats.ttest_1samp(delta, popmean=0)
        except Exception:
            t, p_ttest = np.nan, np.nan

        ci_low, ci_high = bootstrap_ci(delta)

        row = {
            "test_type": "clip_level_paired",
            "metric": metric,
            "n_clips": int(len(delta)),
            "mean_delta_hunter_minus_survivor": float(np.mean(delta)),
            "median_delta_hunter_minus_survivor": float(np.median(delta)),
            "std_delta": float(np.std(delta, ddof=1)),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "wilcoxon_statistic": float(w) if not np.isnan(w) else np.nan,
            "wilcoxon_p_value": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else np.nan,
            "ttest_statistic": float(t) if not np.isnan(t) else np.nan,
            "ttest_p_value": float(p_ttest) if not np.isnan(p_ttest) else np.nan,
            "cohens_dz": float(cohens_d_paired(delta)),
            "num_clips_hunter_higher": int((delta > 0).sum()),
            "num_clips_survivor_higher": int((delta < 0).sum()),
            "num_clips_equal": int((delta == 0).sum()),
        }

        test_rows.append(row)

    return pd.DataFrame(test_rows)


# ============================================================
# Plotting
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


def plot_metric_by_role(series_df, metric, ylabel, title, filename):
    plt.figure(figsize=(7, 5))

    sns.violinplot(
        data=series_df,
        x="role",
        y=metric,
        order=ROLE_ORDER,
        inner="quartile",
        cut=0,
    )

    sns.stripplot(
        data=series_df,
        x="role",
        y=metric,
        order=ROLE_ORDER,
        color="black",
        alpha=0.35,
        size=3,
    )

    plt.xlabel("Role")
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


def plot_clip_delta(clip_df, metric, ylabel, title, filename):
    delta_col = f"delta_{metric}_hunter_minus_survivor"

    if delta_col not in clip_df.columns:
        return

    data = clip_df.sort_values(delta_col).reset_index(drop=True)

    plt.figure(figsize=(10, 5))

    x = np.arange(len(data))

    plt.bar(
        x,
        data[delta_col],
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Clip index sorted by hunter-survivor difference")
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


def plot_hunter_vs_survivor_scatter(clip_df, metric, xlabel, ylabel, title, filename):
    hunter_col = f"hunter_{metric}"
    survivor_col = f"survivor_{metric}"

    if hunter_col not in clip_df.columns or survivor_col not in clip_df.columns:
        return

    plt.figure(figsize=(6, 6))

    plt.scatter(
        clip_df[survivor_col],
        clip_df[hunter_col],
        alpha=0.75,
    )

    min_val = min(clip_df[survivor_col].min(), clip_df[hunter_col].min())
    max_val = max(clip_df[survivor_col].max(), clip_df[hunter_col].max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    save_plot(filename)


# ============================================================
# Main
# ============================================================

def main():
    sns.set_theme(style="whitegrid", font_scale=1.05)

    df = load_dataset()
    df = clean_dataset(df)

    # --------------------------------------------------------
    # Frame-level descriptive summary
    # --------------------------------------------------------
    frame_summary = (
        df.groupby("role")
        .agg(
            n_frames=("heart_rate", "size"),
            mean_hr=("heart_rate", "mean"),
            std_hr=("heart_rate", "std"),
            median_hr=("heart_rate", "median"),
            mean_abs_stress_z=("abs_stress_z", "mean"),
            std_stress_z=("stress_z", "std"),
            high_stress_z1_ratio=("high_stress_z1", "mean"),
            high_stress_z15_ratio=("high_stress_z15", "mean"),
        )
        .reset_index()
    )

    frame_summary_path = OUTPUT_DIR / "role_summary_frame_level.csv"
    frame_summary.to_csv(frame_summary_path, index=False, encoding="utf-8")
    print(f"Saved: {frame_summary_path}")

    # --------------------------------------------------------
    # Panel-series level
    # --------------------------------------------------------
    series_df = build_panel_series_metrics(df)

    series_path = OUTPUT_DIR / "panel_series_metrics.csv"
    series_df.to_csv(series_path, index=False, encoding="utf-8")
    print(f"Saved: {series_path}")

    role_summary = summarize_by_role(series_df)

    role_summary_path = OUTPUT_DIR / "role_summary_panel_series.csv"
    role_summary.to_csv(role_summary_path, index=False, encoding="utf-8")
    print(f"Saved: {role_summary_path}")

    independent_tests = run_independent_role_tests(series_df)

    independent_tests_path = OUTPUT_DIR / "panel_series_role_tests.csv"
    independent_tests.to_csv(independent_tests_path, index=False, encoding="utf-8")
    print(f"Saved: {independent_tests_path}")

    # --------------------------------------------------------
    # Clip-level paired analysis
    # --------------------------------------------------------
    clip_df = build_clip_level_balance(series_df)

    clip_path = OUTPUT_DIR / "clip_level_pressure_balance.csv"
    clip_df.to_csv(clip_path, index=False, encoding="utf-8")
    print(f"Saved: {clip_path}")

    paired_tests = run_clip_level_paired_tests(clip_df)

    paired_tests_path = OUTPUT_DIR / "clip_level_paired_tests.csv"
    paired_tests.to_csv(paired_tests_path, index=False, encoding="utf-8")
    print(f"Saved: {paired_tests_path}")

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------
    summary = {
        "input_path": str(INPUT_PATH),
        "n_frame_rows": int(len(df)),
        "n_panel_series": int(len(series_df)),
        "n_clips_for_paired_analysis": int(len(clip_df)),
        "important_note": (
            "Because stress_z is normalized within each clip-player panel, "
            "mean stress_z is not the main role-comparison metric. "
            "We compare raw HR level and stress exposure metrics such as "
            "mean_abs_stress_z, p90_stress_z, and high_stress ratios."
        ),
        "frame_summary": frame_summary.to_dict(orient="records"),
        "role_summary_panel_series": role_summary.to_dict(orient="records"),
        "panel_series_tests": independent_tests.to_dict(orient="records"),
        "clip_level_paired_tests": paired_tests.to_dict(orient="records"),
    }

    summary_path = OUTPUT_DIR / "pressure_balance_tests.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {summary_path}")

    # A compact summary table for report
    compact_rows = []

    for metric in ["mean_hr", "mean_abs_stress_z", "std_stress_z", "p90_stress_z", "high_stress_z1_ratio"]:
        row = {
            "metric": metric,
        }

        for role in ROLE_ORDER:
            sub = series_df[series_df["role"] == role]
            row[f"{role}_mean"] = sub[metric].mean()

        row["hunter_minus_survivor"] = row["hunter_mean"] - row["survivor_mean"]

        test_row = paired_tests[paired_tests["metric"] == metric]

        if not test_row.empty:
            row["paired_p_value"] = float(test_row.iloc[0]["wilcoxon_p_value"])
            row["cohens_dz"] = float(test_row.iloc[0]["cohens_dz"])
            row["n_clips"] = int(test_row.iloc[0]["n_clips"])
        else:
            row["paired_p_value"] = np.nan
            row["cohens_dz"] = np.nan
            row["n_clips"] = np.nan

        compact_rows.append(row)

    compact_df = pd.DataFrame(compact_rows)

    compact_path = OUTPUT_DIR / "pressure_balance_summary.csv"
    compact_df.to_csv(compact_path, index=False, encoding="utf-8")
    print(f"Saved: {compact_path}")

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_metric_by_role(
        series_df=series_df,
        metric="mean_hr",
        ylabel="Mean Heart Rate (bpm)",
        title="Raw Heart Rate by Role",
        filename="raw_hr_by_role.png",
    )

    plot_metric_by_role(
        series_df=series_df,
        metric="mean_abs_stress_z",
        ylabel="Mean Absolute Stress (|z|)",
        title="Stress Exposure by Role",
        filename="stress_exposure_by_role.png",
    )

    plot_metric_by_role(
        series_df=series_df,
        metric="high_stress_z1_ratio",
        ylabel="Ratio of Frames with stress_z >= 1",
        title="High-Stress Exposure by Role",
        filename="high_stress_ratio_by_role.png",
    )

    plot_clip_delta(
        clip_df=clip_df,
        metric="mean_hr",
        ylabel="Hunter - Survivor Mean HR (bpm)",
        title="Clip-Level Raw HR Difference",
        filename="clip_level_hr_delta.png",
    )

    plot_clip_delta(
        clip_df=clip_df,
        metric="mean_abs_stress_z",
        ylabel="Hunter - Survivor Mean |stress_z|",
        title="Clip-Level Stress Exposure Difference",
        filename="clip_level_abs_stress_delta.png",
    )

    plot_hunter_vs_survivor_scatter(
        clip_df=clip_df,
        metric="mean_hr",
        xlabel="Survivor Mean HR per Clip",
        ylabel="Hunter Mean HR per Clip",
        title="Hunter vs Survivor Mean HR per Clip",
        filename="hunter_vs_survivor_hr_scatter.png",
    )

    print("\n=== Pressure Balance Analysis Complete ===")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    print("\nMain files to check:")
    print(f"1. {compact_path}")
    print(f"2. {role_summary_path}")
    print(f"3. {paired_tests_path}")
    print(f"4. {clip_path}")
    print(f"5. {OUTPUT_DIR / 'raw_hr_by_role.png'}")
    print(f"6. {OUTPUT_DIR / 'stress_exposure_by_role.png'}")


if __name__ == "__main__":
    main()