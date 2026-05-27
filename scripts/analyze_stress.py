"""Core analysis: stress, performance, and balance metrics.

Reads the master dataset and computes:
1. Heart-rate timeline statistics
2. Survivor-state distributions
3. Stress by game phase
4. Stress-performance correlations
5. Pressure symmetry indicators
6. Per-player stress profiles

Usage:
    python scripts/analyze_stress.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")


def load_data():
    """Load and validate the master dataset."""
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Dataset not found: {DATASET_PATH}. Run build_dataset.py first.")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Clips: {df['clip_id'].nunique()}")
    return df


def hr_descriptive_stats(df):
    """Compute overall HR descriptive statistics."""
    hr = df["heart_rate"].dropna()
    result = {
        "count": int(len(hr)),
        "mean": round(float(hr.mean()), 1),
        "std": round(float(hr.std()), 1),
        "median": round(float(hr.median()), 1),
        "min": int(hr.min()),
        "max": int(hr.max()),
        "q25": round(float(hr.quantile(0.25)), 1),
        "q75": round(float(hr.quantile(0.75)), 1),
    }
    print("\n=== HR Descriptive Statistics ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def hr_by_phase(df):
    """Compare HR across game phases (early/mid/late)."""
    if "game_phase" not in df.columns:
        print("No game_phase column available.")
        return {}

    hr = df.dropna(subset=["heart_rate", "game_phase"])
    groups = hr.groupby("game_phase")["heart_rate"]

    result = {}
    print("\n=== HR by Game Phase ===")
    for phase, vals in groups:
        result[phase] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "median": round(float(vals.median()), 1),
        }
        print(f"  {phase}: mean={result[phase]['mean']}, "
              f"std={result[phase]['std']}, n={result[phase]['n']}")

    # Kruskal-Wallis test (non-parametric ANOVA)
    phase_groups = [g["heart_rate"].values for _, g in hr.groupby("game_phase")]
    if len(phase_groups) >= 2 and all(len(g) > 1 for g in phase_groups):
        stat, pval = stats.kruskal(*phase_groups)
        result["kruskal_wallis"] = {"statistic": round(stat, 3), "p_value": round(pval, 4)}
        print(f"  Kruskal-Wallis: H={stat:.3f}, p={pval:.4f}")

    return result


def hr_by_status(df):
    """Compare HR across survivor statuses."""
    if "survivor_status" not in df.columns:
        print("No survivor_status column.")
        return {}

    hr = df.dropna(subset=["heart_rate", "survivor_status"])
    groups = hr.groupby("survivor_status")["heart_rate"]

    result = {}
    print("\n=== HR by Survivor Status ===")
    for status, vals in groups:
        result[status] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
        }
        print(f"  {status}: mean={result[status]['mean']}, "
              f"std={result[status]['std']}, n={result[status]['n']}")

    return result


def hr_by_clip(df):
    """Per-clip HR statistics."""
    hr = df.dropna(subset=["heart_rate"])
    groups = hr.groupby("clip_id")["heart_rate"]

    result = {}
    print("\n=== HR by Clip ===")
    for clip_id, vals in groups:
        result[str(clip_id)] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "min": int(vals.min()),
            "max": int(vals.max()),
        }
    print(f"  {len(result)} clips processed")
    return result


def hr_by_player(df):
    """Per-player-panel HR statistics (proxy for player profiles)."""
    hr = df.dropna(subset=["heart_rate"])
    groups = hr.groupby("player_panel")["heart_rate"]

    result = {}
    print("\n=== HR by Player Panel ===")
    for panel, vals in groups:
        high_stress_ratio = float((vals > vals.mean() + vals.std()).mean())
        result[str(panel)] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "max": int(vals.max()),
            "high_stress_ratio": round(high_stress_ratio, 3),
        }
        print(f"  Panel {panel}: mean={result[str(panel)]['mean']}, "
              f"std={result[str(panel)]['std']}, "
              f"high_stress={result[str(panel)]['high_stress_ratio']:.1%}")
    return result


def status_distribution(df):
    """Compute survivor-status time distribution."""
    if "survivor_status" not in df.columns:
        return {}

    counts = df["survivor_status"].value_counts()
    total = counts.sum()
    result = {}
    print("\n=== Survivor Status Distribution ===")
    for status, count in counts.items():
        pct = round(100 * count / total, 1)
        result[status] = {"count": int(count), "percentage": pct}
        print(f"  {status}: {count} ({pct}%)")
    return result


def stress_performance_correlation(df):
    """Correlate HR metrics with available performance indicators."""
    hr = df.dropna(subset=["heart_rate"])
    if hr.empty:
        return {}

    result = {}
    print("\n=== Stress-Performance Correlation ===")

    # Per-clip mean HR as a metric
    clip_stats = hr.groupby("clip_id")["heart_rate"].agg(["mean", "std", "max"])

    # If match_outcome is available
    if "match_outcome" in df.columns:
        outcomes = df.drop_duplicates("clip_id")[["clip_id", "match_outcome"]]
        merged = clip_stats.merge(outcomes.set_index("clip_id"),
                                  left_index=True, right_index=True)
        merged = merged.dropna(subset=["match_outcome"])
        if len(merged) > 2:
            # Binary: encode win=1, loss=0
            if merged["match_outcome"].dtype == object:
                merged["outcome_num"] = (merged["match_outcome"].str.lower() == "win").astype(int)
                if merged["outcome_num"].nunique() > 1:
                    r, p = stats.pointbiserialr(merged["outcome_num"], merged["mean"])
                    result["hr_vs_outcome"] = {
                        "correlation": round(r, 3), "p_value": round(p, 4)}
                    print(f"  HR mean vs outcome: r={r:.3f}, p={p:.4f}")

    print(f"  Clip-level HR stats computed for {len(clip_stats)} clips")
    result["clip_hr_stats"] = clip_stats.to_dict()
    return result


def pressure_symmetry(df):
    """Compare hunter HR vs survivor HR (pressure symmetry analysis)."""
    hr = df.dropna(subset=["heart_rate"])
    if "player_panel" not in hr.columns:
        return {}

    hunter_hr = hr[hr["player_panel"] == 4]["heart_rate"]
    survivor_hr = hr[hr["player_panel"].isin([0, 1, 2, 3])]["heart_rate"]

    result = {}
    print("\n=== Pressure Symmetry (Hunter vs Survivors) ===")

    if len(hunter_hr) > 1 and len(survivor_hr) > 1:
        result["hunter"] = {
            "n": int(len(hunter_hr)),
            "mean": round(float(hunter_hr.mean()), 1),
            "std": round(float(hunter_hr.std()), 1),
            "median": round(float(hunter_hr.median()), 1),
        }
        result["survivors"] = {
            "n": int(len(survivor_hr)),
            "mean": round(float(survivor_hr.mean()), 1),
            "std": round(float(survivor_hr.std()), 1),
            "median": round(float(survivor_hr.median()), 1),
        }
        print(f"  Hunter:    mean={result['hunter']['mean']}, "
              f"std={result['hunter']['std']}, n={result['hunter']['n']}")
        print(f"  Survivors: mean={result['survivors']['mean']}, "
              f"std={result['survivors']['std']}, n={result['survivors']['n']}")

        # Mann-Whitney U test
        stat, pval = stats.mannwhitneyu(hunter_hr, survivor_hr, alternative='two-sided')
        result["mann_whitney_u"] = {"statistic": round(float(stat), 1), "p_value": round(float(pval), 6)}
        print(f"  Mann-Whitney U: U={stat:.1f}, p={pval:.6f}")

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(hunter_hr)-1)*hunter_hr.std()**2 + (len(survivor_hr)-1)*survivor_hr.std()**2)
            / (len(hunter_hr) + len(survivor_hr) - 2)
        )
        if pooled_std > 0:
            d = (hunter_hr.mean() - survivor_hr.mean()) / pooled_std
            result["cohens_d"] = round(float(d), 3)
            print(f"  Cohen's d: {d:.3f}")

    return result


def outcome_stress_comparison(df):
    """Compare HR across match outcomes (hunter_win, draw, survivor_win)."""
    if "match_outcome" not in df.columns:
        return {}

    hr = df.dropna(subset=["heart_rate", "match_outcome"])
    groups = hr.groupby("match_outcome")["heart_rate"]

    result = {}
    print("\n=== HR by Match Outcome ===")
    group_data = {}
    for outcome, vals in groups:
        group_data[outcome] = vals
        result[outcome] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 1),
            "std": round(float(vals.std()), 1),
            "median": round(float(vals.median()), 1),
        }
        print(f"  {outcome}: mean={result[outcome]['mean']}, "
              f"std={result[outcome]['std']}, n={result[outcome]['n']}")

    # Kruskal-Wallis across outcomes
    outcome_groups = [g.values for g in group_data.values()]
    if len(outcome_groups) >= 2 and all(len(g) > 1 for g in outcome_groups):
        stat, pval = stats.kruskal(*outcome_groups)
        result["kruskal_wallis"] = {"statistic": round(float(stat), 3), "p_value": round(float(pval), 6)}
        print(f"  Kruskal-Wallis: H={stat:.3f}, p={pval:.6f}")

    # Pairwise Mann-Whitney U tests
    outcomes_list = list(group_data.keys())
    pairwise = {}
    for i in range(len(outcomes_list)):
        for j in range(i+1, len(outcomes_list)):
            a, b = outcomes_list[i], outcomes_list[j]
            stat, pval = stats.mannwhitneyu(group_data[a], group_data[b], alternative='two-sided')
            pair_key = f"{a}_vs_{b}"
            pairwise[pair_key] = {"U": round(float(stat), 1), "p_value": round(float(pval), 6)}
            print(f"  {pair_key}: U={stat:.1f}, p={pval:.6f}")
    result["pairwise_tests"] = pairwise

    return result


def main():
    df = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "descriptive": hr_descriptive_stats(df),
        "by_phase": hr_by_phase(df),
        "by_status": hr_by_status(df),
        "by_clip": hr_by_clip(df),
        "by_player_panel": hr_by_player(df),
        "status_distribution": status_distribution(df),
        "stress_performance": stress_performance_correlation(df),
        "pressure_symmetry": pressure_symmetry(df),
        "outcome_stress": outcome_stress_comparison(df),
    }

    out_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save key tables as CSV
    if "by_phase" in results and results["by_phase"]:
        phase_df = pd.DataFrame(results["by_phase"]).T
        phase_df.to_csv(os.path.join(OUTPUT_DIR, "hr_by_phase.csv"))

    if "by_status" in results and results["by_status"]:
        status_df = pd.DataFrame(results["by_status"]).T
        status_df.to_csv(os.path.join(OUTPUT_DIR, "hr_by_status.csv"))


if __name__ == "__main__":
    main()
