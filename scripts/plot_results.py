"""Generate report-ready figures from the master dataset and analysis results.

Usage:
    python scripts/plot_results.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "outputs", "tables", "analysis_results.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("husl", 7)


def save_fig(fig, name):
    """Save a figure as PNG."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_hr_timeline(df, clip_id=None):
    """Plot heart rate over time for a clip, colored by player panel."""
    if clip_id:
        data = df[df["clip_id"] == clip_id]
        suffix = f"_{clip_id}"
    else:
        # Use first available clip
        clip_id = df["clip_id"].iloc[0]
        data = df[df["clip_id"] == clip_id]
        suffix = f"_{clip_id}"

    data = data.dropna(subset=["heart_rate"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for pi in sorted(data["player_panel"].unique()):
        panel_data = data[data["player_panel"] == pi]
        ax.plot(panel_data["timestamp"], panel_data["heart_rate"],
                label=f"Player {pi}", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Video Timestamp (seconds)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title(f"Heart Rate Timeline — Clip {clip_id}")
    ax.legend(loc="upper right")
    ax.set_ylim(50, 200)
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"hr_timeline{suffix}")


def plot_hr_by_phase(df):
    """Box plot of HR by game phase."""
    data = df.dropna(subset=["heart_rate", "game_phase"])
    if data.empty or "game_phase" not in data.columns:
        return

    phase_order = ["early", "mid", "late"]
    data = data[data["game_phase"].isin(phase_order)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="game_phase", y="heart_rate",
                order=phase_order, palette="RdYlGn_r", ax=ax)
    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate Distribution by Game Phase")
    save_fig(fig, "hr_by_phase")


def plot_hr_by_status(df):
    """Violin plot of HR by survivor status."""
    if "survivor_status" not in df.columns:
        print("  Skipping hr_by_status (no survivor_status column)")
        return
    data = df.dropna(subset=["heart_rate", "survivor_status"])
    if data.empty:
        return

    status_order = ["healthy", "injured", "downed", "ballooned",
                    "chaired", "eliminated", "escaped"]
    present = [s for s in status_order if s in data["survivor_status"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=data, x="survivor_status", y="heart_rate",
                   order=present, palette="husl", ax=ax, inner="quartile")
    ax.set_xlabel("Survivor Status")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate by Survivor Status")
    plt.xticks(rotation=30, ha="right")
    save_fig(fig, "hr_by_status")


def plot_status_distribution(df):
    """Bar chart of survivor-status proportions."""
    if "survivor_status" not in df.columns:
        print("  Skipping status_distribution (no survivor_status column)")
        return

    counts = df["survivor_status"].value_counts()
    total = counts.sum()
    pcts = (counts / total * 100).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("husl", len(pcts))
    pcts.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Percentage of Observations (%)")
    ax.set_ylabel("Survivor Status")
    ax.set_title("Survivor Status Distribution Across All Clips")
    for i, (v, label) in enumerate(zip(pcts.values, pcts.index)):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    save_fig(fig, "status_distribution")


def plot_hr_per_clip(df):
    """Bar chart of mean HR per clip with error bars."""
    data = df.dropna(subset=["heart_rate"])
    if data.empty:
        return

    clip_stats = data.groupby("clip_id")["heart_rate"].agg(["mean", "std"])
    clip_stats = clip_stats.sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(clip_stats))
    ax.bar(x, clip_stats["mean"], yerr=clip_stats["std"],
           capsize=3, color=sns.color_palette("coolwarm", len(clip_stats)),
           alpha=0.8, edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in clip_stats.index], rotation=45, ha="right")
    ax.set_xlabel("Clip ID")
    ax.set_ylabel("Mean Heart Rate (bpm)")
    ax.set_title("Average Heart Rate per Clip (±1 SD)")
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(fig, "hr_per_clip")


def plot_player_panel_comparison(df):
    """Compare HR distributions across player panels."""
    data = df.dropna(subset=["heart_rate"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="player_panel", y="heart_rate",
                palette="Set2", ax=ax)
    ax.set_xlabel("Player Panel Position")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate by Player Panel Position")
    save_fig(fig, "hr_by_player_panel")


def plot_hr_heatmap(df):
    """Heatmap of mean HR by clip and player panel."""
    data = df.dropna(subset=["heart_rate"])
    if data.empty:
        return

    pivot = data.pivot_table(values="heart_rate", index="player_panel",
                             columns="clip_id", aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=ax, cbar_kws={"label": "Mean HR (bpm)"})
    ax.set_xlabel("Clip ID")
    ax.set_ylabel("Player Panel")
    ax.set_title("Mean Heart Rate Heatmap (Player × Clip)")
    save_fig(fig, "hr_heatmap")


def plot_hr_over_match_time(df):
    """Scatter/line of HR vs match time (seconds elapsed)."""
    if "match_time_sec" not in df.columns:
        return
    data = df.dropna(subset=["heart_rate", "match_time_sec"])
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data["match_time_sec"], data["heart_rate"],
               alpha=0.15, s=5, color="steelblue")

    # Rolling mean trend
    sorted_data = data.sort_values("match_time_sec")
    if len(sorted_data) > 30:
        window = max(10, len(sorted_data) // 30)
        rolling = sorted_data.set_index("match_time_sec")["heart_rate"].rolling(
            window=window, center=True).mean()
        ax.plot(rolling.index, rolling.values, color="red", linewidth=2,
                label=f"Rolling mean (w={window})")
        ax.legend()

    ax.set_xlabel("Match Time (seconds)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate vs. In-Game Match Time")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "hr_vs_match_time")


def main():
    if not os.path.exists(DATASET_PATH):
        sys.exit(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} rows from master dataset")

    print("\nGenerating figures...")

    # Timeline for first few clips
    clips = sorted(df["clip_id"].unique())
    for clip_id in clips[:3]:
        plot_hr_timeline(df, clip_id)

    plot_hr_by_phase(df)
    plot_hr_by_status(df)
    plot_status_distribution(df)
    plot_hr_per_clip(df)
    plot_player_panel_comparison(df)
    plot_hr_heatmap(df)
    plot_hr_over_match_time(df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
