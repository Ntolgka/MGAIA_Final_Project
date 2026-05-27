"""Merge extracted data into a single structured dataset.

Combines heart-rate CSVs, match-time CSVs, survivor-status CSVs,
and manual metadata into one master dataset.

Usage:
    python scripts/build_dataset.py
"""

import os
import sys
import glob

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HR_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_rate")
TIME_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "match_time")
STATUS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "survivor_status")
META_PATH = os.path.join(PROJECT_ROOT, "data", "annotations", "clip_metadata.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")


def load_csvs(directory, pattern="*.csv"):
    """Load and concatenate all CSVs in a directory."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def build_dataset():
    """Build the master dataset by merging all extracted data."""
    print("Loading heart-rate data...")
    hr_df = load_csvs(HR_DIR, "*_hr.csv")

    print("Loading match-time data...")
    time_df = load_csvs(TIME_DIR, "*_time.csv")

    print("Loading survivor-status data...")
    status_df = load_csvs(STATUS_DIR, "*_status.csv")

    print("Loading clip metadata...")
    if os.path.exists(META_PATH):
        meta_df = pd.read_csv(META_PATH)
    else:
        meta_df = pd.DataFrame()

    if hr_df.empty and status_df.empty:
        print("No extracted data found. Run extraction scripts first.")
        return

    # Start with HR data as the primary time series
    if not hr_df.empty:
        # Filter to valid readings
        hr_valid = hr_df[hr_df["valid"] == 1].copy()
        hr_valid["heart_rate"] = hr_valid["heart_rate"].astype(int)
        hr_valid = hr_valid.drop(columns=["valid"])

        master = hr_valid.copy()
    else:
        master = pd.DataFrame()

    # Merge match time (join on clip_id + timestamp)
    if not time_df.empty and not master.empty:
        time_df = time_df.rename(columns={"match_time_seconds": "match_time_sec"})
        time_cols = ["clip_id", "timestamp", "match_time_sec"]
        time_df = time_df[[c for c in time_cols if c in time_df.columns]]
        master = master.merge(time_df, on=["clip_id", "timestamp"], how="left")

    # Merge survivor status (join on clip_id + timestamp + player_panel)
    if not status_df.empty and not master.empty:
        status_cols = ["clip_id", "timestamp", "player_panel", "status", "confidence"]
        status_merge = status_df[[c for c in status_cols if c in status_df.columns]].copy()
        status_merge = status_merge.rename(columns={
            "confidence": "status_confidence",
            "status": "survivor_status",
        })
        master = master.merge(
            status_merge,
            on=["clip_id", "timestamp", "player_panel"],
            how="left",
        )

    # Merge clip metadata
    if not meta_df.empty and not master.empty:
        master = master.merge(meta_df, on="clip_id", how="left")

    # Add derived columns
    if "match_time_sec" in master.columns:
        # Game phase: early (0-120s), mid (120-300s), late (300+s)
        def classify_phase(sec):
            if pd.isna(sec):
                return "unknown"
            sec = float(sec)
            if sec < 120:
                return "early"
            elif sec < 300:
                return "mid"
            else:
                return "late"
        master["game_phase"] = master["match_time_sec"].apply(classify_phase)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMaster dataset saved to {OUTPUT_PATH}")
    print(f"  Shape: {master.shape}")
    print(f"  Columns: {list(master.columns)}")
    if not master.empty:
        print(f"  Clips: {master['clip_id'].nunique()}")
        print(f"  Total rows: {len(master)}")


if __name__ == "__main__":
    build_dataset()
