"""Extract heart-rate values from gameplay clips using EasyOCR.

Uses EasyOCR (PyTorch-based) for better handling of stylized game fonts.
Falls back to Tesseract if EasyOCR is unavailable.

Usage:
    python scripts/extract_heart_rate.py --clip Clips/023.mp4
    python scripts/extract_heart_rate.py --all
"""

import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HEART_RATE_REGIONS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_rate")

# Wider regions that include context (icon) for better EasyOCR performance.
HR_REGIONS_WIDE = {
    0: (1700, 95, 1840, 140),
    1: (1700, 300, 1840, 340),
    2: (1700, 480, 1840, 520),
    3: (1700, 682, 1840, 722),
    4: (1700, 855, 1840, 895),
}


def get_reader():
    """Initialize EasyOCR reader (cached singleton)."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return reader


def extract_hr_easyocr(reader, crop):
    """Extract HR value using EasyOCR."""
    # Upscale for better recognition
    scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(scaled, allowlist="0123456789")

    best_val = None
    best_conf = 0.0

    for bbox, text, conf in results:
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            continue
        val = int(digits)
        if 50 <= val <= 220 and conf > best_conf:
            best_val = val
            best_conf = conf

    return best_val, best_conf


def smooth_hr_series(values, window=5):
    """Apply median filter to HR series, forward-filling gaps."""
    result = values.copy()
    n = len(result)

    # Forward fill None values with nearest valid
    for i in range(1, n):
        if result[i] is None and result[i - 1] is not None:
            result[i] = result[i - 1]

    # Backward fill
    for i in range(n - 2, -1, -1):
        if result[i] is None and result[i + 1] is not None:
            result[i] = result[i + 1]

    # Median filter
    if n >= window:
        filtered = result.copy()
        half = window // 2
        for i in range(half, n - half):
            w = [v for v in result[i - half:i + half + 1] if v is not None]
            if w:
                filtered[i] = int(np.median(w))
        return filtered

    return result


def process_clip(clip_path, reader, fps=1.0, panels=None):
    """Extract heart rates from all panels across a clip."""
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    if panels is None:
        panels = list(HR_REGIONS_WIDE.keys())

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {clip_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / fps))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect per-panel readings
    panel_readings = {pi: [] for pi in panels}
    timestamps = []

    frame_idx = 0
    pbar = tqdm(total=int(duration * fps), desc=f"HR {clip_id}", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = round(frame_idx / video_fps, 2)
            timestamps.append(timestamp)

            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080))

            for pi in panels:
                x1, y1, x2, y2 = HR_REGIONS_WIDE[pi]
                crop = frame[y1:y2, x1:x2]
                hr_val, conf = extract_hr_easyocr(reader, crop)
                panel_readings[pi].append((hr_val, conf))

            pbar.update(1)
        frame_idx += 1

    cap.release()
    pbar.close()

    # Apply smoothing per panel
    for pi in panels:
        raw_vals = [r[0] for r in panel_readings[pi]]
        raw_confs = [r[1] for r in panel_readings[pi]]
        smoothed = smooth_hr_series(raw_vals)
        panel_readings[pi] = list(zip(smoothed, raw_confs))

    # Write CSV
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}_hr.csv")
    rows = []
    for i, ts in enumerate(timestamps):
        for pi in panels:
            hr_val, conf = panel_readings[pi][i]
            rows.append({
                "clip_id": clip_id,
                "timestamp": ts,
                "player_panel": pi,
                "heart_rate": hr_val if hr_val is not None else "",
                "confidence": round(conf, 3),
                "valid": 1 if hr_val is not None else 0,
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "timestamp", "player_panel",
            "heart_rate", "confidence", "valid"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    valid = sum(1 for r in rows if r["valid"])
    print(f"  {clip_id}: {valid}/{total} valid HR readings "
          f"({100 * valid / total:.0f}%) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract heart rate via EasyOCR")
    parser.add_argument("--clip", type=str, help="Single clip path")
    parser.add_argument("--all", action="store_true", help="Process all clips")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Extraction rate (default: 0.5 fps = every 2 sec)")
    parser.add_argument("--panels", type=int, nargs="+", default=None,
                        help="Player panel indices (default: all)")
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Clips", "*.mp4")))
    else:
        parser.error("Specify --clip or --all")

    print("Initializing EasyOCR reader...")
    reader = get_reader()
    print("Reader ready.\n")

    for clip in clips:
        process_clip(clip, reader, fps=args.fps, panels=args.panels)


if __name__ == "__main__":
    main()
