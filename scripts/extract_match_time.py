"""Extract match timer values from gameplay clips using EasyOCR.

Usage:
    python scripts/extract_match_time.py --clip Clips/023.mp4
    python scripts/extract_match_time.py --all
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
from config import TIMER_REGION

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "match_time")


def get_reader():
    """Initialize EasyOCR reader."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def parse_timer(text):
    """Parse MM:SS format, return seconds elapsed."""
    text = text.strip().replace(" ", "")
    match = re.search(r"(\d{1,2})[:\.](\d{2})", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        if minutes < 20 and seconds < 60:
            return minutes * 60 + seconds
    return None


def process_clip(clip_path, reader, fps=1.0):
    """Extract match time from a clip."""
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {clip_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / fps))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}_time.csv")

    rows = []
    frame_idx = 0
    pbar = tqdm(total=int(duration * fps), desc=f"Time {clip_id}", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = round(frame_idx / video_fps, 2)
            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080))

            x1, y1, x2, y2 = TIMER_REGION
            crop = frame[y1:y2, x1:x2]
            scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            try:
                results = reader.readtext(scaled, allowlist="0123456789:")
                raw_text = " ".join(r[1] for r in results)
                match_seconds = parse_timer(raw_text)
            except Exception:
                raw_text = ""
                match_seconds = None

            rows.append({
                "clip_id": clip_id,
                "timestamp": timestamp,
                "raw_text": raw_text,
                "match_time_seconds": match_seconds if match_seconds is not None else "",
            })
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "timestamp", "raw_text", "match_time_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(1 for r in rows if r["match_time_seconds"] != "")
    print(f"  {clip_id}: {valid}/{len(rows)} valid timer readings -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract match timer via EasyOCR")
    parser.add_argument("--clip", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=float, default=0.5)
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Clips", "*.mp4")))
    else:
        parser.error("Specify --clip or --all")

    print("Initializing EasyOCR reader...")
    reader = get_reader()

    for clip in clips:
        process_clip(clip, reader, fps=args.fps)


if __name__ == "__main__":
    main()
