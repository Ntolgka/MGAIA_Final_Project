"""Extract frames and UI crops from gameplay clips.

For each clip in Clips/, extracts targeted UI crops at configurable FPS:
  - Heart-rate digit regions (5 player panels, right sidebar)
  - Survivor portrait regions (5 panels, for CNN inference)
  - Match timer region (bottom-left)
  - Cipher count region (top-center)
  - Optionally full frames

Usage:
    python scripts/extract_frames.py --clip Clips/023.mp4 --fps 1
    python scripts/extract_frames.py --all --fps 1
"""

import argparse
import os
import sys
import glob

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (HEART_RATE_REGIONS, PORTRAIT_REGIONS,
                    TIMER_REGION, CIPHER_REGION)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_clip(clip_path, fps, output_base, save_full_frames=False):
    """Extract frames and UI crops from a single clip."""
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    dirs = {
        "portraits": os.path.join(output_base, "portraits", clip_id),
        "heart_rate": os.path.join(output_base, "heart_rate_crops", clip_id),
        "timer": os.path.join(output_base, "timer_crops", clip_id),
        "cipher": os.path.join(output_base, "cipher_crops", clip_id),
    }
    if save_full_frames:
        dirs["frames"] = os.path.join(output_base, "frames", clip_id)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {clip_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / fps))

    print(f"Clip {clip_id}: {duration:.1f}s, {video_fps:.0f}fps, "
          f"extracting every {frame_interval} frames ({fps} fps output)")

    frame_idx = 0
    extracted = 0
    pbar = tqdm(total=int(duration * fps), desc=clip_id, unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            ts_str = f"{timestamp:07.2f}"

            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080))

            if save_full_frames:
                cv2.imwrite(
                    os.path.join(dirs["frames"], f"frame_{ts_str}.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Survivor portraits
            for pi, (x1, y1, x2, y2) in PORTRAIT_REGIONS.items():
                crop = frame[y1:y2, x1:x2]
                cv2.imwrite(
                    os.path.join(dirs["portraits"], f"sec_{ts_str}_p{pi}.jpg"),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Heart-rate digits
            for pi, (x1, y1, x2, y2) in HEART_RATE_REGIONS.items():
                crop = frame[y1:y2, x1:x2]
                cv2.imwrite(
                    os.path.join(dirs["heart_rate"], f"hr_{ts_str}_p{pi}.jpg"),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Timer
            x1, y1, x2, y2 = TIMER_REGION
            cv2.imwrite(
                os.path.join(dirs["timer"], f"timer_{ts_str}.jpg"),
                frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Cipher count
            x1, y1, x2, y2 = CIPHER_REGION
            cv2.imwrite(
                os.path.join(dirs["cipher"], f"cipher_{ts_str}.jpg"),
                frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted += 1
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()
    print(f"  Extracted {extracted} frames from {clip_id}")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames and UI crops from clips")
    parser.add_argument("--clip", type=str, help="Path to a single clip")
    parser.add_argument("--all", action="store_true",
                        help="Process all clips in Clips/")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Extraction rate (default: 1 fps)")
    parser.add_argument("--full-frames", action="store_true",
                        help="Also save full frames")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "raw"),
                        help="Base output directory")
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Clips", "*.mp4")))
    else:
        parser.error("Specify --clip or --all")

    print(f"Processing {len(clips)} clip(s) at {args.fps} fps")
    total = 0
    for clip in clips:
        n = extract_clip(clip, args.fps, args.output,
                         save_full_frames=args.full_frames)
        total += n
    print(f"\nDone. Total frames extracted: {total}")


if __name__ == "__main__":
    main()
