"""Run CNN inference on survivor portrait crops from gameplay clips.

Loads the trained CNN model, extracts portrait crops from each frame,
predicts survivor status, and applies temporal smoothing.

Usage:
    python scripts/run_cnn_inference.py --clip Clips/023.mp4
    python scripts/run_cnn_inference.py --all
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (PORTRAIT_REGIONS, CNN_IMG_HEIGHT, CNN_IMG_WIDTH,
                    CNN_CLASS_NAMES, STATUS_LABELS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "cnn", "outputs", "best_avatar_model.keras")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "survivor_status")


def load_model():
    """Load the trained CNN model."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run cnn/train_cnn.py first.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def predict_portrait(model, crop):
    """Predict survivor status from a portrait crop."""
    resized = cv2.resize(crop, (CNN_IMG_WIDTH, CNN_IMG_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return CNN_CLASS_NAMES[pred_idx], confidence, probs


def smooth_predictions(predictions, window=5):
    """Apply majority-vote smoothing over a sliding window."""
    if len(predictions) < window:
        return predictions

    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window // 2)
        end = min(len(predictions), i + window // 2 + 1)
        window_preds = predictions[start:end]
        # Majority vote
        from collections import Counter
        counts = Counter(window_preds)
        smoothed.append(counts.most_common(1)[0][0])
    return smoothed


def process_clip(clip_path, model, fps=1.0, panels=None, smooth_window=5):
    """Run CNN inference on a clip's portrait crops."""
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    if panels is None:
        panels = list(PORTRAIT_REGIONS.keys())

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {clip_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / fps))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect raw predictions per panel
    panel_predictions = {pi: [] for pi in panels}
    timestamps = []

    frame_idx = 0
    pbar = tqdm(total=int(duration * fps), desc=f"CNN {clip_id}", unit="f")

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
                x1, y1, x2, y2 = PORTRAIT_REGIONS[pi]
                crop = frame[y1:y2, x1:x2]
                status, conf, _ = predict_portrait(model, crop)
                panel_predictions[pi].append((status, conf))

            pbar.update(1)
        frame_idx += 1

    cap.release()
    pbar.close()

    # Apply smoothing
    for pi in panels:
        raw_labels = [p[0] for p in panel_predictions[pi]]
        raw_confs = [p[1] for p in panel_predictions[pi]]
        smoothed = smooth_predictions(raw_labels, window=smooth_window)
        panel_predictions[pi] = list(zip(smoothed, raw_confs))

    # Write CSV
    out_path = os.path.join(OUTPUT_DIR, f"{clip_id}_status.csv")
    rows = []
    for i, ts in enumerate(timestamps):
        for pi in panels:
            status, conf = panel_predictions[pi][i]
            rows.append({
                "clip_id": clip_id,
                "timestamp": ts,
                "player_panel": pi,
                "raw_status": status,
                "status": STATUS_LABELS.get(status, status),
                "confidence": round(conf, 3),
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "timestamp", "player_panel",
            "raw_status", "status", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {clip_id}: {len(timestamps)} frames x {len(panels)} panels -> {out_path}")

    # Summary stats
    summary_path = os.path.join(OUTPUT_DIR, f"{clip_id}_summary.json")
    summary = {"clip_id": clip_id, "total_frames": len(timestamps), "panels": {}}
    for pi in panels:
        from collections import Counter
        statuses = [p[0] for p in panel_predictions[pi]]
        counts = Counter(statuses)
        summary["panels"][str(pi)] = {
            STATUS_LABELS.get(s, s): c for s, c in counts.items()
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="CNN survivor status inference")
    parser.add_argument("--clip", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--panels", type=int, nargs="+", default=None)
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window")
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Clips", "*.mp4")))
    else:
        parser.error("Specify --clip or --all")

    print("Loading CNN model...")
    model = load_model()
    print("Model loaded.\n")

    for clip in clips:
        process_clip(clip, model, fps=args.fps, panels=args.panels,
                     smooth_window=args.smooth)


if __name__ == "__main__":
    main()
