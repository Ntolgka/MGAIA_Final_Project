"""Master pipeline: run all extraction and analysis steps.

Usage:
    python scripts/run_pipeline.py --clip Clips/023.mp4        # Single clip test
    python scripts/run_pipeline.py --all                        # All clips
    python scripts/run_pipeline.py --all --skip-extraction      # Analysis only
"""

import argparse
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(name, func, *args, **kwargs):
    """Run a pipeline step with timing."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ {name} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_extract_hr(clips, reader, fps):
    """Extract heart rates from clips."""
    from extract_heart_rate import process_clip
    for clip in clips:
        process_clip(clip, reader, fps=fps)


def step_extract_time(clips, reader, fps):
    """Extract match time from clips."""
    from extract_match_time import process_clip
    for clip in clips:
        process_clip(clip, reader, fps=fps)


def step_cnn_inference(clips, fps):
    """Run CNN inference on clips."""
    from run_cnn_inference import load_model, process_clip
    model = load_model()
    for clip in clips:
        process_clip(clip, model, fps=fps)


def step_build_dataset():
    """Build master dataset from extracted data."""
    from build_dataset import build_dataset
    build_dataset()


def step_analyze():
    """Run stress analysis."""
    from analyze_stress import main as analyze_main
    analyze_main()


def step_plot():
    """Generate figures."""
    from plot_results import main as plot_main
    plot_main()


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument("--clip", type=str, help="Single clip path")
    parser.add_argument("--all", action="store_true", help="Process all clips")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Extraction rate (default: 0.5 fps)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, run analysis only")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip CNN inference")
    args = parser.parse_args()

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted(glob.glob(os.path.join(PROJECT_ROOT, "Clips", "*.mp4")))
    else:
        parser.error("Specify --clip or --all")

    print(f"Pipeline starting: {len(clips)} clip(s), {args.fps} fps")
    total_start = time.time()

    if not args.skip_extraction:
        # Initialize EasyOCR reader (shared across steps)
        print("\nInitializing EasyOCR reader...")
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("Reader ready.")

        # Phase 1: Heart-rate extraction
        run_step("Heart-Rate Extraction", step_extract_hr, clips, reader, args.fps)

        # Phase 2: Match-time extraction
        run_step("Match-Time Extraction", step_extract_time, clips, reader, args.fps)

        # Phase 3: CNN inference
        if not args.skip_cnn:
            run_step("CNN Survivor-Status Inference", step_cnn_inference, clips, args.fps)

    # Phase 4: Build dataset
    run_step("Build Master Dataset", step_build_dataset)

    # Phase 5: Analysis
    run_step("Stress Analysis", step_analyze)

    # Phase 6: Figures
    run_step("Generate Figures", step_plot)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
