# Heartbeats and Confrontation — IVL 2025 Stress Analysis

## Project Overview

Analysis of stress, performance, and balance in professional asymmetrical play using gameplay clips from the Identity V League (IVL) 2025 Autumn season.

**Course**: Modern Game AI Algorithms, Leiden University, 2025–2026  
**Authors**: Naixin Wang (s4657128), Shuai Cao (s4851978),
Zhou Tong (s4857456), and Ntolgka Nalmpant (s4953932)

## Quick Start

```bash
# 1. Create virtual environment (must use Python 3.11 — TensorFlow requires it)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline on a single clip (test)
python3 scripts/run_pipeline.py --clip Clips/023.mp4

# 4. Run the full pipeline on all 87 clips (~3-4 hours)
python3.11 scripts/run_pipeline.py --all --fps 0.5

# 5. Build the master dataset
python3.11 scripts/build_dataset.py

# 6. Run baseline analysis + plotting
python3.11 scripts/analyze_stress.py
python3.11 scripts/plot_results.py

# 7. Run advanced modeling
python3.11 scripts/model_hr_explanation.py
python3.11 scripts/model_pressure_balance.py
python3.11 scripts/model_pressured_state_prediction.py
python3.11 scripts/model_state_progression.py
python3.11 scripts/model_stress_index.py
python3.11 scripts/model_stress_state_phase.py
```

## Project Structure

```bash
Final_Project/
├── Clips/                  # IVL match clips
├── cnn/
│   ├── datasets/           # 1339 labeled images (7 classes)
│   ├── outputs/            # Trained model + evaluation artifacts
│   └── train_cnn.py        # CNN training script
├── data/
│   ├── annotations/        # clip_metadata.csv (all clips, all fields populated)
│   ├── processed/          # Extracted CSVs (HR, time, status) + master_dataset.csv
│   └── raw/                # Frame extracts and crop debug images
├── outputs/
│   ├── figures/            # Plots 
│   ├── modeling/           # Output directory for advanced statistical models
│   └── tables/             # Analysis results (JSON, CSV)
├── scripts/
│   ├── config.py           # UI region coordinates + CNN config
│   ├── extract_heart_rate.py  # EasyOCR-based HR extraction
│   ├── extract_match_time.py  # EasyOCR-based timer extraction
│   ├── run_cnn_inference.py   # 7-class survivor status classification
│   ├── extract_frames.py
│   ├── build_dataset.py    # Merge all data into master_dataset.csv
│   ├── analyze_stress.py   # Statistical analysis + hypothesis testing
│   ├── plot_results.py     # Generate all figures
│   ├── run_pipeline.py     # Master orchestrator
│   ├── model_hr_explanation.py # OLS Regression
│   ├── model_pressure_balance.py # Hunter vs Survivor balance
│   ├── model_pressured_state_prediction.py # Logistic Regression predictions
│   ├── model_state_progression.py # Status progression (healthy -> injured -> downed -> chaired)
│   ├── model_stress_index.py # Normalized stress calculations
│   └── model_stress_state_phase.py # Phase and state interaction models
└── requirements.txt
```

## Pipeline Stages

1. **HR Extraction** — EasyOCR reads heart-rate digits from 5 right-sidebar panels
2. **Timer Extraction** — EasyOCR reads match timer (MM:SS) from bottom-left
3. **CNN Inference** — 7-class status classifier (92.4% accuracy) on survivor portraits
4. **Dataset Build** — Merge all extracted data + clip metadata into `master_dataset.csv`
5. **Analysis** — Descriptive stats, Kruskal-Wallis phase test, stress-status correlation
6. **Visualization** — 12 figure types

## Key Results

| Metric | Value |
|--------|-------|
| Total data points | 67,856 |
| Clips analyzed | 87 |
| Mean HR | 121.6 bpm (σ=35.7) |
| HR range | 50–219 bpm |
| Phase effect | Kruskal-Wallis H=96.7, p<0.001 |
| Highest status HR | Chaired: 132.2 bpm |
| Lowest status HR | Injured: 117.5 bpm |
| Hunter vs Survivor | Hunter (+7.84 bpm higher average) |

## CNN Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| healthy | 0.96 | 1.00 | 0.98 |
| injured | 1.00 | 1.00 | 1.00 |
| downed | 0.92 | 1.00 | 0.96 |
| ballooned | 1.00 | 0.90 | 0.95 |
| chaired | 1.00 | 0.96 | 0.98 |
| eliminated | 0.81 | 0.94 | 0.87 |
| escaped | 0.87 | 0.48 | 0.62 |
| **Overall** | **0.93** | **0.92** | **0.92** |

## Dependencies

- **Python 3.11** (required — TensorFlow does not support Python 3.14)
- TensorFlow 2.21
- EasyOCR 1.7 (with PyTorch backend)
- OpenCV, NumPy, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn
- statsmodels (for OLS regression analysis)
