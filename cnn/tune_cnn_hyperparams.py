import os
import re
import json
import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# Path config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs_hyperparameter_tuning"
PIPELINE_MODEL_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_KERAS = PIPELINE_MODEL_DIR / "best_avatar_model.keras"
FINAL_MODEL_H5 = PIPELINE_MODEL_DIR / "best_avatar_model.h5"
CLASS_MAP_PATH = PIPELINE_MODEL_DIR / "class_mapping.json"


# ============================================================
# Basic config
# ============================================================

IMG_HEIGHT = 92
IMG_WIDTH = 98

BATCH_SIZE = 8

# 调参阶段建议先用 50 或 60，节省时间。
# 如果时间充足，可以改成 80。
MAX_EPOCHS_TUNING = 50

# final model 会使用最佳 config 的平均 best_epoch 训练。
MIN_FINAL_EPOCHS = 5

# 如果数据量不大，可以先用 3-fold 调参；
# 最终报告如果想更严谨，可以改回 5。
N_SPLITS = 3

SEED = 42
MAX_SPLIT_TRIALS = 500

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# sequential: 分阶段调参，推荐
# full_grid: 跑完整 4*3*3*4 组合，非常慢
TUNING_STRATEGY = "sequential"


CLASS_NAMES = [
    "0_healthy",
    "1_injured",
    "2_downed",
    "3_ballooned",
    "4_chaired",
    "5_eliminated",
    "6_escaped",
]


# ============================================================
# Search space
# ============================================================

LEARNING_RATES = [0.001, 0.0005, 0.0003, 0.0001]

AUGMENTATION_CONFIGS = {
    "none": {
        "rotation_range": 0,
        "width_shift_range": 0.0,
        "height_shift_range": 0.0,
        "zoom_range": 0.0,
        "brightness_range": None,
    },
    "mild": {
        "rotation_range": 5,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "zoom_range": 0.08,
        "brightness_range": [0.85, 1.15],
    },
    "medium": {
        "rotation_range": 8,
        "width_shift_range": 0.08,
        "height_shift_range": 0.08,
        "zoom_range": 0.12,
        "brightness_range": [0.8, 1.2],
    },
}

MODEL_SIZE_CONFIGS = {
    "small": [8, 16, 32, 64],
    "current": [16, 32, 64, 128],
    "compact": [16, 32, 64],
}

LABEL_SMOOTHING_VALUES = [0.0, 0.03, 0.05, 0.1]


DEFAULT_CONFIG = {
    "learning_rate": 0.0005,
    "augmentation": "mild",
    "model_size": "current",
    "label_smoothing": 0.05,
}


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Filename parser
# ============================================================

def parse_filename(filename: str):
    """
    Expected:
        sec_0095_p3_v5.jpg

    Meaning:
        second = 95
        player = p3
        video group = v05
    """
    stem = Path(filename).stem
    pattern = r"sec_(\d+)_p(\d+)_v(\d+)"
    match = re.match(pattern, stem, re.IGNORECASE)

    if match is None:
        raise ValueError(
            f"Cannot parse filename: {filename}\n"
            f"Expected format: sec_0095_p3_v5.jpg"
        )

    second = int(match.group(1))
    player = f"p{int(match.group(2))}"
    group = f"v{int(match.group(3)):02d}"

    return second, player, group


# ============================================================
# Dataset loading
# ============================================================

def load_dataset():
    rows = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Cannot find dataset folder: {DATASET_DIR}")

    for label in CLASS_NAMES:
        class_dir = DATASET_DIR / label

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            second, player, group = parse_filename(img_path.name)

            rows.append({
                "filepath": str(img_path.resolve()),
                "filename": img_path.name,
                "label": label,
                "second": second,
                "player": player,
                "group": group,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No images found.")

    return df


def print_dataset_summary(df):
    print("\n=== Dataset Summary ===")
    print(f"Images: {len(df)}")
    print(f"Video groups: {df['group'].nunique()}")
    print(f"Players: {sorted(df['player'].unique())}")

    print("\nLabel counts:")
    print(df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0))

    print("\nGroup counts:")
    print(df["group"].value_counts().sort_index())


def save_group_distribution(df):
    rows = []

    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]

        row = {
            "group": group,
            "total": len(sub),
        }

        for label in CLASS_NAMES:
            row[label] = int((sub["label"] == label).sum())

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "group_distribution.csv",
        index=False,
        encoding="utf-8"
    )


def save_label_group_coverage(df):
    rows = []

    print("\n=== Label Coverage Across Video Groups ===")
    for label in CLASS_NAMES:
        sub = df[df["label"] == label]
        groups = sorted(sub["group"].unique())

        print(f"{label}: {len(groups)} groups -> {groups}")

        rows.append({
            "label": label,
            "num_images": len(sub),
            "num_groups": len(groups),
            "groups": ",".join(groups),
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "label_group_coverage.csv",
        index=False,
        encoding="utf-8"
    )


# ============================================================
# Valid Group K-Fold split
# ============================================================

def validate_one_split(df, train_idx, val_idx):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    all_classes = set(CLASS_NAMES)

    train_groups = set(train_df["group"].unique())
    val_groups = set(val_df["group"].unique())
    overlap_groups = train_groups.intersection(val_groups)

    train_labels = set(train_df["label"].unique())
    val_labels = set(val_df["label"].unique())

    missing_train_labels = all_classes - train_labels
    missing_val_labels = all_classes - val_labels

    is_valid = True
    reasons = []

    if overlap_groups:
        is_valid = False
        reasons.append(f"group_overlap={sorted(overlap_groups)}")

    if missing_train_labels:
        is_valid = False
        reasons.append(f"missing_train_labels={sorted(missing_train_labels)}")

    return {
        "is_valid": is_valid,
        "reasons": "; ".join(reasons),
        "missing_train_labels": sorted(missing_train_labels),
        "missing_val_labels": sorted(missing_val_labels),
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "train_labels": sorted(train_labels),
        "val_labels": sorted(val_labels),
    }


def find_valid_group_kfold_splits(df):
    X = df["filepath"].values
    y = df["label"].values
    groups = df["group"].values

    invalid_logs = []

    for trial in range(MAX_SPLIT_TRIALS):
        seed = SEED + trial

        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed
        )

        splits = list(sgkf.split(X, y, groups))

        all_valid = True
        trial_logs = []

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            check = validate_one_split(df, train_idx, val_idx)

            row = {
                "trial": trial,
                "seed": seed,
                "fold": fold_id,
                "is_valid": check["is_valid"],
                "reasons": check["reasons"],
                "missing_train_labels": ",".join(check["missing_train_labels"]),
                "missing_val_labels": ",".join(check["missing_val_labels"]),
                "train_groups": ",".join(check["train_groups"]),
                "val_groups": ",".join(check["val_groups"]),
            }

            trial_logs.append(row)

            if not check["is_valid"]:
                all_valid = False

        if all_valid:
            print(f"\nFound valid StratifiedGroupKFold split with random_state={seed}")
            split_log = pd.DataFrame(trial_logs)
            split_log.to_csv(
                OUTPUT_DIR / "valid_split_log.csv",
                index=False,
                encoding="utf-8"
            )
            return splits, seed

        invalid_logs.extend(trial_logs)

    pd.DataFrame(invalid_logs).to_csv(
        OUTPUT_DIR / "invalid_split_trials.csv",
        index=False,
        encoding="utf-8"
    )

    raise RuntimeError(
        f"Could not find valid {N_SPLITS}-fold split after {MAX_SPLIT_TRIALS} trials.\n"
        f"Try reducing N_SPLITS or collecting more samples for rare classes."
    )


# ============================================================
# Data generators
# ============================================================

def make_datagen(augmentation_name, train=True):
    if not train:
        return ImageDataGenerator(rescale=1.0 / 255)

    aug = AUGMENTATION_CONFIGS[augmentation_name]

    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        zoom_range=aug["zoom_range"],
        brightness_range=aug["brightness_range"],
        fill_mode="nearest"
    )


def make_generator(df, augmentation_name, train=True):
    datagen = make_datagen(augmentation_name, train=train)

    gen = datagen.flow_from_dataframe(
        df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=train,
        seed=SEED
    )

    return gen


def compute_class_weights(gen):
    classes = np.unique(gen.classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=gen.classes
    )

    return {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }


# ============================================================
# Model
# ============================================================

def conv_block(x, filters, dropout_rate):
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    return x


def get_dropout_schedule(num_blocks):
    if num_blocks == 3:
        return [0.10, 0.15, 0.20]
    if num_blocks == 4:
        return [0.10, 0.15, 0.20, 0.25]

    return [0.15 for _ in range(num_blocks)]


def build_model(model_size_name):
    filters_list = MODEL_SIZE_CONFIGS[model_size_name]
    dropout_schedule = get_dropout_schedule(len(filters_list))

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs

    for filters, dropout_rate in zip(filters_list, dropout_schedule):
        x = conv_block(x, filters, dropout_rate)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    return models.Model(inputs, outputs)


def compile_model(model, learning_rate, label_smoothing):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, val_df, augmentation_name):
    val_gen = make_generator(val_df, augmentation_name, train=False)
    val_gen.reset()

    probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_gen.classes

    all_labels = list(range(len(CLASS_NAMES)))
    present_labels = sorted(np.unique(y_true))

    acc = accuracy_score(y_true, y_pred)

    macro_f1_present = f1_score(
        y_true,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0
    )

    macro_f1_all = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )

    return acc, macro_f1_present, macro_f1_all, y_true, y_pred


# ============================================================
# One config cross-validation
# ============================================================

def config_to_name(config):
    lr = str(config["learning_rate"]).replace(".", "p")
    ls = str(config["label_smoothing"]).replace(".", "p")
    return (
        f"lr_{lr}"
        f"__aug_{config['augmentation']}"
        f"__model_{config['model_size']}"
        f"__ls_{ls}"
    )


def train_one_fold(config, fold_id, train_df, val_df, config_dir):
    tf.keras.backend.clear_session()

    random.seed(SEED + fold_id)
    np.random.seed(SEED + fold_id)
    tf.random.set_seed(SEED + fold_id)

    train_gen = make_generator(
        train_df,
        augmentation_name=config["augmentation"],
        train=True
    )

    val_gen = make_generator(
        val_df,
        augmentation_name=config["augmentation"],
        train=False
    )

    class_weight = compute_class_weights(train_gen)

    model = build_model(config["model_size"])
    model = compile_model(
        model,
        learning_rate=config["learning_rate"],
        label_smoothing=config["label_smoothing"]
    )

    fold_dir = config_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fold_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS_TUNING,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0
    )

    acc, macro_present, macro_all, y_true, y_pred = evaluate_model(
        model,
        val_df,
        augmentation_name=config["augmentation"]
    )

    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history.history["val_accuracy"]))

    fold_metrics = {
        "fold": fold_id,
        "val_accuracy": float(acc),
        "val_macro_f1_present_classes": float(macro_present),
        "val_macro_f1_all_classes": float(macro_all),
        "best_epoch": best_epoch,
        "best_val_accuracy_during_training": best_val_acc,
        "epochs_ran": len(history.history["accuracy"]),
        "train_images": len(train_df),
        "val_images": len(val_df),
        "train_groups": ",".join(sorted(train_df["group"].unique())),
        "val_groups": ",".join(sorted(val_df["group"].unique())),
    }

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(fold_dir / "history.csv", index=False, encoding="utf-8")

    return fold_metrics, y_true, y_pred


def evaluate_config(config, df, splits, experiment_name):
    config_name = config_to_name(config)
    config_dir = OUTPUT_DIR / "experiments" / experiment_name / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Evaluating config: {config_name}")
    print(json.dumps(config, indent=2))
    print("=" * 80)

    fold_rows = []
    all_y_true = []
    all_y_pred = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        check = validate_one_split(df, train_idx, val_idx)

        if not check["is_valid"]:
            raise RuntimeError(f"Invalid fold {fold_id}: {check['reasons']}")

        fold_metrics, y_true, y_pred = train_one_fold(
            config=config,
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            config_dir=config_dir
        )

        fold_metrics["missing_val_labels"] = ",".join(check["missing_val_labels"])
        fold_rows.append(fold_metrics)

        all_y_true.extend(list(y_true))
        all_y_pred.extend(list(y_pred))

        print(
            f"Fold {fold_id}: "
            f"acc={fold_metrics['val_accuracy']:.4f}, "
            f"f1_present={fold_metrics['val_macro_f1_present_classes']:.4f}, "
            f"best_epoch={fold_metrics['best_epoch']}"
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(config_dir / "fold_results.csv", index=False, encoding="utf-8")

    all_labels = list(range(len(CLASS_NAMES)))
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_macro_f1 = f1_score(
        all_y_true,
        all_y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )

    mean_acc = folds_df["val_accuracy"].mean()
    std_acc = folds_df["val_accuracy"].std(ddof=1)

    mean_f1_present = folds_df["val_macro_f1_present_classes"].mean()
    std_f1_present = folds_df["val_macro_f1_present_classes"].std(ddof=1)

    mean_f1_all = folds_df["val_macro_f1_all_classes"].mean()
    std_f1_all = folds_df["val_macro_f1_all_classes"].std(ddof=1)

    median_best_epoch = int(np.median(folds_df["best_epoch"]))
    mean_best_epoch = float(folds_df["best_epoch"].mean())

    summary = {
        **config,
        "experiment_name": experiment_name,
        "config_name": config_name,
        "mean_val_accuracy": float(mean_acc),
        "std_val_accuracy": float(std_acc),
        "mean_val_macro_f1_present_classes": float(mean_f1_present),
        "std_val_macro_f1_present_classes": float(std_f1_present),
        "mean_val_macro_f1_all_classes": float(mean_f1_all),
        "std_val_macro_f1_all_classes": float(std_f1_all),
        "overall_pooled_accuracy": float(overall_acc),
        "overall_pooled_macro_f1": float(overall_macro_f1),
        "median_best_epoch": median_best_epoch,
        "mean_best_epoch": mean_best_epoch,
    }

    with open(config_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = classification_report(
        all_y_true,
        all_y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    with open(config_dir / "overall_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d"
    )
    plt.title(f"Overall CV Confusion Matrix\n{config_name}")
    plt.tight_layout()
    plt.savefig(config_dir / "overall_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(
        f"Summary: "
        f"mean_acc={mean_acc:.4f}, "
        f"mean_f1_present={mean_f1_present:.4f}, "
        f"overall_macro_f1={overall_macro_f1:.4f}, "
        f"median_best_epoch={median_best_epoch}"
    )

    return summary


# ============================================================
# Tuning strategies
# ============================================================

def choose_best_config(results):
    """
    Primary metric:
        overall_pooled_macro_f1

    Tie breaker:
        mean_val_accuracy
        smaller model size
    """
    model_size_rank = {
        "compact": 0,
        "small": 1,
        "current": 2,
    }

    sorted_results = sorted(
        results,
        key=lambda r: (
            r["overall_pooled_macro_f1"],
            r["mean_val_accuracy"],
            -model_size_rank.get(r["model_size"], 99)
        ),
        reverse=True
    )

    return sorted_results[0]


def run_sequential_tuning(df, splits):
    all_results = []
    current_config = DEFAULT_CONFIG.copy()

    stages = [
        ("learning_rate", LEARNING_RATES),
        ("augmentation", list(AUGMENTATION_CONFIGS.keys())),
        ("model_size", list(MODEL_SIZE_CONFIGS.keys())),
        ("label_smoothing", LABEL_SMOOTHING_VALUES),
    ]

    for stage_name, candidate_values in stages:
        print("\n" + "#" * 90)
        print(f"START STAGE: {stage_name}")
        print("#" * 90)

        stage_results = []

        for value in candidate_values:
            config = current_config.copy()
            config[stage_name] = value

            summary = evaluate_config(
                config=config,
                df=df,
                splits=splits,
                experiment_name=f"stage_{stage_name}"
            )

            stage_results.append(summary)
            all_results.append(summary)

        stage_df = pd.DataFrame(stage_results)
        stage_df.to_csv(
            OUTPUT_DIR / f"stage_{stage_name}_results.csv",
            index=False,
            encoding="utf-8"
        )

        best = choose_best_config(stage_results)
        current_config = {
            "learning_rate": best["learning_rate"],
            "augmentation": best["augmentation"],
            "model_size": best["model_size"],
            "label_smoothing": best["label_smoothing"],
        }

        print("\n" + "-" * 80)
        print(f"Best after stage {stage_name}:")
        print(json.dumps(current_config, indent=2))
        print("-" * 80)

    all_df = pd.DataFrame(all_results)
    all_df.to_csv(
        OUTPUT_DIR / "all_tuning_results.csv",
        index=False,
        encoding="utf-8"
    )

    best_overall = choose_best_config(all_results)

    return best_overall, all_results


def run_full_grid_tuning(df, splits):
    all_results = []

    configs = []

    for lr, aug, model_size, ls in itertools.product(
        LEARNING_RATES,
        list(AUGMENTATION_CONFIGS.keys()),
        list(MODEL_SIZE_CONFIGS.keys()),
        LABEL_SMOOTHING_VALUES
    ):
        configs.append({
            "learning_rate": lr,
            "augmentation": aug,
            "model_size": model_size,
            "label_smoothing": ls,
        })

    print(f"\nFull grid search will evaluate {len(configs)} configs.")
    print("This can take a very long time.")

    for idx, config in enumerate(configs, start=1):
        print(f"\nConfig {idx}/{len(configs)}")

        summary = evaluate_config(
            config=config,
            df=df,
            splits=splits,
            experiment_name="full_grid"
        )

        all_results.append(summary)

        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "all_tuning_results.csv",
            index=False,
            encoding="utf-8"
        )

    best_overall = choose_best_config(all_results)

    return best_overall, all_results


# ============================================================
# Final model training
# ============================================================

def train_final_model(df, best_config):
    print("\n" + "#" * 90)
    print("TRAIN FINAL DEPLOYMENT MODEL")
    print("#" * 90)

    final_epochs = int(best_config["median_best_epoch"])
    final_epochs = max(MIN_FINAL_EPOCHS, final_epochs)

    print("Best config:")
    print(json.dumps(best_config, indent=2))
    print(f"Training final model on all data for {final_epochs} epochs.")

    tf.keras.backend.clear_session()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    full_gen = make_generator(
        df,
        augmentation_name=best_config["augmentation"],
        train=True
    )

    class_weight = compute_class_weights(full_gen)

    model = build_model(best_config["model_size"])
    model = compile_model(
        model,
        learning_rate=best_config["learning_rate"],
        label_smoothing=best_config["label_smoothing"]
    )

    history = model.fit(
        full_gen,
        epochs=final_epochs,
        class_weight=class_weight,
        verbose=1
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(
        OUTPUT_DIR / "final_model_training_history.csv",
        index=False,
        encoding="utf-8"
    )

    model.save(str(FINAL_MODEL_KERAS))
    print(f"\nSaved final Keras model to: {FINAL_MODEL_KERAS}")

    try:
        model.save(str(FINAL_MODEL_H5))
        print(f"Saved final H5 model to: {FINAL_MODEL_H5}")
    except Exception as e:
        print(f"Could not save H5 model: {e}")

    class_mapping = {
        class_name: idx
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    final_summary = {
        "final_epochs": final_epochs,
        "best_config": best_config,
        "final_model_keras": str(FINAL_MODEL_KERAS),
        "final_model_h5": str(FINAL_MODEL_H5),
        "class_mapping": class_mapping,
    }

    with open(OUTPUT_DIR / "final_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    return model


# ============================================================
# Main
# ============================================================

def main():
    df = load_dataset()
    print_dataset_summary(df)

    df.to_csv(OUTPUT_DIR / "all_images_metadata.csv", index=False, encoding="utf-8")
    save_group_distribution(df)
    save_label_group_coverage(df)

    missing_classes = [
        label for label in CLASS_NAMES
        if (df["label"] == label).sum() == 0
    ]

    if missing_classes:
        raise RuntimeError(f"Missing classes in dataset: {missing_classes}")

    if df["group"].nunique() < N_SPLITS:
        raise RuntimeError(
            f"Not enough groups for {N_SPLITS}-fold CV. "
            f"Found {df['group'].nunique()} groups."
        )

    splits, used_seed = find_valid_group_kfold_splits(df)

    print("\nUsing K-Fold split seed:", used_seed)

    if TUNING_STRATEGY == "sequential":
        best_config, all_results = run_sequential_tuning(df, splits)
    elif TUNING_STRATEGY == "full_grid":
        best_config, all_results = run_full_grid_tuning(df, splits)
    else:
        raise ValueError(f"Unknown TUNING_STRATEGY: {TUNING_STRATEGY}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        OUTPUT_DIR / "all_tuning_results.csv",
        index=False,
        encoding="utf-8"
    )

    best_config_path = OUTPUT_DIR / "best_hyperparameter_config.json"
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("\n" + "=" * 90)
    print("BEST HYPERPARAMETER CONFIG")
    print("=" * 90)
    print(json.dumps(best_config, indent=2))

    train_final_model(df, best_config)

    print("\n" + "=" * 90)
    print("TUNING COMPLETE")
    print("=" * 90)
    print(f"All tuning results: {OUTPUT_DIR / 'all_tuning_results.csv'}")
    print(f"Best config: {best_config_path}")
    print(f"Final model: {FINAL_MODEL_KERAS}")
    print(f"H5 compatibility model: {FINAL_MODEL_H5}")

    print("\nRecommended next command:")
    print(r"python scripts\run_cnn_inference.py --clip Clips\45.mp4 --fps 0.5")


if __name__ == "__main__":
    main()