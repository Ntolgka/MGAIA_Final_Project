import os
import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs_final_cnn"
PIPELINE_MODEL_DIR = BASE_DIR / "outputs"

IMG_HEIGHT = 92
IMG_WIDTH = 98

BATCH_SIZE = 8
MAX_EPOCHS = 80
LEARNING_RATE = 0.0005
SEED = 42

VAL_SIZE = 0.20
MAX_SPLIT_TRIALS = 500

# These class names must match scripts/config.py exactly.
CLASS_NAMES = [
    "0_healthy",
    "1_injured",
    "2_downed",
    "3_ballooned",
    "4_chaired",
    "5_eliminated",
    "6_escaped",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_KERAS = PIPELINE_MODEL_DIR / "best_avatar_model.keras"
FINAL_MODEL_H5 = PIPELINE_MODEL_DIR / "best_avatar_model.h5"

CLASS_MAP_PATH = PIPELINE_MODEL_DIR / "class_mapping.json"


# ============================================================
# Filename parser
# ============================================================

def parse_filename(filename: str):
    """
    Expected filename:
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
# Load dataset
# ============================================================

def load_dataset():
    rows = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Cannot find dataset folder: {DATASET_DIR.resolve()}")

    for label in CLASS_NAMES:
        class_dir = DATASET_DIR / label

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Please check your dataset structure."
            )

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
        raise ValueError("No image files found.")

    return df


def print_summary(df, title):
    print(f"\n=== {title} ===")
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

    print("\n=== Label coverage across video groups ===")
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
# Group-level validation split for monitoring
# ============================================================

def find_group_validation_split(df):
    """
    Find a group-level train/validation split.

    Rules:
    1. No video group overlap.
    2. Training set must contain all classes.
    3. Prefer validation set containing all classes.
    4. Prefer validation label distribution close to full dataset.
    """
    X = df["filepath"].values
    y = df["label"].values
    groups = df["group"].values

    full_dist = df["label"].value_counts(normalize=True).reindex(CLASS_NAMES, fill_value=0)

    best_score = float("inf")
    best_split = None
    best_info = None

    for trial in range(MAX_SPLIT_TRIALS):
        seed = SEED + trial

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VAL_SIZE,
            random_state=seed
        )

        train_idx, val_idx = next(splitter.split(X, y, groups))

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_labels = set(train_df["label"].unique())
        val_labels = set(val_df["label"].unique())
        all_labels = set(CLASS_NAMES)

        missing_train = sorted(all_labels - train_labels)
        missing_val = sorted(all_labels - val_labels)

        # Training must contain all classes.
        if missing_train:
            continue

        val_dist = val_df["label"].value_counts(normalize=True).reindex(CLASS_NAMES, fill_value=0)
        distribution_gap = float(np.abs(val_dist - full_dist).sum())

        # Strong penalty if validation misses classes.
        score = len(missing_val) * 100.0 + distribution_gap

        if score < best_score:
            best_score = score
            best_split = (train_idx, val_idx)
            best_info = {
                "seed": seed,
                "missing_val": missing_val,
                "distribution_gap": distribution_gap,
                "train_groups": sorted(train_df["group"].unique()),
                "val_groups": sorted(val_df["group"].unique()),
            }

        # Perfect enough: validation contains all labels.
        if best_info and not best_info["missing_val"]:
            break

    if best_split is None:
        raise RuntimeError(
            "Could not find a valid group-level validation split.\n"
            "This probably means some classes appear in too few video groups."
        )

    print("\n=== Selected group-level validation split ===")
    print(f"Seed: {best_info['seed']}")
    print(f"Validation missing labels: {best_info['missing_val']}")
    print(f"Distribution gap: {best_info['distribution_gap']:.4f}")
    print(f"Train groups: {best_info['train_groups']}")
    print(f"Validation groups: {best_info['val_groups']}")

    return best_split, best_info


# ============================================================
# Data generators
# ============================================================

def make_train_generator(df, augment=True, shuffle=True):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.08,
            brightness_range=[0.85, 1.15],
            fill_mode="nearest"
        )
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255)

    gen = datagen.flow_from_dataframe(
        df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=shuffle,
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


def build_model():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = conv_block(inputs, 16, 0.10)
    x = conv_block(x, 32, 0.15)
    x = conv_block(x, 64, 0.20)
    x = conv_block(x, 128, 0.25)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    return models.Model(inputs, outputs)


def compile_model(model):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Plotting and evaluation
# ============================================================

def save_training_curves(history, out_path, title_prefix):
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["accuracy"], label="train acc")
    if "val_accuracy" in hist:
        axes[0].plot(hist["val_accuracy"], label="val acc")
    axes[0].set_title(f"{title_prefix} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train loss")
    if "val_loss" in hist:
        axes[1].plot(hist["val_loss"], label="val loss")
    axes[1].set_title(f"{title_prefix} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_on_validation(model, val_df):
    val_gen = make_train_generator(val_df, augment=False, shuffle=False)
    val_gen.reset()

    probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_gen.classes

    all_labels = list(range(len(CLASS_NAMES)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\n=== Monitor validation report ===")
    print(report)

    with open(OUTPUT_DIR / "monitor_validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

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
    plt.title("Monitor Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monitor_validation_confusion_matrix.png", dpi=200)
    plt.close(fig)

    metrics = {
        "monitor_val_accuracy": float(acc),
        "monitor_val_macro_f1": float(macro_f1),
    }

    with open(OUTPUT_DIR / "monitor_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ============================================================
# Training stages
# ============================================================

def train_monitor_model(train_df, val_df):
    print("\n\n=== Stage 1: Train monitor model with group-level validation ===")

    train_gen = make_train_generator(train_df, augment=True, shuffle=True)
    val_gen = make_train_generator(val_df, augment=False, shuffle=False)

    class_weight = compute_class_weights(train_gen)

    model = build_model()
    model = compile_model(model)

    monitor_model_path = OUTPUT_DIR / "monitor_best_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(monitor_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    save_training_curves(
        history,
        OUTPUT_DIR / "monitor_training_curves.png",
        "Monitor Model"
    )

    history_dict = {
        k: [float(x) for x in v]
        for k, v in history.history.items()
    }

    with open(OUTPUT_DIR / "monitor_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history.history["val_accuracy"]))

    print(f"\nBest monitor validation accuracy: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")

    metrics = evaluate_on_validation(model, val_df)

    return best_epoch, best_val_acc, metrics


def train_final_deployment_model(full_df, final_epochs):
    print("\n\n=== Stage 2: Train final deployment model on all labeled data ===")
    print(f"Training on all data for {final_epochs} epochs.")

    full_gen = make_train_generator(full_df, augment=True, shuffle=True)
    class_weight = compute_class_weights(full_gen)

    model = build_model()
    model = compile_model(model)

    history = model.fit(
        full_gen,
        epochs=final_epochs,
        class_weight=class_weight,
        verbose=1
    )

    save_training_curves(
        history,
        OUTPUT_DIR / "final_deployment_training_curves.png",
        "Final Deployment Model"
    )

    history_dict = {
        k: [float(x) for x in v]
        for k, v in history.history.items()
    }

    with open(OUTPUT_DIR / "final_deployment_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    # Save final deployment model to pipeline location.
    model.save(str(FINAL_MODEL_KERAS))
    print(f"\nSaved final Keras model to: {FINAL_MODEL_KERAS}")

    # Also save H5 for compatibility with teammate's original code.
    try:
        model.save(str(FINAL_MODEL_H5))
        print(f"Saved H5 compatibility model to: {FINAL_MODEL_H5}")
    except Exception as e:
        print(f"Could not save H5 model: {e}")

    class_mapping = {
        class_name: idx
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    print(f"Saved class mapping to: {CLASS_MAP_PATH}")

    return model


# ============================================================
# Main
# ============================================================

def main():
    df = load_dataset()

    print_summary(df, "Full labeled dataset")
    df.to_csv(OUTPUT_DIR / "all_images_metadata.csv", index=False, encoding="utf-8")

    save_group_distribution(df)
    save_label_group_coverage(df)

    # Check that every class has at least one sample.
    missing_classes = [
        label for label in CLASS_NAMES
        if (df["label"] == label).sum() == 0
    ]

    if missing_classes:
        raise RuntimeError(
            f"These classes have no samples: {missing_classes}\n"
            f"Cannot train a 7-class final model."
        )

    # Stage 1: group-level validation split for choosing training length.
    (train_idx, val_idx), split_info = find_group_validation_split(df)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_df.to_csv(OUTPUT_DIR / "monitor_train_metadata.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR / "monitor_val_metadata.csv", index=False, encoding="utf-8")

    print_summary(train_df, "Monitor train set")
    print_summary(val_df, "Monitor validation set")

    with open(OUTPUT_DIR / "monitor_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    best_epoch, best_val_acc, monitor_metrics = train_monitor_model(train_df, val_df)

    # Stage 2: train final deployment model on all data.
    # Use the epoch selected by group-level validation.
    final_epochs = max(3, best_epoch)

    final_model = train_final_deployment_model(df, final_epochs)

    summary = {
        "best_epoch_from_monitor_validation": best_epoch,
        "final_epochs": final_epochs,
        "best_monitor_val_accuracy": best_val_acc,
        "monitor_metrics": monitor_metrics,
        "final_model_keras": str(FINAL_MODEL_KERAS),
        "final_model_h5": str(FINAL_MODEL_H5),
        "class_names": CLASS_NAMES,
    }

    with open(OUTPUT_DIR / "final_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final training complete ===")
    print(f"Final epochs: {final_epochs}")
    print(f"Final model for pipeline: {FINAL_MODEL_KERAS}")
    print(f"H5 compatibility model: {FINAL_MODEL_H5}")
    print(f"Training summary: {OUTPUT_DIR / 'final_training_summary.json'}")


if __name__ == "__main__":
    main()