"""Train the survivor-status CNN classifier.

Loads cropped survivor-portrait images from cnn/datasets/ (7 classes),
trains a lightweight CNN, and saves the best model, training history,
confusion matrix, and classification report.
"""

import os
import sys
import json
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_HEIGHT = 92
IMG_WIDTH = 98
BATCH_SIZE = 8
NUM_CLASSES = 7
EPOCHS = 50
LEARNING_RATE = 0.0005

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_avatar_model.h5")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
CM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
CLASS_MAP_PATH = os.path.join(OUTPUT_DIR, "class_mapping.json")

# --- Dataset inspection ---
print("=== Dataset distribution ===")
class_counts = {}
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_dir) and not class_name.startswith("."):
        count = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        class_counts[class_name] = count
        print(f"  {class_name}: {count} images")
print(f"  Total: {sum(class_counts.values())} images")

# --- Data generators ---
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"\n=== Class mapping ===\n{json.dumps(class_indices, indent=2)}")
with open(CLASS_MAP_PATH, "w") as f:
    json.dump(class_indices, f, indent=2)

# --- Model architecture ---
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

model = build_model((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# --- Training ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10,
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    ),
]

print("\n=== Training ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# --- Save training history ---
hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_PATH, "w") as f:
    json.dump(hist, f, indent=2)

# --- Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist["accuracy"], label="Train")
axes[0].plot(hist["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist["loss"], label="Train")
axes[1].plot(hist["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
print(f"Training curves saved to {CURVES_PATH}")

# --- Evaluation on validation set ---
print("\n=== Validation evaluation ===")
val_gen.reset()
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open(REPORT_PATH, "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("Survivor Status CNN — Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
print(f"Confusion matrix saved to {CM_PATH}")

best_val_acc = max(hist["val_accuracy"])
print(f"\nBest validation accuracy: {best_val_acc:.4f}")
print(f"Model saved to {MODEL_PATH}")
