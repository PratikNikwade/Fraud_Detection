"""
=================================================================
  Internship Certificate Fraud Detection — Local Training Script
  Dataset: godzilla04/internship-certificates (Kaggle)
  Model: EfficientNetB0 Transfer Learning (2-Phase Training)
  
  Improvements:
    • Label smoothing for better generalisation
    • Mixup augmentation for virtual sample creation
    • Cosine-annealing LR schedule with warm-up
    • Stronger online augmentation layer
    • Test-Time Augmentation (TTA) for robust evaluation
    • Stratified train / val / test split via scikit-learn
    • Comprehensive JSON training report
=================================================================
"""

import os
import sys
import json
import zipfile
import shutil
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from pathlib import Path

# ── Step 0: Configuration ─────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "training_output")
KAGGLE_JSON = os.path.join(PROJECT_DIR, "kaggle.json")

CONFIG = {
    "img_size":       (224, 224),
    "batch_size":     8,
    "epochs_frozen":  30,       # Phase 1: train head only
    "epochs_finetune": 40,      # Phase 2: fine-tune top layers
    "lr_frozen":      1e-3,
    "lr_finetune":    1e-5,
    "dropout":        0.4,
    "l2_reg":         1e-4,
    "val_split":      0.2,
    "test_split":     0.15,     # 15% held out for final testing
    "aug_factor":     8,        # reduced from 15 to prevent overfitting
    "label_smoothing": 0.1,     # NEW: prevents overconfident predictions
    "mixup_alpha":    0.2,      # NEW: mixup augmentation strength
    "warmup_epochs":  3,        # NEW: LR warmup
    "tta_augments":   5,        # NEW: test-time augmentation count
    "model_name":     "best_fraud_model.keras",
}

CLASSES = ["real", "fake"]  # 0 = real, 1 = fake


# ── Step 1: Download Dataset from Kaggle ──────────────────────────
def setup_kaggle_and_download():
    """Configure Kaggle API and download the dataset."""
    print("\n" + "=" * 60)
    print("  STEP 1: Downloading Dataset from Kaggle")
    print("=" * 60)

    # Set up Kaggle credentials
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest_kaggle = os.path.join(kaggle_dir, "kaggle.json")
    shutil.copy2(KAGGLE_JSON, dest_kaggle)

    # On Windows, no chmod needed; on Linux/Mac:
    if os.name != "nt":
        os.chmod(dest_kaggle, 0o600)

    print(f"  ✅ Kaggle credentials configured from {KAGGLE_JSON}")

    # Import kaggle after setting up credentials
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  📦 Installing kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle --quiet")
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset_name = "godzilla04/internship-certificates"
    zip_path = os.path.join(PROJECT_DIR, "internship-certificates.zip")

    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"  📁 Dataset already exists at {DATASET_DIR}, skipping download.")
    else:
        print(f"  ⬇️  Downloading '{dataset_name}' ...")
        api.dataset_download_files(dataset_name, path=PROJECT_DIR, unzip=False)
        print(f"  📦 Extracting dataset ...")
        os.makedirs(DATASET_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        # Clean up zip
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"  ✅ Dataset extracted to {DATASET_DIR}")

    # Discover the dataset structure
    print("\n  📂 Dataset contents:")
    fake_dir, real_dir = None, None
    for root, dirs, files in os.walk(DATASET_DIR):
        for d in dirs:
            full_path = os.path.join(root, d)
            dl = d.lower()
            if "fake" in dl:
                fake_dir = full_path
                n_files = len([f for f in os.listdir(full_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])
                print(f"    📁 FAKE: {full_path} ({n_files} images)")
            elif "real" in dl:
                real_dir = full_path
                n_files = len([f for f in os.listdir(full_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])
                print(f"    📁 REAL: {full_path} ({n_files} images)")

    if fake_dir is None or real_dir is None:
        # Try one level deeper
        for item in os.listdir(DATASET_DIR):
            subdir = os.path.join(DATASET_DIR, item)
            if os.path.isdir(subdir):
                for sub_item in os.listdir(subdir):
                    sub_path = os.path.join(subdir, sub_item)
                    if os.path.isdir(sub_path):
                        sl = sub_item.lower()
                        if "fake" in sl and fake_dir is None:
                            fake_dir = sub_path
                        elif "real" in sl and real_dir is None:
                            real_dir = sub_path

    if fake_dir is None or real_dir is None:
        print("\n  ❌ Could not find fake/real directories. Listing all contents:")
        for root, dirs, files in os.walk(DATASET_DIR):
            level = root.replace(DATASET_DIR, "").count(os.sep)
            indent = "    " * (level + 2)
            print(f"{indent}📁 {os.path.basename(root)}/")
            if level < 3:
                for f in files[:5]:
                    print(f"{indent}  📄 {f}")
                if len(files) > 5:
                    print(f"{indent}  ... and {len(files) - 5} more")
        sys.exit(1)

    return fake_dir, real_dir


# ── Step 2: Load & Preprocess Images ─────────────────────────────
def load_images_from_folder(folder, img_size=(224, 224)):
    """Load all images from a folder, return (images_np, filenames)."""
    images, names = [], []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    for ext in exts:
        for p in sorted(glob.glob(os.path.join(folder, ext))):
            img = cv2.imread(p)
            if img is None:
                print(f"    ⚠ Could not read {os.path.basename(p)}, skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            images.append(img)
            names.append(os.path.basename(p))
    # Also try case-insensitive on Windows
    if len(images) == 0:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                p = os.path.join(folder, f)
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                names.append(f)
    return np.array(images, dtype=np.float32), names


def load_dataset(fake_dir, real_dir):
    """Load the full dataset, split into train/val/test with stratification."""
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("  STEP 2: Loading & Splitting Dataset (Stratified)")
    print("=" * 60)

    X_real, n_real = load_images_from_folder(real_dir, CONFIG["img_size"])
    X_fake, n_fake = load_images_from_folder(fake_dir, CONFIG["img_size"])

    print(f"  Real certificates loaded: {len(X_real)}")
    print(f"  Fake certificates loaded: {len(X_fake)}")

    if len(X_real) == 0 or len(X_fake) == 0:
        raise ValueError(f"Dataset is empty! Real: {len(X_real)}, Fake: {len(X_fake)}")

    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.array([0] * len(X_real) + [1] * len(X_fake), dtype=np.int32)
    names = list(n_real) + list(n_fake)

    # Stratified split: first carve out test set, then val from remainder
    X_remain, X_test, y_remain, y_test, names_remain, test_names = train_test_split(
        X, y, names,
        test_size=CONFIG["test_split"],
        random_state=42,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain,
        test_size=CONFIG["val_split"],
        random_state=42,
        stratify=y_remain
    )

    print(f"\n  📊 Data Split (Stratified):")
    print(f"    Train : {len(X_train)} (real={np.sum(y_train==0)}, fake={np.sum(y_train==1)})")
    print(f"    Val   : {len(X_val)}  (real={np.sum(y_val==0)}, fake={np.sum(y_val==1)})")
    print(f"    Test  : {len(X_test)}  (real={np.sum(y_test==0)}, fake={np.sum(y_test==1)})")

    return X_train, y_train, X_val, y_val, X_test, y_test, test_names


# ── Step 3: Offline Augmentation with Mixup ───────────────────────
def augment_offline(X, y, factor=8):
    """Expand the training set via offline augmentation + mixup."""
    print(f"\n  🔄 Augmenting training data ({factor}x) ...")

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode="reflect",
    )

    X_aug, y_aug = [X.copy()], [y.copy().astype(np.float32)]

    for i in range(factor - 1):
        batch = np.zeros_like(X)
        for j, img in enumerate(X):
            sample = img[np.newaxis]
            gen = aug.flow(sample, batch_size=1)
            batch[j] = next(gen)[0]
        X_aug.append(batch)
        y_aug.append(y.copy().astype(np.float32))
        if (i + 1) % 3 == 0:
            print(f"    Progress: {i + 2}/{factor}")

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    X_out = np.clip(X_out, 0, 255)

    # ── Mixup augmentation ──
    print(f"  🔀 Applying Mixup augmentation (alpha={CONFIG['mixup_alpha']}) ...")
    alpha = CONFIG["mixup_alpha"]
    n_mixup = len(X_out) // 4  # create 25% extra samples via mixup
    indices_a = np.random.randint(0, len(X_out), n_mixup)
    indices_b = np.random.randint(0, len(X_out), n_mixup)
    lam = np.random.beta(alpha, alpha, n_mixup).astype(np.float32)

    X_mix = np.zeros((n_mixup, *X_out.shape[1:]), dtype=np.float32)
    y_mix = np.zeros(n_mixup, dtype=np.float32)
    for k in range(n_mixup):
        X_mix[k] = lam[k] * X_out[indices_a[k]] + (1 - lam[k]) * X_out[indices_b[k]]
        y_mix[k] = lam[k] * y_out[indices_a[k]] + (1 - lam[k]) * y_out[indices_b[k]]

    X_out = np.concatenate([X_out, X_mix], axis=0)
    y_out = np.concatenate([y_out, y_mix], axis=0)

    # Shuffle
    indices = np.random.permutation(len(X_out))
    X_out, y_out = X_out[indices], y_out[indices]

    print(f"  ✅ Augmented: {len(X)} → {len(X_out)} images (incl. {n_mixup} mixup)")
    return X_out, y_out


# ── Step 4: Build Model ──────────────────────────────────────────
def build_model(trainable_base=False):
    """Build EfficientNetB0 with stronger augmentation and classification head."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    inputs = keras.Input(shape=(*CONFIG["img_size"], 3), name="input_image")

    # Enhanced online augmentation (active only during training)
    aug_layer = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="online_augmentation")

    x = aug_layer(inputs)

    # Pre-trained backbone
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = trainable_base

    x = base.output

    # Classification head with stronger regularisation
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=keras.regularizers.l2(CONFIG["l2_reg"]),
        name="dense_256",
    )(x)
    x = layers.Dropout(CONFIG["dropout"])(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=keras.regularizers.l2(CONFIG["l2_reg"]),
        name="dense_128",
    )(x)
    x = layers.Dropout(CONFIG["dropout"] * 0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FraudDetector")
    return model, base


def unfreeze_top_layers(model, base_model, n_layers=30):
    """Unfreeze the top N layers of the backbone for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Backbone trainable layers: {trainable_count} / {len(base_model.layers)}")


# ── Cosine Annealing with Warmup ─────────────────────────────────
class WarmupCosineDecay(object):
    """Learning rate schedule: linear warmup → cosine decay."""
    def __init__(self, base_lr, total_epochs, warmup_epochs=3):
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        import math
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Step 5: Training ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val):
    """Two-phase training: frozen head → fine-tune with label smoothing."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateScheduler
    )

    print("\n" + "=" * 60)
    print("  STEP 4: Training Model (2-Phase with Label Smoothing)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Augment training data
    X_train_aug, y_train_aug = augment_offline(X_train, y_train, CONFIG["aug_factor"])

    # Label smoothing: convert hard labels to soft labels
    ls = CONFIG["label_smoothing"]
    y_train_smooth = y_train_aug * (1 - ls) + (1 - y_train_aug) * ls
    # Validation labels stay hard for accurate metric computation
    y_val_float = y_val.astype(np.float32)

    # ── Phase 1: Train head only ─────────────────────
    print("\n  ═══ Phase 1: Training Classification Head (frozen backbone) ═══")
    model, base = build_model(trainable_base=False)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr_frozen"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    ckpt_path_1 = os.path.join(OUTPUT_DIR, "ckpt_phase1.keras")
    lr_schedule_1 = WarmupCosineDecay(CONFIG["lr_frozen"], CONFIG["epochs_frozen"], CONFIG["warmup_epochs"])

    callbacks_1 = [
        ModelCheckpoint(ckpt_path_1, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=10, restore_best_weights=True, verbose=1),
        LearningRateScheduler(lr_schedule_1, verbose=0),
    ]

    h1 = model.fit(
        X_train_aug, y_train_smooth,
        validation_data=(X_val, y_val_float),
        epochs=CONFIG["epochs_frozen"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks_1,
        verbose=1,
    )

    # ── Phase 2: Fine-tune top layers ────────────────
    print("\n  ═══ Phase 2: Fine-tuning Top Backbone Layers ═══")
    unfreeze_top_layers(model, base, n_layers=30)

    lr_schedule_2 = WarmupCosineDecay(CONFIG["lr_finetune"], CONFIG["epochs_finetune"], 2)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr_finetune"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    ckpt_path_2 = os.path.join(OUTPUT_DIR, "ckpt_phase2.keras")
    callbacks_2 = [
        ModelCheckpoint(ckpt_path_2, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max",
                      patience=12, restore_best_weights=True, verbose=1),
        LearningRateScheduler(lr_schedule_2, verbose=0),
    ]

    h2 = model.fit(
        X_train_aug, y_train_smooth,
        validation_data=(X_val, y_val_float),
        epochs=CONFIG["epochs_finetune"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks_2,
        verbose=1,
    )

    return model, h1, h2


# ── Step 6: Test-Time Augmentation (TTA) ──────────────────────────
def predict_with_tta(model, X, n_augments=5):
    """Run TTA: average predictions over original + augmented versions."""
    import tensorflow as tf

    # Original prediction
    preds = model.predict(X.astype(np.float32), verbose=0).ravel()
    all_preds = [preds]

    for i in range(n_augments):
        X_aug = X.copy()
        for j in range(len(X_aug)):
            img = X_aug[j]
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = img[:, ::-1, :]
            # Random brightness
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 255)
            # Random slight rotation via affine
            angle = np.random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img.astype(np.uint8), M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            X_aug[j] = img

        aug_preds = model.predict(X_aug.astype(np.float32), verbose=0).ravel()
        all_preds.append(aug_preds)

    # Average all predictions
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


# ── Step 7: Evaluate on Test Set ──────────────────────────────────
def evaluate_model(model, X_test, y_test, test_names):
    """Comprehensive evaluation with TTA on the held-out test set."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, accuracy_score, f1_score
    )

    print("\n" + "=" * 60)
    print("  STEP 5: Testing Model on Held-Out Test Set (with TTA)")
    print("=" * 60)

    # Predict with TTA
    print(f"  Running Test-Time Augmentation ({CONFIG['tta_augments']} augments) ...")
    y_prob = predict_with_tta(model, X_test, CONFIG["tta_augments"])
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    print(f"\n  📊 TEST RESULTS (with TTA):")
    print(f"  {'─' * 40}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  {'─' * 40}")

    # Per-class metrics
    print(f"\n  📋 Classification Report:")
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Real   Fake")
    print(f"    True Real  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"    True Fake  [{cm[1][0]:4d}  {cm[1][1]:4d}]")

    # Per-image predictions
    print(f"\n  🔍 Individual Test Predictions:")
    print(f"  {'─' * 65}")
    print(f"  {'Filename':<30s} {'True':>6s} {'Pred':>6s} {'Prob':>8s} {'✓/✗':>4s}")
    print(f"  {'─' * 65}")

    correct_real = 0
    total_real = 0
    correct_fake = 0
    total_fake = 0

    for i in range(len(X_test)):
        true_label = CLASSES[y_test[i]]
        pred_label = CLASSES[y_pred[i]]
        prob = y_prob[i]
        correct = "✓" if y_pred[i] == y_test[i] else "✗"
        name = test_names[i] if i < len(test_names) else f"img_{i}"

        if y_test[i] == 0:
            total_real += 1
            if y_pred[i] == 0:
                correct_real += 1
        else:
            total_fake += 1
            if y_pred[i] == 1:
                correct_fake += 1

        print(f"  {name:<30s} {true_label:>6s} {pred_label:>6s} {prob:>8.4f} {correct:>4s}")

    print(f"  {'─' * 65}")

    # Per-class accuracy summary
    real_acc = correct_real / total_real * 100 if total_real > 0 else 0
    fake_acc = correct_fake / total_fake * 100 if total_fake > 0 else 0

    print(f"\n  🎯 Per-Class Accuracy:")
    print(f"    REAL certificates: {correct_real}/{total_real} correct ({real_acc:.1f}%)")
    print(f"    FAKE certificates: {correct_fake}/{total_fake} correct ({fake_acc:.1f}%)")
    print(f"    Overall          : {np.sum(y_pred == y_test)}/{len(y_test)} correct ({acc*100:.1f}%)")

    return acc, f1, auc, cm


# ── Step 8: Save Final Model ─────────────────────────────────────
def save_model(model, acc, f1, auc):
    """Save the trained model and training report."""
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("  STEP 6: Saving Model")
    print("=" * 60)

    # Save to output dir
    model_path = os.path.join(OUTPUT_DIR, CONFIG["model_name"])
    model.save(model_path)
    print(f"  ✅ Model saved → {model_path}")

    # Also copy to project root for the Flask app
    root_model = os.path.join(PROJECT_DIR, CONFIG["model_name"])
    shutil.copy2(model_path, root_model)
    print(f"  ✅ Model copied → {root_model} (for Flask API)")

    # Save comprehensive training report
    report = {
        "dataset": "godzilla04/internship-certificates",
        "model": "EfficientNetB0 (Transfer Learning)",
        "test_accuracy": round(acc, 4),
        "test_f1_score": round(f1, 4),
        "test_auc_roc": round(auc, 4),
        "improvements": [
            "Label smoothing (0.1)",
            "Mixup augmentation (alpha=0.2)",
            "Cosine annealing LR with warmup",
            "Enhanced online augmentation",
            "Test-Time Augmentation (5 augments)",
            "Stratified train/val/test split",
        ],
        "config": {k: str(v) for k, v in CONFIG.items()},
    }
    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Training report → {report_path}")

    # Also save to project root for Flask API to read
    root_report = os.path.join(PROJECT_DIR, "training_report.json")
    shutil.copy2(report_path, root_report)
    print(f"  ✅ Training report copied → {root_report}")

    return model_path


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🔍 Internship Certificate Fraud Detection")
    print("  📦 Dataset: godzilla04/internship-certificates")
    print("  🧠 Model: EfficientNetB0 Transfer Learning")
    print("  ⚡ Improvements: Label Smoothing + Mixup + TTA")
    print("=" * 60)

    # GPU info
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs available: {len(gpus)}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Step 1: Download dataset
    fake_dir, real_dir = setup_kaggle_and_download()

    # Step 2: Load and split data (stratified)
    X_train, y_train, X_val, y_val, X_test, y_test, test_names = \
        load_dataset(fake_dir, real_dir)

    # Step 3-4: Train model (includes augmentation + label smoothing)
    model, h1, h2 = train_model(X_train, y_train, X_val, y_val)

    # Step 5: Test model with TTA
    acc, f1, auc, cm = evaluate_model(model, X_test, y_test, test_names)

    # Step 6: Save model
    save_model(model, acc, f1, auc)

    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print(f"  📊 Test Accuracy: {acc*100:.1f}%")
    print(f"  📊 Test F1 Score: {f1:.4f}")
    print(f"  📊 Test AUC-ROC : {auc:.4f}")
    print("  📁 Output: training_output/")
    print("  🚀 Model ready for Flask API: best_fraud_model.keras")
    print("=" * 60)


if __name__ == "__main__":
    main()
