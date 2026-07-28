"""
Train a classical ML classifier that predicts an IGA-scale acne severity
grade from a facial photo, using the Roboflow "Acne Severity Classification"
dataset (folder-structure export) unzipped under data/raw/acne-severity/.

Usage: python3 ml/train_severity_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import extract_features  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "acne-severity"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "severity_classifier.joblib"
METADATA_PATH = MODEL_DIR / "severity_classifier_metadata.json"


def load_split(split: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for class_name in classes:
        class_dir = DATA_DIR / split / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            with Image.open(image_path) as image:
                features.append(extract_features(image))
            labels.append(class_name)
    return np.array(features), np.array(labels)


def main() -> None:
    classes = sorted(
        p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir()
    )
    print(f"Classes: {classes}")

    x_train, y_train = load_split("train", classes)
    x_valid, y_valid = load_split("valid", classes)
    x_test, y_test = load_split("test", classes)
    print(
        f"Loaded {len(x_train)} train, {len(x_valid)} valid, "
        f"{len(x_test)} test images."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    valid_predictions = model.predict(x_valid)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    print(f"\nValidation accuracy (train-only fit): {valid_accuracy:.3f}")
    print(classification_report(y_valid, valid_predictions, zero_division=0))

    # Refit on train+valid combined for the final model (more training data),
    # then report accuracy on the held-out test split.
    x_train_full = np.concatenate([x_train, x_valid])
    y_train_full = np.concatenate([y_train, y_valid])

    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(x_train_full, y_train_full)

    test_predictions = final_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nHeld-out test accuracy (train+valid fit): {test_accuracy:.3f}")
    print(classification_report(y_test, test_predictions, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    # Compression matters here: the fully grown forest is ~96 MB raw, which
    # GitHub warns on at 50 MB and rejects at 100 MB. compress=3 brings it
    # to ~23 MB losslessly. Capping tree depth would shrink it further but
    # costs 3-10 points of accuracy, so compression is the better trade.
    joblib.dump(final_model, MODEL_PATH, compress=3)

    metadata = {
        "classes": classes,
        "train_images": len(x_train_full),
        "test_images": len(x_test),
        "test_accuracy": test_accuracy,
        "feature_extractor": "ml.features.extract_features",
        "dataset": "Roboflow Acne Severity Classification (Taschenbier), v1",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
