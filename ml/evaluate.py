"""
Skin-tone-stratified evaluation of both halves of the vision pipeline:

  1. the inflammatory-lesion detector (ml/lesion_detector.py), scored by how
     well its count ranks images by clinician IGA grade, and
  2. the IGA severity classifier (ml/severity_classifier.py), scored by
     accuracy and mean absolute grade error.

Both are reported per Individual Typology Angle bucket, because an aggregate
number can hide a pipeline that works on light skin and fails on dark skin.

Evaluated on images that were never used to fit the thing being measured:
the classifier is scored on the test split only, and the detector on the
test split plus every train/valid image the tuner did not sample.

Usage: python3 ml/evaluate.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import extract_features  # noqa: E402
from ml.lesion_detector import find_lesion_contours  # noqa: E402
from ml.skin_tone import estimate_ita, ita_to_bucket  # noqa: E402
from ml.tune_detector import IGA_ORDINAL, collect_images, spearman  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "acne-severity"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "severity_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "ml" / "models" / "evaluation_report.json"

BUCKET_ORDER = (
    "very_light",
    "light",
    "intermediate",
    "tan",
    "brown",
    "dark",
)
MIN_BUCKET_SIZE = 15


def unique_images(splits: tuple[str, ...]) -> list[tuple[str, str]]:
    by_base: dict[str, tuple[str, str]] = {}
    for split in splits:
        for path in sorted(glob.glob(str(DATA_DIR / split / "*" / "*"))):
            label = Path(path).parent.name
            base = re.sub(
                r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", path, flags=re.I
            )
            by_base.setdefault(base, (path, label))
    return list(by_base.values())


def main() -> None:
    tuner_paths = {path for path, _ in collect_images()}

    test_images = unique_images(("test",))
    detector_images = test_images + [
        (path, label)
        for path, label in unique_images(("train", "valid"))
        if path not in tuner_paths
    ]
    print(
        f"detector eval on {len(detector_images)} unseen images; "
        f"classifier eval on {len(test_images)} test images"
    )

    model = joblib.load(MODEL_PATH)

    records = []
    for index, (path, label) in enumerate(detector_images):
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            bucket = ita_to_bucket(estimate_ita(rgb_image))
            bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
            count = len(find_lesion_contours(bgr_array))
            is_test = path.startswith(str(DATA_DIR / "test"))
            predicted = (
                str(model.predict(extract_features(rgb_image).reshape(1, -1))[0])
                if is_test
                else None
            )

        records.append(
            {
                "bucket": bucket,
                "grade": IGA_ORDINAL[label],
                "count": count,
                "predicted": predicted,
                "is_test": is_test,
            }
        )
        if (index + 1) % 200 == 0:
            print(f"  processed {index + 1}/{len(detector_images)}")

    buckets = np.array([r["bucket"] for r in records])
    grades = np.array([r["grade"] for r in records], dtype=float)
    counts = np.array([r["count"] for r in records], dtype=float)
    is_test = np.array([r["is_test"] for r in records])

    report: dict[str, object] = {}

    print("\n" + "=" * 74)
    print("DETECTOR: does inflammatory-lesion count rank severity correctly?")
    print("=" * 74)
    print(f"{'skin tone':<14}{'n':>5}{'Spearman':>11}"
          f"{'med IGA0':>10}{'med IGA3-4':>12}")
    detector_rows = {}
    for bucket in BUCKET_ORDER + ("ALL",):
        mask = np.ones(len(records), bool) if bucket == "ALL" else (
            buckets == bucket
        )
        if mask.sum() < MIN_BUCKET_SIZE:
            continue
        clear = counts[mask & (grades == 0)]
        severe = counts[mask & (grades >= 3)]
        row = {
            "n": int(mask.sum()),
            "spearman": spearman(counts[mask], grades[mask]),
            "median_count_iga0": float(np.median(clear)) if clear.size else None,
            "median_count_iga34": (
                float(np.median(severe)) if severe.size else None
            ),
        }
        detector_rows[bucket] = row
        print(
            f"{bucket:<14}{row['n']:>5}{row['spearman']:>11.3f}"
            f"{(row['median_count_iga0'] or 0):>10.0f}"
            f"{(row['median_count_iga34'] or 0):>12.0f}"
        )
    report["detector"] = detector_rows

    tone_only = [v["spearman"] for k, v in detector_rows.items() if k != "ALL"]
    if tone_only:
        print(
            f"\nworst tone bucket: {min(tone_only):.3f}   "
            f"spread across tones: {max(tone_only) - min(tone_only):.3f}"
        )

    print("\n" + "=" * 74)
    print("CLASSIFIER: IGA grade accuracy (test split only)")
    print("=" * 74)
    print(f"{'skin tone':<14}{'n':>5}{'accuracy':>11}{'within 1':>11}{'MAE':>8}")
    predicted = np.array(
        [r["predicted"] if r["predicted"] else "" for r in records]
    )
    truth_ordinal = grades
    predicted_ordinal = np.array(
        [IGA_ORDINAL.get(p, -1) for p in predicted], dtype=float
    )

    classifier_rows = {}
    for bucket in BUCKET_ORDER + ("ALL",):
        mask = is_test if bucket == "ALL" else (is_test & (buckets == bucket))
        if mask.sum() < MIN_BUCKET_SIZE:
            continue
        error = np.abs(predicted_ordinal[mask] - truth_ordinal[mask])
        row = {
            "n": int(mask.sum()),
            "accuracy": float((error == 0).mean()),
            "within_one_grade": float((error <= 1).mean()),
            "mean_absolute_error": float(error.mean()),
        }
        classifier_rows[bucket] = row
        print(
            f"{bucket:<14}{row['n']:>5}{row['accuracy']:>11.3f}"
            f"{row['within_one_grade']:>11.3f}"
            f"{row['mean_absolute_error']:>8.2f}"
        )
    report["classifier"] = classifier_rows

    tone_accuracy = [
        v["accuracy"] for k, v in classifier_rows.items() if k != "ALL"
    ]
    if tone_accuracy:
        print(
            f"\nworst tone bucket: {min(tone_accuracy):.3f}   "
            f"spread across tones: "
            f"{max(tone_accuracy) - min(tone_accuracy):.3f}"
        )

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
