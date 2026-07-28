"""
Tune the inflammatory-lesion detector thresholds in ml/lesion_detector.py.

There is no per-image lesion-count ground truth available, so the objective
is built from the labels that do exist. IGA severity is ordinally driven by
inflammatory lesion burden, so a useful detector must produce counts that
rise monotonically with IGA grade, and a detector that is *reliable across
skin tones* must do so about equally well at every level of pigmentation.

The search therefore maximises the worst-case per-skin-tone Spearman rank
correlation between detected count and IGA grade (a maximin / Rawlsian
objective), which cannot be won by doing well on light skin alone.

Only train+valid are used; the test split is held back for final reporting.

Usage: python3 ml/tune_detector.py
"""

from __future__ import annotations

import glob
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.lesion_detector import (  # noqa: E402
    erythema_and_darkness_maps,
    find_lesion_contours,
)
from ml.skin_tone import estimate_ita, ita_to_bucket  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "acne-severity"
RESULTS_PATH = PROJECT_ROOT / "ml" / "models" / "detector_tuning.json"

IGA_ORDINAL = {"IGA0": 0, "IGA1": 1, "IGA2": 2, "IGA3": 3, "IGA4": 4}
MAX_PER_CLASS = 110
MIN_BUCKET_SIZE = 25
RANDOM_SEED = 42

SEARCH_SPACE = {
    # Multiples of the robust (MAD) erythema noise scale, which measures
    # ~0.019-0.025 on skin across every severity grade in this dataset.
    "std_multiplier": [1.5, 2.0, 2.5, 3.0],
    "min_absolute_erythema": [0.010, 0.025, 0.045],
    "min_area_fraction": [0.00005, 0.0002, 0.0006],
    "min_circularity": [0.35, 0.55],
    "max_darkness": [8.0, 20.0, float("inf")],
}


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata (no scipy dependency)."""
    sorter = np.argsort(values, kind="mergesort")
    inverse = np.empty(len(values), dtype=int)
    inverse[sorter] = np.arange(len(values))
    sorted_values = values[sorter]
    is_new = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = is_new.cumsum()[inverse]
    counts = np.r_[np.nonzero(is_new)[0], len(values)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rank_x, rank_y = _rank(x), _rank(y)
    if rank_x.std() == 0 or rank_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def collect_images() -> list[tuple[str, str]]:
    """One file per source image (Roboflow ships ~3 augmented copies each)."""
    by_base: dict[str, tuple[str, str]] = {}
    for split in ("train", "valid"):
        for path in glob.glob(str(DATA_DIR / split / "*" / "*")):
            label = Path(path).parent.name
            base = re.sub(
                r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", "", path, flags=re.I
            )
            by_base.setdefault(base, (path, label))

    per_class = defaultdict(list)
    for path, label in by_base.values():
        per_class[label].append(path)

    rng = np.random.default_rng(RANDOM_SEED)
    sampled = []
    for label, paths in sorted(per_class.items()):
        paths = sorted(paths)
        if len(paths) > MAX_PER_CLASS:
            chosen = rng.choice(len(paths), MAX_PER_CLASS, replace=False)
            paths = [paths[i] for i in chosen]
        sampled.extend((p, label) for p in paths)
    return sampled


def main() -> None:
    images = collect_images()
    combos = [
        dict(zip(SEARCH_SPACE, values))
        for values in itertools.product(*SEARCH_SPACE.values())
    ]
    print(f"{len(images)} images x {len(combos)} parameter combinations")

    grades = np.zeros(len(images), dtype=float)
    buckets: list[str] = []
    counts = np.zeros((len(images), len(combos)), dtype=float)

    for row, (path, label) in enumerate(images):
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            bucket = ita_to_bucket(estimate_ita(rgb_image))
        bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        maps = erythema_and_darkness_maps(bgr_array)

        grades[row] = IGA_ORDINAL[label]
        buckets.append(bucket)
        for column, params in enumerate(combos):
            counts[row, column] = len(
                find_lesion_contours(bgr_array, maps=maps, **params)
            )

        if (row + 1) % 100 == 0:
            print(f"  processed {row + 1}/{len(images)}")

    bucket_array = np.array(buckets)
    tone_groups = [
        (name, bucket_array == name)
        for name in sorted(set(buckets))
        if (bucket_array == name).sum() >= MIN_BUCKET_SIZE
    ]
    print(
        "tone buckets scored: "
        + ", ".join(f"{n} (n={int(m.sum())})" for n, m in tone_groups)
    )

    results = []
    for column, params in enumerate(combos):
        column_counts = counts[:, column]
        per_bucket = {
            name: spearman(column_counts[mask], grades[mask])
            for name, mask in tone_groups
        }
        finite = [v for v in per_bucket.values() if np.isfinite(v)]
        if len(finite) < len(tone_groups):
            continue
        results.append(
            {
                "params": params,
                "overall_spearman": spearman(column_counts, grades),
                "per_bucket_spearman": per_bucket,
                "worst_bucket_spearman": min(finite),
                "bucket_spread": max(finite) - min(finite),
                "median_count_iga0": float(
                    np.median(column_counts[grades == 0])
                ),
                "median_count_iga4": float(
                    np.median(column_counts[grades == 4])
                ),
            }
        )

    results.sort(
        key=lambda r: (r["worst_bucket_spearman"], r["overall_spearman"]),
        reverse=True,
    )

    print("\nTop 10 by worst-case per-tone Spearman:")
    for entry in results[:10]:
        params = entry["params"]
        print(
            f"  worst={entry['worst_bucket_spearman']:.3f} "
            f"overall={entry['overall_spearman']:.3f} "
            f"spread={entry['bucket_spread']:.3f} "
            f"IGA0med={entry['median_count_iga0']:.0f} "
            f"IGA4med={entry['median_count_iga4']:.0f}  "
            f"k={params['std_multiplier']} "
            f"floor={params['min_absolute_erythema']} "
            f"area={params['min_area_fraction']} "
            f"circ={params['min_circularity']} "
            f"dark={params['max_darkness']}"
        )

    best = results[0]
    print("\nBest parameters:")
    print(json.dumps(best, indent=2, default=str))

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results[:50], indent=2, default=str))
    print(f"\nWrote top-50 results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
