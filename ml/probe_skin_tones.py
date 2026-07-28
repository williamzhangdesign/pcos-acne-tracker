"""
Regression probe for ml/lesion_detector.py across skin tones.

Builds synthetic skin at six pigmentation levels and plants either
inflammatory papules (which raise the R/G ratio, as haemoglobin does) or
comedones (which darken all channels, leaving the ratio intact). A correct
detector finds the papules at every tone and none of the comedones.

This caught a real defect: gating detection on skin-coloured *pixels*
deleted inflamed pixels, which are by definition off the skin locus, and
silently zeroed recall at some tones.

Usage: python3 ml/probe_skin_tones.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from ml.lesion_detector import detect_lesions
from ml.skin_tone import estimate_ita, ita_to_bucket

TONES = {
    "very_light": (245, 220, 205),
    "light": (230, 195, 175),
    "intermediate": (205, 165, 140),
    "tan": (175, 130, 100),
    "brown": (130, 90, 65),
    "dark": (85, 58, 45),
}
N_LESIONS = 20
SIZE = 512


def build(base, kind, seed):
    rng = np.random.default_rng(seed)
    arr = np.full((SIZE, SIZE, 3), base, dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    arr += (np.sin(xx / 90.0) * 7 + np.cos(yy / 70.0) * 7)[..., None]
    arr += rng.normal(0, 4, arr.shape)

    placed = []
    for _ in range(N_LESIONS if kind != "clear" else 0):
        while True:
            x, y = rng.integers(40, SIZE - 40, 2)
            if all((x - px) ** 2 + (y - py) ** 2 > 40 ** 2 for px, py in placed):
                break
        placed.append((x, y))
        r = int(rng.integers(5, 10))
        yy2, xx2 = np.mgrid[0:SIZE, 0:SIZE]
        d = np.sqrt((xx2 - x) ** 2 + (yy2 - y) ** 2)
        w = np.clip(1.0 - d / r, 0, 1)[..., None]
        if kind == "inflammatory":
            # Papule: raises R relative to G (haemoglobin), tone-relative.
            target = np.array([base[0] * 0.80, base[1] * 0.72, min(base[2] * 1.18 + 25, 255)])
        else:
            # Comedone: darkens all channels, preserving the R/G ratio.
            target = np.array([base[0] * 0.32, base[1] * 0.32, base[2] * 0.34])
        arr = arr * (1 - w) + target * w

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


print(f"{'tone':<14}{'est ITA':>9}{'bucket':>14}"
      f"{'clear':>8}{'inflam':>9}{'comedone':>10}")
print("-" * 64)
rows = []
for name, base_rgb in TONES.items():
    base_bgr = (base_rgb[2], base_rgb[1], base_rgb[0])
    results = {}
    for kind in ("clear", "inflammatory", "comedone"):
        img = build(base_bgr, kind, seed=abs(hash(name)) % 1000)
        results[kind] = detect_lesions(img).lesion_count
        if kind == "clear":
            ita = estimate_ita(img)
    rows.append((name, results))
    print(f"{name:<14}{ita:>9.1f}{ita_to_bucket(ita):>14}"
          f"{results['clear']:>8}{results['inflammatory']:>9}"
          f"{results['comedone']:>10}")

print("-" * 64)
print(f"(planted {N_LESIONS} lesions in the inflam and comedone columns; "
      f"clear column should be ~0, comedone column should be ~0)")
inflam = [r['inflammatory'] for _, r in rows]
com = [r['comedone'] for _, r in rows]
clear = [r['clear'] for _, r in rows]
print(f"\ninflammatory recall: min={min(inflam)}/{N_LESIONS} "
      f"max={max(inflam)}/{N_LESIONS}")
print(f"comedone false positives: min={min(com)} max={max(com)}")
print(f"clear-skin false positives: min={min(clear)} max={max(clear)}")
