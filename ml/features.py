from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ml.lesion_detector import find_lesion_contours

_STANDARD_SIZE = (256, 256)
_HSV_BINS = (8, 4, 4)


def _blob_stats(bgr_array: np.ndarray) -> tuple[float, float, float, float]:
    """
    Reuse the lesion-blob heuristic from ml/lesion_detector.py (redder-than-
    local-skin for inflamed papules, darker-than-local-skin for comedones)
    to turn "how much does this photo look like it has lesions" into a few
    numeric summaries: fraction of flagged pixels, blob count, mean blob
    area, max blob area.
    """
    contours = find_lesion_contours(bgr_array)
    areas = [cv2.contourArea(contour) for contour in contours]

    height, width = bgr_array.shape[:2]
    flagged_fraction = float(sum(areas) / (height * width)) if areas else 0.0
    blob_count = float(len(areas))
    mean_area = float(np.mean(areas)) if areas else 0.0
    max_area_value = float(np.max(areas)) if areas else 0.0

    return flagged_fraction, blob_count, mean_area, max_area_value


def extract_features(image: Image.Image) -> np.ndarray:
    """
    Turn a facial photo into a fixed-length numeric feature vector for a
    classical ML severity classifier: color histograms, redness/inflammation
    summaries, and a texture-roughness signal.
    """
    rgb_image = image.convert("RGB").resize(_STANDARD_SIZE)
    bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        list(_HSV_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()

    lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = (
        lab[:, :, i].astype(np.float32) for i in range(3)
    )
    color_stats = np.array(
        [
            l_channel.mean(), l_channel.std(),
            a_channel.mean(), a_channel.std(),
            b_channel.mean(), b_channel.std(),
        ]
    )

    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    texture_roughness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())

    flagged_fraction, blob_count, mean_area, max_area_value = _blob_stats(
        bgr_array
    )

    extra = np.array(
        [
            texture_roughness,
            contrast,
            flagged_fraction,
            blob_count,
            mean_area,
            max_area_value,
        ]
    )

    return np.concatenate([hist, color_stats, extra]).astype(np.float32)
