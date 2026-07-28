from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Individual Typology Angle (ITA), the standard objective measure of
# constitutive skin colour in dermatology literature (Chardon et al. 1991;
# Del Bino & Bernerd 2013):
#
#     ITA = arctan((L* - 50) / b*) * 180/pi
#
# Bucket boundaries below follow the conventional six-category split, which
# corresponds approximately - not exactly - to Fitzpatrick phototypes I-VI.
# ITA is a measure of pigmentation, not of self-reported race or phototype,
# and should be reported as such.
_ITA_BUCKETS = (
    (55.0, "very_light"),
    (41.0, "light"),
    (28.0, "intermediate"),
    (10.0, "tan"),
    (-30.0, "brown"),
)
_DARKEST_BUCKET = "dark"

# Classic YCrCb skin-chrominance gate. Note this rule was itself derived
# largely on lighter skin and is known to under-select very dark skin, so a
# fallback to the whole frame is used when it selects too little.
_MIN_SKIN_FRACTION = 0.02


def skin_mask(bgr_array: np.ndarray) -> np.ndarray:
    """
    Boolean mask of skin-like pixels, used to keep hair, eyes, clothing and
    background out of skin statistics. Falls back to the whole frame when it
    selects implausibly little, since the underlying chrominance rule is
    known to under-select very dark skin.
    """
    ycrcb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 176) & (cb >= 77) & (cb <= 130)
    if mask.mean() < _MIN_SKIN_FRACTION:
        return np.ones(mask.shape, dtype=bool)
    return mask


def skin_region_mask(bgr_array: np.ndarray) -> np.ndarray:
    """
    Mask of the skin *region* rather than of skin-coloured pixels.

    Inflamed lesions are chromatically outside the nominal skin gate - that
    is what makes them lesions - so skin_mask() punches holes exactly where
    the interesting pixels are. Morphologically closing the mask fills those
    lesion-sized holes while still excluding large non-skin areas such as
    hair, clothing and background.
    """
    mask = skin_mask(bgr_array).astype(np.uint8)
    height, width = mask.shape
    radius = max(9, int(max(height, width) * 0.05)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def _skin_mask(bgr_array: np.ndarray) -> np.ndarray:
    return skin_mask(bgr_array)


def estimate_ita(image: Image.Image) -> float:
    """
    Estimate the Individual Typology Angle of the skin in a photo, in
    degrees. Higher is lighter. Uses the median over skin-like pixels so
    that background, hair and clothing do not dominate.
    """
    rgb_image = image.convert("RGB")
    bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
    # OpenCV packs 8-bit LAB as L in [0,255] and a/b offset by 128.
    l_star = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    b_star = lab[:, :, 2].astype(np.float32) - 128.0

    mask = skin_mask(bgr_array)
    ita = np.degrees(np.arctan2(l_star - 50.0, b_star))
    return float(np.median(ita[mask]))


def ita_to_bucket(ita: float) -> str:
    """Map an ITA angle onto a named pigmentation category."""
    for lower_bound, name in _ITA_BUCKETS:
        if ita > lower_bound:
            return name
    return _DARKEST_BUCKET
