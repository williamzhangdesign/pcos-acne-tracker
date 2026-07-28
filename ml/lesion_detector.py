from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ml.skin_tone import skin_mask, skin_region_mask

# Target: *inflammatory* ("active") acne lesions - papules, pustules and
# nodules - as distinguished in the acne literature from non-inflammatory
# comedones (blackheads and whiteheads), which are deliberately excluded.
# The discriminating sign is erythema, so detection keys on redness rather
# than on any dark spot.

_DENOISE_SIGMA = 1.0
_BASELINE_SIGMA_DIVISOR = 15.0

# Defaults below were selected by ml/tune_detector.py, which maximises the
# worst-case per-skin-tone rank correlation between detected lesion count
# and clinician IGA grade. See that script for the search and its results.
_STD_MULTIPLIER = 1.5
_MIN_ABSOLUTE_ERYTHEMA = 0.045  # log10(R/G) units
_MIN_LESION_AREA_FRACTION = 0.0002
_MAX_LESION_AREA_FRACTION = 0.02  # ignore blobs bigger than 2% of the image
_MIN_CIRCULARITY = 0.55  # 4*pi*area/perimeter^2; rejects thin shapes like
# hair strands, eyebrows and shadow edges

# Darkness-based comedone rejection is available but disabled by default.
# The tuning sweep found it lowers agreement with clinician grades in every
# pigmentation bucket, and does most damage in the darkest (-0.19 to -0.31
# Spearman, versus -0.05 to -0.14 in the lightest). Inflammatory acne in
# richly pigmented skin commonly presents as hyperpigmented or violaceous
# rather than bright red, so discarding blobs for being dark preferentially
# discards true lesions on darker skin. Comedones are instead excluded by
# the erythema gate above, which they do not trip.
_MAX_DARKNESS = float("inf")


@dataclass
class LesionDetectionResult:
    lesion_count: int
    annotated_image: Image.Image


@dataclass
class LesionMaps:
    erythema_elevation: np.ndarray
    darkness_elevation: np.ndarray
    skin: np.ndarray
    erythema_scale: float


def erythema_and_darkness_maps(bgr_array: np.ndarray) -> LesionMaps:
    """
    Build the per-pixel evidence used for lesion detection.

    Erythema uses a log reflectance ratio, log10(R/G): haemoglobin absorbs
    green light far more strongly than red, whereas melanin absorbs both
    broadly, so the ratio largely divides pigmentation out and leaves
    redness. This is what makes the detector usable across skin tones - a
    raw LAB a* threshold tracks melanin as much as it tracks inflammation.

    Both signals are expressed as an elevation above a locally blurred
    baseline, and the noise scale is a median-absolute-deviation taken over
    skin pixels only. Plain standard deviation over the whole frame is not
    usable here: on a full-face photo it is dominated by hair, eyes and
    background rather than by skin, and it is inflated by the very lesions
    being looked for.
    """
    channels = bgr_array.astype(np.float32) + 1.0
    green = channels[:, :, 1]
    red = channels[:, :, 2]
    erythema = np.log10(red) - np.log10(green)
    erythema = cv2.GaussianBlur(erythema, (0, 0), sigmaX=_DENOISE_SIGMA)

    lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    lightness = cv2.GaussianBlur(lightness, (0, 0), sigmaX=_DENOISE_SIGMA)

    height, width = erythema.shape
    sigma = max(height, width) / _BASELINE_SIGMA_DIVISOR
    erythema_baseline = cv2.GaussianBlur(erythema, (0, 0), sigmaX=sigma)
    lightness_baseline = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma)

    erythema_elevation = erythema - erythema_baseline
    # Noise floor is measured over strictly skin-coloured pixels, which
    # excludes the lesions and so keeps the floor clean; detection is then
    # allowed anywhere in the filled skin region, which includes them.
    on_skin = erythema_elevation[skin_mask(bgr_array)]
    # 1.4826 * MAD is the standard-deviation-equivalent for a normal noise
    # floor, but unlike the standard deviation it is not dragged upward by
    # the lesions themselves.
    scale = float(
        1.4826 * np.median(np.abs(on_skin - np.median(on_skin)))
    ) if on_skin.size else 0.0

    return LesionMaps(
        erythema_elevation=erythema_elevation,
        darkness_elevation=lightness_baseline - lightness,
        skin=skin_region_mask(bgr_array),
        erythema_scale=scale,
    )


def find_lesion_contours(
    bgr_array: np.ndarray,
    *,
    std_multiplier: float = _STD_MULTIPLIER,
    min_absolute_erythema: float = _MIN_ABSOLUTE_ERYTHEMA,
    min_area_fraction: float = _MIN_LESION_AREA_FRACTION,
    max_area_fraction: float = _MAX_LESION_AREA_FRACTION,
    min_circularity: float = _MIN_CIRCULARITY,
    max_darkness: float = _MAX_DARKNESS,
    maps: LesionMaps | None = None,
) -> list[np.ndarray]:
    """
    Find blob-shaped regions of raised erythema - the visible signature of
    inflammatory acne lesions. Blobs that are simply darker than the
    surrounding skin without accompanying redness are rejected as
    comedones. This is a heuristic approximation of lesion morphology, not
    clinical lesion identification.

    `maps` allows a caller to pass precomputed output of
    erythema_and_darkness_maps to avoid recomputing it (used when sweeping
    thresholds over the same image).
    """
    if maps is None:
        maps = erythema_and_darkness_maps(bgr_array)
    erythema_elevation = maps.erythema_elevation
    darkness_elevation = maps.darkness_elevation

    threshold = max(
        min_absolute_erythema, std_multiplier * maps.erythema_scale
    )
    mask = (
        (erythema_elevation > threshold) & maps.skin
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = erythema_elevation.shape
    image_area = height * width
    min_area = max(3.0, image_area * min_area_fraction)
    max_area = image_area * max_area_fraction

    accepted_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue

        if 4 * np.pi * area / (perimeter**2) < min_circularity:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        centre_x = int(np.clip(moments["m10"] / moments["m00"], 0, width - 1))
        centre_y = int(np.clip(moments["m01"] / moments["m00"], 0, height - 1))

        # Comedones read as a pronounced dark spot; inflammatory lesions do
        # not darken the skin anywhere near as much relative to erythema.
        if darkness_elevation[centre_y, centre_x] > max_darkness:
            continue

        accepted_contours.append(contour)

    return accepted_contours


def detect_lesions(image: Image.Image) -> LesionDetectionResult:
    """
    Count inflammatory acne lesions in a photo and draw an annotated
    overlay. See find_lesion_contours() for the detection heuristic.
    """
    rgb_image = image.convert("RGB")
    bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    accepted_contours = find_lesion_contours(bgr_array)

    annotated = bgr_array.copy()
    for contour in accepted_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(
            annotated,
            (int(x), int(y)),
            max(int(radius), 3),
            (0, 0, 255),
            2,
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return LesionDetectionResult(
        lesion_count=len(accepted_contours),
        annotated_image=Image.fromarray(annotated_rgb),
    )
