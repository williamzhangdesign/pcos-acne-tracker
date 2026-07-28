from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_MODEL_PATH = (
    Path(__file__).resolve().parent / "cascades" / "face_detection_yunet.onnx"
)
_MIN_EYE_DISTANCE = 5.0
_PATCH_SIZE_MULTIPLIER = 1.3
_MIN_PATCH_SIZE = 20

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(str(_MODEL_PATH), "", (320, 320))
    return _detector


def crop_face_patch(image: Image.Image) -> Image.Image:
    """
    Find a face in a normal photo and crop a close-up patch of cheek skin
    around it, matching the framing of the macro skin-patch photos the
    severity classifier was trained on (see ml/train_severity_model.py).

    If no face is found, the photo is returned unchanged rather than
    guessing a crop: it may already be a close-up patch in the training
    format (which naturally has no detectable face), and an unvalidated
    guessed crop risks doing more harm than good.
    """
    rgb_image = image.convert("RGB")
    bgr_array = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    height, width = bgr_array.shape[:2]

    detector = _get_detector()
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr_array)

    if faces is None or len(faces) == 0:
        return rgb_image

    # face row: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
    #            x_rcm, y_rcm, x_lcm, y_lcm, score]
    face = max(faces, key=lambda row: row[2] * row[3])
    x_re, y_re, x_le, y_le = face[4], face[5], face[6], face[7]
    x_rcm, y_rcm = face[10], face[11]

    eye_distance = float(np.hypot(x_le - x_re, y_le - y_re))
    if eye_distance < _MIN_EYE_DISTANCE:
        return rgb_image

    # Midpoint between the (right) eye and the (right) mouth corner lands
    # roughly on the cheek, without needing a full landmark model.
    cheek_x = (x_re + x_rcm) / 2
    cheek_y = (y_re + y_rcm) / 2
    patch_size = max(eye_distance * _PATCH_SIZE_MULTIPLIER, _MIN_PATCH_SIZE)

    left = int(np.clip(cheek_x - patch_size / 2, 0, width))
    top = int(np.clip(cheek_y - patch_size / 2, 0, height))
    right = int(np.clip(cheek_x + patch_size / 2, 0, width))
    bottom = int(np.clip(cheek_y + patch_size / 2, 0, height))

    if right - left < _MIN_PATCH_SIZE or bottom - top < _MIN_PATCH_SIZE:
        return rgb_image

    return rgb_image.crop((left, top, right, bottom))
