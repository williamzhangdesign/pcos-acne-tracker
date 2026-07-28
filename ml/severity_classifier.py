from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from PIL import Image

from ml.face_crop import crop_face_patch
from ml.features import extract_features

MODEL_PATH = Path(__file__).resolve().parent / "models" / "severity_classifier.joblib"

IGA_DESCRIPTIONS = {
    "IGA0": "Clear",
    "IGA1": "Almost clear",
    "IGA2": "Mild",
    "IGA3": "Moderate",
    "IGA4": "Severe",
}
_IGA_DESCRIPTIONS = IGA_DESCRIPTIONS

_GRADE_BY_DESCRIPTION = {
    description.lower(): grade
    for grade, description in IGA_DESCRIPTIONS.items()
}


def format_severity(grade: str) -> str:
    """Render a grade the one way it is stored, e.g. 'IGA2 (Mild)'."""
    return f"{grade} ({IGA_DESCRIPTIONS.get(grade, grade)})"


def severity_from_description(description: str) -> str:
    """
    Map a plain severity word onto its IGA grade. The IGA scale's own
    labels are Clear / Almost clear / Mild / Moderate / Severe, so a
    manually assigned 'Moderate' is IGA3 by definition rather than by any
    invented threshold. Unrecognised words are returned unchanged so the
    caller still stores something meaningful.
    """
    grade = _GRADE_BY_DESCRIPTION.get(description.strip().lower())
    return format_severity(grade) if grade else description

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@dataclass
class SeverityPrediction:
    grade: str
    description: str
    confidence: float
    analyzed_patch: Image.Image


def predict_severity(image: Image.Image) -> SeverityPrediction:
    """
    Predict an IGA-scale acne severity grade from a facial photo using the
    RandomForest classifier trained in ml/train_severity_model.py on the
    Roboflow Acne Severity Classification dataset. The training photos are
    close-up skin patches, not whole faces, so a face is first located and
    cropped down to a matching cheek-patch region before scoring (see
    ml/face_crop.py) - otherwise confidence collapses on realistic photos.
    """
    patch = crop_face_patch(image)

    model = _load_model()
    features = extract_features(patch).reshape(1, -1)

    grade = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return SeverityPrediction(
        grade=grade,
        description=_IGA_DESCRIPTIONS.get(grade, grade),
        confidence=confidence,
        analyzed_patch=patch,
    )
