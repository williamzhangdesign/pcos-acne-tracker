from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DietaryTriggerResult:
    """
    Stores the dietary categories suggested by the keyword detector.

    The result is intended for user review. It is not a medical conclusion.
    """

    high_glycemic: bool
    dairy: bool
    refined_sugar: bool
    matched_high_glycemic: tuple[str, ...]
    matched_dairy: tuple[str, ...]
    matched_refined_sugar: tuple[str, ...]

    @property
    def detected_labels(self) -> list[str]:
        """Return readable labels for the detected categories."""
        labels: list[str] = []

        if self.high_glycemic:
            labels.append("High glycemic")

        if self.dairy:
            labels.append("Dairy")

        if self.refined_sugar:
            labels.append("Refined sugar")

        return labels


# These lists are deliberately simple and transparent.
# They can be expanded later or replaced by a nutrition API.
HIGH_GLYCEMIC_TERMS = {
    "white bread",
    "white rice",
    "sweet cereal",
    "breakfast cereal",
    "corn flakes",
    "pancake",
    "pancakes",
    "waffle",
    "waffles",
    "french fries",
    "fries",
    "potato chips",
    "chips",
    "mashed potato",
    "instant noodles",
    "noodles",
    "cake",
    "cupcake",
    "cookies",
    "cookie",
    "donut",
    "doughnut",
    "pastry",
    "pastries",
    "candy",
    "soda",
    "soft drink",
    "sweetened drink",
    "energy drink",
    "coke",
    "pepsi",
    "sprite",
    "fanta",
    "mountain dew",
    "coke",
    "regular coke",
    "cola",
    "regular soda",
    "sandwich bread",
    "white bread sandwich",
 
}

DAIRY_TERMS = {
    "milk",
    "whole milk",
    "skim milk",
    "cheese",
    "cream cheese",
    "yogurt",
    "greek yogurt",
    "ice cream",
    "butter",
    "cream",
    "sour cream",
    "whipped cream",
    "milkshake",
    "latte",
}

REFINED_SUGAR_TERMS = {
    "sugar",
    "white sugar",
    "brown sugar",
    "syrup",
    "corn syrup",
    "sweetened",
    "sweet coffee",
    "sweet tea",
    "soda",
    "soft drink",
    "energy drink",
    "candy",
    "chocolate bar",
    "cake",
    "cupcake",
    "cookie",
    "cookies",
    "donut",
    "doughnut",
    "ice cream",
    "pastry",
    "pastries",
    "coke",
    "pepsi",
    "sprite",
    "fanta",
    "mountain dew",
    "regular coke",
    "cola",
    "regular soda",
}


def normalize_food_text(food_text: str) -> str:
    """
    Convert food text into a simpler form for matching.

    Example:
        "Cereal, Milk & SWEET Coffee!"
        becomes
        "cereal milk sweet coffee"
    """
    lowered_text = food_text.lower().strip()
    cleaned_text = re.sub(r"[^a-z0-9\s-]", " ", lowered_text)
    return re.sub(r"\s+", " ", cleaned_text)


def find_matching_terms(
    normalized_text: str,
    terms: set[str],
) -> tuple[str, ...]:
    """
    Return the food terms found in the description.

    Word boundaries are used so that a short word does not accidentally
    match part of a different word.
    """
    matches: list[str] = []

    for term in sorted(terms):
        pattern = rf"\b{re.escape(term)}\b"

        if re.search(pattern, normalized_text):
            matches.append(term)

    return tuple(matches)


def detect_dietary_triggers(food_text: str) -> DietaryTriggerResult:
    """
    Suggest dietary categories from a free-text meal description.

    The user should always review the suggestions before saving them.
    """
    if not food_text or not food_text.strip():
        return DietaryTriggerResult(
            high_glycemic=False,
            dairy=False,
            refined_sugar=False,
            matched_high_glycemic=(),
            matched_dairy=(),
            matched_refined_sugar=(),
        )

    normalized_text = normalize_food_text(food_text)

    high_glycemic_matches = find_matching_terms(
        normalized_text,
        HIGH_GLYCEMIC_TERMS,
    )
    dairy_matches = find_matching_terms(
        normalized_text,
        DAIRY_TERMS,
    )
    refined_sugar_matches = find_matching_terms(
        normalized_text,
        REFINED_SUGAR_TERMS,
    )

    return DietaryTriggerResult(
        high_glycemic=bool(high_glycemic_matches),
        dairy=bool(dairy_matches),
        refined_sugar=bool(refined_sugar_matches),
        matched_high_glycemic=high_glycemic_matches,
        matched_dairy=dairy_matches,
        matched_refined_sugar=refined_sugar_matches,
    )