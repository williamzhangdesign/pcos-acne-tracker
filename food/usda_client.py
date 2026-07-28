from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests


USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

DATA_TYPE_PRIORITY = {
    "Survey (FNDDS)": 0,
    "Foundation": 1,
    "SR Legacy": 2,
    "Branded": 3,
}


class USDAFoodDataError(RuntimeError):
    """Raised when a USDA FoodData Central request fails."""


@dataclass(frozen=True)
class USDAFoodResult:
    """One food item returned by USDA."""

    original_food_text: str
    fdc_id: int
    description: str
    data_type: str
    calories: float
    carbohydrate: float
    sugars: float
    protein: float
    fat: float


@dataclass(frozen=True)
class USDAMealResult:
    """Several USDA food matches from one meal description."""

    foods: tuple[USDAFoodResult, ...]

    @property
    def total_calories(self) -> float:
        return sum(food.calories for food in self.foods)

    @property
    def total_carbohydrate(self) -> float:
        return sum(food.carbohydrate for food in self.foods)

    @property
    def total_sugars(self) -> float:
        return sum(food.sugars for food in self.foods)

    @property
    def total_protein(self) -> float:
        return sum(food.protein for food in self.foods)

    @property
    def total_fat(self) -> float:
        return sum(food.fat for food in self.foods)


def _safe_float(value: Any) -> float:
    """Convert a value to float safely."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _find_nutrient(
    nutrients: list[dict[str, Any]],
    possible_names: tuple[str, ...],
) -> float:
    """Find one nutrient using possible USDA nutrient names."""

    for nutrient in nutrients:
        nutrient_name = str(
            nutrient.get("nutrientName", "")
        ).strip().lower()

        if nutrient_name in possible_names:
            return _safe_float(nutrient.get("value"))

    return 0.0


def split_meal_description(meal_description: str) -> list[str]:
    """
    Split a meal sentence into smaller food phrases.

    Example:
    white rice with butter chicken and sweetened coffee

    becomes:
    white rice
    butter chicken
    sweetened coffee
    """

    cleaned_description = meal_description.strip()

    if not cleaned_description:
        return []

    pieces = re.split(
        r"\s+(?:with|and|plus)\s+|[,;]",
        cleaned_description,
        flags=re.IGNORECASE,
    )

    food_items = [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]

    # Avoid sending too many API requests from one entry.
    return food_items[:5]


def search_usda_food(
    food_description: str,
    api_key: str,
    timeout_seconds: int = 15,
) -> USDAFoodResult:
    """Search USDA for one food phrase."""

    cleaned_description = food_description.strip()

    if not cleaned_description:
        raise ValueError("Food description cannot be empty.")

    if not api_key.strip():
        raise USDAFoodDataError("The USDA API key is missing.")

    request_body = {
        "query": cleaned_description,
        "pageSize": 10,
        "dataType": [
            "Survey (FNDDS)",
            "Foundation",
            "SR Legacy",
            "Branded",
        ],
    }

    try:
        response = requests.post(
            USDA_SEARCH_URL,
            params={"api_key": api_key.strip()},
            json=request_body,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise USDAFoodDataError(
            "The USDA service could not be reached."
        ) from error

    if response.status_code == 403:
        raise USDAFoodDataError(
            "The USDA API key was rejected."
        )

    if response.status_code == 429:
        raise USDAFoodDataError(
            "The USDA request limit was reached."
        )

    if not response.ok:
        raise USDAFoodDataError(
            f"USDA returned status code {response.status_code}."
        )

    try:
        response_data = response.json()
    except ValueError as error:
        raise USDAFoodDataError(
            "USDA returned an unreadable response."
        ) from error

    foods = response_data.get("foods", [])

    if not foods:
        raise USDAFoodDataError(
            f'No USDA match was found for "{cleaned_description}".'
        )

    # Prefer generic USDA records before commercial branded products.
    foods.sort(
        key=lambda food: DATA_TYPE_PRIORITY.get(
            str(food.get("dataType", "")),
            99,
        )
    )

    selected_food = foods[0]
    nutrients = selected_food.get("foodNutrients", [])

    calories = _find_nutrient(
        nutrients,
        (
            "energy",
            "energy (atwater general factors)",
            "energy (atwater specific factors)",
        ),
    )

    carbohydrate = _find_nutrient(
        nutrients,
        (
            "carbohydrate, by difference",
            "carbohydrate, by summation",
        ),
    )

    sugars = _find_nutrient(
        nutrients,
        (
            "sugars, total including nlea",
            "sugars, total",
            "total sugars",
        ),
    )

    protein = _find_nutrient(
        nutrients,
        ("protein",),
    )

    fat = _find_nutrient(
        nutrients,
        (
            "total lipid (fat)",
            "total fat",
        ),
    )

    return USDAFoodResult(
        original_food_text=cleaned_description,
        fdc_id=int(selected_food.get("fdcId", 0)),
        description=str(
            selected_food.get("description", "Unknown food")
        ),
        data_type=str(
            selected_food.get("dataType", "Unknown")
        ),
        calories=calories,
        carbohydrate=carbohydrate,
        sugars=sugars,
        protein=protein,
        fat=fat,
    )


def search_usda_meal(
    meal_description: str,
    api_key: str,
) -> USDAMealResult:
    """Split a meal and search USDA for each food component."""

    food_items = split_meal_description(meal_description)

    if not food_items:
        raise USDAFoodDataError(
            "No searchable food items were found."
        )

    results: list[USDAFoodResult] = []
    errors: list[str] = []

    for food_item in food_items:
        try:
            result = search_usda_food(
                food_description=food_item,
                api_key=api_key,
            )
            results.append(result)
        except USDAFoodDataError:
            errors.append(food_item)

    if not results:
        raise USDAFoodDataError(
            "USDA could not match any part of this meal."
        )

    return USDAMealResult(foods=tuple(results))