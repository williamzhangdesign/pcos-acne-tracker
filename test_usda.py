import streamlit as st

from food.usda_client import (
    USDAFoodDataError,
    search_usda_food,
)


test_food = "white rice"

try:
    result = search_usda_food(
        food_description=test_food,
        api_key=st.secrets["USDA_API_KEY"],
    )

    print("USDA test succeeded")
    print("-------------------")
    print("Matched food:", result.description)
    print("FDC ID:", result.fdc_id)
    print("Data type:", result.data_type)
    print("Calories:", result.calories)
    print("Carbohydrate:", result.carbohydrate)
    print("Sugars:", result.sugars)
    print("Protein:", result.protein)
    print("Fat:", result.fat)

except KeyError:
    print("USDA_API_KEY was not found in secrets.toml.")

except USDAFoodDataError as error:
    print("USDA error:", error)