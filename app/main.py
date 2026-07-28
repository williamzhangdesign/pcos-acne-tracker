from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from food.trigger_detector import detect_dietary_triggers
from food.usda_client import USDAFoodDataError, search_usda_meal
from ml.severity import classify_acne_severity


DATA_FOLDER = PROJECT_ROOT / "data"
DATA_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATA_FOLDER / "tracker.db"


st.set_page_config(
    page_title="PCOS Acne Severity Tracker",
    page_icon="📱",
    layout="centered",
)


# ---------------------------------------------------------
# Database functions
# ---------------------------------------------------------

def connect_to_database() -> sqlite3.Connection:
    """Open the local SQLite database."""
    return sqlite3.connect(DATABASE_PATH)


def create_database_tables() -> None:
    """Create the database tables if they do not already exist."""
    with connect_to_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS acne_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                lesion_count INTEGER NOT NULL,
                severity TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS food_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                food_description TEXT NOT NULL,
                high_glycemic INTEGER NOT NULL,
                dairy INTEGER NOT NULL,
                refined_sugar INTEGER NOT NULL
            )
            """
        )


def save_acne_record(
    lesion_count: int,
    severity: str,
) -> None:
    """Save an acne result in the database."""
    with connect_to_database() as connection:
        connection.execute(
            """
            INSERT INTO acne_records (
                recorded_at,
                lesion_count,
                severity
            )
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                lesion_count,
                severity,
            ),
        )


def save_food_record(
    food_description: str,
    high_glycemic: bool,
    dairy: bool,
    refined_sugar: bool,
) -> None:
    """Save a food entry and its reviewed categories."""
    with connect_to_database() as connection:
        connection.execute(
            """
            INSERT INTO food_records (
                recorded_at,
                food_description,
                high_glycemic,
                dairy,
                refined_sugar
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                food_description,
                int(high_glycemic),
                int(dairy),
                int(refined_sugar),
            ),
        )


def load_acne_records() -> pd.DataFrame:
    """Read all acne records from the database."""
    with connect_to_database() as connection:
        return pd.read_sql_query(
            """
            SELECT
                recorded_at,
                lesion_count,
                severity
            FROM acne_records
            ORDER BY recorded_at
            """,
            connection,
        )


def load_food_records() -> pd.DataFrame:
    """Read all food records from the database."""
    with connect_to_database() as connection:
        return pd.read_sql_query(
            """
            SELECT
                recorded_at,
                food_description,
                high_glycemic,
                dairy,
                refined_sugar
            FROM food_records
            ORDER BY recorded_at
            """,
            connection,
        )


# ---------------------------------------------------------
# Capture page
# ---------------------------------------------------------

def show_capture_page() -> None:
    """Display the image upload and acne score page."""
    st.header("Capture and Acne Score")

    st.write(
        "Upload a facial image and record an acne lesion count. "
        "The live computer-vision API will be connected separately."
    )

    uploaded_image = st.file_uploader(
        "Choose a facial image",
        type=["jpg", "jpeg", "png"],
        key="facial_image_uploader",
    )

    if uploaded_image is None:
        st.info("Upload a JPG or PNG facial image to begin.")
        return

    st.image(
        uploaded_image,
        caption="Selected image",
        use_container_width=True,
    )

    st.info(
        "Prototype mode: the lesion count is currently entered manually. "
        "Do not present this value as an automatic computer-vision result."
    )

    lesion_count = st.number_input(
        "Lesion count",
        min_value=0,
        max_value=500,
        value=18,
        step=1,
        key="manual_lesion_count",
    )

    severity = classify_acne_severity(int(lesion_count))

    first_column, second_column = st.columns(2)

    with first_column:
        st.metric(
            label="Lesion count",
            value=int(lesion_count),
        )

    with second_column:
        st.metric(
            label="Estimated severity",
            value=severity,
        )

    if st.button(
        "Save acne result",
        type="primary",
        key="save_acne_result",
    ):
        save_acne_record(
            lesion_count=int(lesion_count),
            severity=severity,
        )

        st.success("The acne result was saved.")


# ---------------------------------------------------------
# Food log page
# ---------------------------------------------------------

def initialize_food_log_state() -> None:
    """Create Streamlit session-state values for the food page."""
    default_values = {
        "analyzed_food_description": "",
        "suggested_high_glycemic": False,
        "suggested_dairy": False,
        "suggested_refined_sugar": False,
        "matched_high_glycemic": (),
        "matched_dairy": (),
        "matched_refined_sugar": (),
        "food_saved_message": "",
        "usda_result": None,
        "usda_error": None,
        "reviewed_high_glycemic": False,
        "reviewed_dairy": False,
        "reviewed_refined_sugar": False,
        "reset_food_form": False,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def analyze_food_description(food_description: str) -> None:
    """Analyze a meal and save its suggestions in session state."""
    detection_result = detect_dietary_triggers(food_description)

    st.session_state.analyzed_food_description = food_description.strip()

    st.session_state.suggested_high_glycemic = (
        detection_result.high_glycemic
    )
    st.session_state.suggested_dairy = detection_result.dairy
    st.session_state.suggested_refined_sugar = (
        detection_result.refined_sugar
    )

    st.session_state.matched_high_glycemic = (
        detection_result.matched_high_glycemic
    )
    st.session_state.matched_dairy = (
        detection_result.matched_dairy
    )
    st.session_state.matched_refined_sugar = (
        detection_result.matched_refined_sugar
    )

    # Keep the review checkboxes synchronized with the newest analysis.
    st.session_state.reviewed_high_glycemic = (
        detection_result.high_glycemic
    )
    st.session_state.reviewed_dairy = detection_result.dairy
    st.session_state.reviewed_refined_sugar = (
        detection_result.refined_sugar
    )


def get_usda_result(food_description: str):
    """Return USDA meal results or a readable error message."""
    try:
        api_key = st.secrets["USDA_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None, "The USDA API key is not configured."

    try:
        result = search_usda_meal(
            meal_description=food_description,
            api_key=api_key,
        )
        return result, None

    except USDAFoodDataError as error:
        return None, str(error)

def clear_food_analysis_state() -> None:
    """Request a safe food-form reset on the next Streamlit rerun."""
    st.session_state.reset_food_form = True

def show_food_log_page() -> None:
    """Display automatic dietary-trigger suggestions and food logging."""
    initialize_food_log_state()

    # Reset widget-backed values before their widgets are created.
    # This avoids StreamlitAPIException after saving an entry.
    if st.session_state.reset_food_form:
        st.session_state.analyzed_food_description = ""

        st.session_state.suggested_high_glycemic = False
        st.session_state.suggested_dairy = False
        st.session_state.suggested_refined_sugar = False

        st.session_state.matched_high_glycemic = ()
        st.session_state.matched_dairy = ()
        st.session_state.matched_refined_sugar = ()

        st.session_state.usda_result = None
        st.session_state.usda_error = None

        st.session_state.reviewed_high_glycemic = False
        st.session_state.reviewed_dairy = False
        st.session_state.reviewed_refined_sugar = False
        st.session_state.food_description_input = ""

        st.session_state.reset_food_form = False

    st.header("Food Log")

    st.write(
        "Describe your meal in ordinary language. The app will suggest "
        "dietary categories for you to review before saving."
    )

    if st.session_state.food_saved_message:
        st.success(st.session_state.food_saved_message)
        st.session_state.food_saved_message = ""

    food_description = st.text_input(
        "What did you eat?",
        placeholder="Example: cereal with milk and sweetened coffee",
        key="food_description_input",
    )

    if st.button(
        "Analyze meal",
        key="analyze_meal_button",
    ):
        if not food_description.strip():
            st.warning("Please type a food or meal first.")
        else:
            analyze_food_description(food_description)

            usda_result, usda_error = get_usda_result(
                food_description
            )
            st.session_state.usda_result = usda_result
            st.session_state.usda_error = usda_error

    has_analysis = bool(
        st.session_state.analyzed_food_description
    )

    if not has_analysis:
        st.info(
            "Enter a meal description and select Analyze meal "
            "to receive automatic suggestions."
        )
        return

    st.subheader("Suggested dietary categories")

    detected_labels: list[str] = []

    if st.session_state.suggested_high_glycemic:
        detected_labels.append("High glycemic")

    if st.session_state.suggested_dairy:
        detected_labels.append("Dairy")

    if st.session_state.suggested_refined_sugar:
        detected_labels.append("Refined sugar")

    if detected_labels:
        st.success(
            "Suggested categories: "
            + ", ".join(detected_labels)
        )
    else:
        st.info(
            "No supported dietary category was detected. "
            "You can still select categories manually."
        )

    st.caption(
        "Please review these suggestions. The keyword detector "
        "may be incomplete or incorrect."
    )

    high_glycemic = st.checkbox(
        "High-glycemic food",
        key="reviewed_high_glycemic",
    )

    dairy = st.checkbox(
        "Contains dairy",
        key="reviewed_dairy",
    )

    refined_sugar = st.checkbox(
        "Contains refined sugar",
        key="reviewed_refined_sugar",
    )

    with st.expander("Why were these categories suggested?"):
        any_match_found = False

        if st.session_state.matched_high_glycemic:
            any_match_found = True
            st.write(
                "**High-glycemic matches:** "
                + ", ".join(
                    st.session_state.matched_high_glycemic
                )
            )

        if st.session_state.matched_dairy:
            any_match_found = True
            st.write(
                "**Dairy matches:** "
                + ", ".join(
                    st.session_state.matched_dairy
                )
            )

        if st.session_state.matched_refined_sugar:
            any_match_found = True
            st.write(
                "**Refined-sugar matches:** "
                + ", ".join(
                    st.session_state.matched_refined_sugar
                )
            )

        if not any_match_found:
            st.write(
                "No keywords from the current dietary lists "
                "were found."
            )

        st.caption(
            "This prototype uses transparent keyword matching. "
            "It does not calculate exact glycemic load or provide "
            "medical or nutritional advice."
        )

    st.subheader("USDA nutrition lookup")

    usda_result = st.session_state.usda_result
    usda_error = st.session_state.usda_error

    if usda_result is not None:
        st.success(
            f"USDA matched {len(usda_result.foods)} "
            "food component(s)."
        )

        nutrition_rows = []

        for food in usda_result.foods:
            nutrition_rows.append(
                {
                    "Your entry": food.original_food_text.title(),
                    "USDA match": food.description.title(),
                    "Data type": food.data_type,
                    "Calories": round(food.calories, 1),
                    "Carbohydrate (g)": round(
                        food.carbohydrate,
                        1,
                    ),
                    "Sugars (g)": round(food.sugars, 1),
                    "Protein (g)": round(food.protein, 1),
                    "Fat (g)": round(food.fat, 1),
                    "FDC ID": food.fdc_id,
                }
            )

        st.dataframe(
            pd.DataFrame(nutrition_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.write("**Approximate combined values**")

        first_metric, second_metric, third_metric = st.columns(3)

        with first_metric:
            st.metric(
                "Calories",
                f"{usda_result.total_calories:.0f}",
            )

        with second_metric:
            st.metric(
                "Carbohydrate",
                f"{usda_result.total_carbohydrate:.1f} g",
            )

        with third_metric:
            st.metric(
                "Total sugars",
                f"{usda_result.total_sugars:.1f} g",
            )

        fourth_metric, fifth_metric = st.columns(2)

        with fourth_metric:
            st.metric(
                "Protein",
                f"{usda_result.total_protein:.1f} g",
            )

        with fifth_metric:
            st.metric(
                "Fat",
                f"{usda_result.total_fat:.1f} g",
            )

        st.caption("Source: USDA FoodData Central.")

        st.warning(
            "These are approximate database matches. "
            "Serving definitions may differ between USDA records, "
            "so the combined values are not an exact meal calculation."
        )

    elif usda_error:
        st.warning(
            usda_error
            + " The dietary-trigger detector still works."
        )

    if st.button(
        "Save reviewed food entry",
        type="primary",
        key="save_reviewed_food_entry",
    ):
        save_food_record(
            food_description=(
                st.session_state.analyzed_food_description
            ),
            high_glycemic=high_glycemic,
            dairy=dairy,
            refined_sugar=refined_sugar,
        )

        st.session_state.food_saved_message = (
            "The food entry and reviewed categories were saved."
        )

        clear_food_analysis_state()

        st.rerun()


# ---------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------

def show_dashboard_page() -> None:
    """Display saved acne and dietary records."""
    st.header("Dashboard")

    acne_data = load_acne_records()
    food_data = load_food_records()

    first_column, second_column = st.columns(2)

    with first_column:
        st.metric(
            label="Acne records",
            value=len(acne_data),
        )

    with second_column:
        st.metric(
            label="Food records",
            value=len(food_data),
        )

    st.subheader("Acne trend")

    if acne_data.empty:
        st.info("No acne records have been saved yet.")
    else:
        acne_data["recorded_at"] = pd.to_datetime(
            acne_data["recorded_at"]
        )

        chart_data = acne_data.set_index("recorded_at")[
            ["lesion_count"]
        ]

        st.line_chart(chart_data)

        acne_display = acne_data.copy()
        acne_display["recorded_at"] = (
            acne_display["recorded_at"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

        acne_display = acne_display.rename(
            columns={
                "recorded_at": "Recorded at",
                "lesion_count": "Lesion count",
                "severity": "Severity",
            }
        )

        st.dataframe(
            acne_display,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Dietary category summary")

    if food_data.empty:
        st.info("No food entries have been saved yet.")
    else:
        trigger_summary = pd.DataFrame(
            {
                "Category": [
                    "High glycemic",
                    "Dairy",
                    "Refined sugar",
                ],
                "Number of entries": [
                    int(food_data["high_glycemic"].sum()),
                    int(food_data["dairy"].sum()),
                    int(food_data["refined_sugar"].sum()),
                ],
            }
        ).set_index("Category")

        st.bar_chart(
            trigger_summary,
            horizontal=True,
        )

        food_display = food_data.copy()

        food_display["recorded_at"] = pd.to_datetime(
            food_display["recorded_at"]
        ).dt.strftime("%Y-%m-%d %H:%M")

        food_display["high_glycemic"] = food_display[
            "high_glycemic"
        ].map(
            {
                1: "Yes",
                0: "No",
            }
        )

        food_display["dairy"] = food_display["dairy"].map(
            {
                1: "Yes",
                0: "No",
            }
        )

        food_display["refined_sugar"] = food_display[
            "refined_sugar"
        ].map(
            {
                1: "Yes",
                0: "No",
            }
        )

        food_display = food_display.rename(
            columns={
                "recorded_at": "Recorded at",
                "food_description": "Food description",
                "high_glycemic": "High glycemic",
                "dairy": "Dairy",
                "refined_sugar": "Refined sugar",
            }
        )

        st.dataframe(
            food_display,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "This prototype supports self-reflection only. "
        "It does not diagnose PCOS, acne, or dietary causes."
    )


# ---------------------------------------------------------
# Profile page
# ---------------------------------------------------------

def show_profile_page() -> None:
    """Display basic profile and prototype settings."""
    st.header("Profile and Settings")

    st.text_input(
        "Display name",
        key="display_name",
    )

    st.selectbox(
        "Skin tone category for future model evaluation",
        [
            "Prefer not to answer",
            "Fitzpatrick I",
            "Fitzpatrick II",
            "Fitzpatrick III",
            "Fitzpatrick IV",
            "Fitzpatrick V",
            "Fitzpatrick VI",
        ],
        key="skin_tone_category",
    )

    st.checkbox(
        "Enable a daily logging reminder",
        key="daily_reminder",
    )

    st.warning(
        "This project is an educational prototype and is not "
        "a medical device."
    )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main() -> None:
    """Start the Streamlit application."""
    create_database_tables()

    st.title("PCOS Acne Severity Tracker")

    st.caption(
        "A mobile-oriented acne and dietary logging prototype"
    )

    selected_page = st.sidebar.radio(
        "Choose a page",
        [
            "Capture",
            "Food Log",
            "Dashboard",
            "Profile",
        ],
    )

    if selected_page == "Capture":
        show_capture_page()

    elif selected_page == "Food Log":
        show_food_log_page()

    elif selected_page == "Dashboard":
        show_dashboard_page()

    else:
        show_profile_page()


if __name__ == "__main__":
    main()