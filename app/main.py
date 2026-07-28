from __future__ import annotations

import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from food.trigger_detector import detect_dietary_triggers
from food.usda_client import USDAFoodDataError, search_usda_meal
from ml.lesion_detector import detect_lesions
from ml.severity import classify_acne_severity
from ml.severity_classifier import (
    format_severity,
    predict_severity,
    severity_from_description,
)


DATA_FOLDER = PROJECT_ROOT / "data"
DATA_FOLDER.mkdir(exist_ok=True)

# Uploaded face photos are saved here with unique filenames.
IMAGES_FOLDER = DATA_FOLDER / "images"
IMAGES_FOLDER.mkdir(exist_ok=True)

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
                severity TEXT NOT NULL,
                image_path TEXT,
                source TEXT NOT NULL DEFAULT 'manual',
                confidence REAL
            )
            """
        )

        # Guarded migration for any acne_records table created before the
        # image_path/source/confidence columns existed.
        for migration in (
            "ALTER TABLE acne_records ADD COLUMN image_path TEXT",
            "ALTER TABLE acne_records ADD COLUMN source "
            "TEXT NOT NULL DEFAULT 'manual'",
            "ALTER TABLE acne_records ADD COLUMN confidence REAL",
        ):
            try:
                connection.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists.

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
    image_path: str | None = None,
    source: str = "manual",
    confidence: float | None = None,
) -> None:
    """Save an acne result in the database."""
    with connect_to_database() as connection:
        connection.execute(
            """
            INSERT INTO acne_records (
                recorded_at,
                lesion_count,
                severity,
                image_path,
                source,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                lesion_count,
                severity,
                image_path,
                source,
                confidence,
            ),
        )


def save_uploaded_image(uploaded_image) -> Path:
    """Save an uploaded image to disk under a unique filename."""
    suffix = Path(uploaded_image.name).suffix or ".png"
    unique_name = (
        f"{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}{suffix}"
    )
    destination = IMAGES_FOLDER / unique_name
    destination.write_bytes(uploaded_image.getbuffer())
    return destination


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
                severity,
                image_path,
                source,
                confidence
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
        "Upload a facial image and record an acne lesion count, either "
        "automatically or by hand."
    )

    # The hosted build runs as a single shared instance with one database,
    # so any upload is visible to every other visitor on the Dashboard.
    st.warning(
        "**Shared public demo - do not upload a real photo of yourself or "
        "of anyone else.** This deployment stores every upload in one "
        "shared database, so images and results are visible to all other "
        "visitors. Use a sample or stock image. Uploads are also erased "
        "whenever the server restarts."
    )

    uploaded_image = st.file_uploader(
        "Choose a facial image",
        type=["jpg", "jpeg", "png"],
        key="facial_image_uploader",
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Selected image",
            use_container_width=True,
        )

        count_source = st.radio(
            "Lesion count",
            ["Automated (beta)", "Enter manually"],
            horizontal=True,
        )

        confidence = None

        if count_source == "Automated (beta)":
            st.info(
                "Automated mode: a severity classifier (trained on the "
                "Roboflow Acne Severity Classification dataset) predicts "
                "an IGA-scale grade from the photo, and an erythema-based "
                "heuristic counts inflammatory ('active') lesions for "
                "reference - comedones are deliberately not counted. Both "
                "are rough research-prototype estimates, not a diagnosis; "
                "review them before saving."
            )

            uploaded_image.seek(0)
            pil_image = Image.open(uploaded_image)

            prediction = predict_severity(pil_image)
            severity = format_severity(prediction.grade)
            confidence = prediction.confidence

            uploaded_image.seek(0)
            detection = detect_lesions(Image.open(uploaded_image))
            lesion_count = detection.lesion_count
            source = "automated"

            patch_column, overlay_column = st.columns(2)
            with patch_column:
                st.image(
                    prediction.analyzed_patch,
                    caption="Skin patch analyzed by the model",
                    use_container_width=True,
                )
            with overlay_column:
                st.image(
                    detection.annotated_image,
                    caption="Inflammatory lesions detected (reference only)",
                    use_container_width=True,
                )

            first_column, second_column, third_column = st.columns(3)
            with first_column:
                st.metric("Predicted severity", severity)
            with second_column:
                st.metric("Model confidence", f"{confidence:.0%}")
            with third_column:
                st.metric("Inflammatory lesions", lesion_count)
        else:
            st.info(
                "Manual mode: count the inflammatory ('active') lesions "
                "yourself - papules, pustules and nodules, not comedones - "
                "so the number stays comparable with automated entries. "
                "Do not claim that this number was automatically detected."
            )

            lesion_count = int(
                st.number_input(
                    "Inflammatory lesion count",
                    min_value=0,
                    max_value=500,
                    value=18,
                    step=1,
                    key="manual_lesion_count",
                )
            )
            source = "manual"
            severity = severity_from_description(
                classify_acne_severity(int(lesion_count))
            )

            first_column, second_column = st.columns(2)
            with first_column:
                st.metric("Inflammatory lesions", int(lesion_count))
            with second_column:
                st.metric("Severity", severity)

        if st.button(
            "Save acne result",
            type="primary",
            key="save_acne_result",
        ):
            saved_image_path = save_uploaded_image(uploaded_image)
            save_acne_record(
                lesion_count=int(lesion_count),
                severity=severity,
                image_path=str(saved_image_path.relative_to(PROJECT_ROOT)),
                source=source,
                confidence=confidence,
            )
            st.success("The acne result was saved.")


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

    first_column, second_column, third_column, fourth_column = st.columns(4)

    with first_column:
        st.metric("Acne records", len(acne_data))

    with second_column:
        st.metric("Food records", len(food_data))

    with third_column:
        automated_count = (
            int((acne_data["source"] == "automated").sum())
            if not acne_data.empty
            else 0
        )
        st.metric("Automated acne records", automated_count)

    with fourth_column:
        automated_confidence = (
            acne_data.loc[acne_data["source"] == "automated", "confidence"]
            if not acne_data.empty
            else pd.Series(dtype=float)
        )
        average_confidence = (
            f"{automated_confidence.mean():.0%}"
            if not automated_confidence.empty
            else "-"
        )
        st.metric("Avg. model confidence", average_confidence)

    st.subheader("Acne trend")

    if acne_data.empty:
        st.info("No acne records have been saved yet.")
    else:
        latest_record = acne_data.iloc[-1]
        if latest_record["image_path"]:
            latest_image_path = PROJECT_ROOT / latest_record["image_path"]
            if latest_image_path.exists():
                st.image(
                    str(latest_image_path),
                    caption="Most recent capture",
                    width=200,
                )

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

        acne_display["confidence"] = acne_display["confidence"].map(
            lambda value: "-" if pd.isna(value) else f"{value:.0%}"
        )

        acne_display = acne_display.rename(
            columns={
                "recorded_at": "Recorded at",
                "lesion_count": "Inflammatory lesions",
                "severity": "Severity",
                "image_path": "Image",
                "source": "Source",
                "confidence": "Model confidence",
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

        st.bar_chart(trigger_summary)

        food_display = food_data.copy()

        food_display["recorded_at"] = pd.to_datetime(
            food_display["recorded_at"]
        ).dt.strftime("%Y-%m-%d %H:%M")

        for flag_column in ("high_glycemic", "dairy", "refined_sugar"):
            food_display[flag_column] = food_display[flag_column].map(
                {1: "Yes", 0: "No"}
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