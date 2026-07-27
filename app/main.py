from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to find the ml folder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.severity import classify_acne_severity


# Create the data folder if it does not already exist.
DATA_FOLDER = PROJECT_ROOT / "data"
DATA_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATA_FOLDER / "tracker.db"


st.set_page_config(
    page_title="PCOS Acne Severity Tracker",
    page_icon="📱",
    layout="centered",
)


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


def save_acne_record(lesion_count: int, severity: str) -> None:
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
    """Save a food entry in the database."""
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
            SELECT recorded_at, lesion_count, severity
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


def show_capture_page() -> None:
    st.header("Capture and Acne Score")

    st.write(
        "Upload a facial image and record an acne lesion count. "
        "The live computer-vision API will be connected separately."
    )

    uploaded_image = st.file_uploader(
        "Choose a facial image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Selected image",
            use_container_width=True,
        )

        st.info(
            "Prototype mode: enter the lesion count manually. "
            "Do not claim that this number was automatically detected."
        )

        lesion_count = st.number_input(
            "Lesion count",
            min_value=0,
            max_value=500,
            value=18,
            step=1,
        )

        severity = classify_acne_severity(int(lesion_count))

        first_column, second_column = st.columns(2)

        with first_column:
            st.metric("Lesion count", int(lesion_count))

        with second_column:
            st.metric("Severity", severity)

        if st.button("Save acne result", type="primary"):
            save_acne_record(
                lesion_count=int(lesion_count),
                severity=severity,
            )
            st.success("The acne result was saved.")


def show_food_log_page() -> None:
    st.header("Food Log")

    food_description = st.text_input(
        "What did you eat?",
        placeholder="Example: cereal with milk and sweetened coffee",
    )

    high_glycemic = st.checkbox("High-glycemic food")
    dairy = st.checkbox("Contains dairy")
    refined_sugar = st.checkbox("Contains refined sugar")

    if st.button("Save food entry", type="primary"):
        if not food_description.strip():
            st.warning("Please type a food or meal first.")
            return

        save_food_record(
            food_description=food_description.strip(),
            high_glycemic=high_glycemic,
            dairy=dairy,
            refined_sugar=refined_sugar,
        )

        st.success("The food entry was saved.")


def show_dashboard_page() -> None:
    st.header("Dashboard")

    acne_data = load_acne_records()
    food_data = load_food_records()

    first_column, second_column = st.columns(2)

    with first_column:
        st.metric("Acne records", len(acne_data))

    with second_column:
        st.metric("Food records", len(food_data))

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

        st.dataframe(
            acne_data,
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

        st.dataframe(
            food_data,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "This prototype supports self-reflection only. "
        "It does not diagnose PCOS, acne, or dietary causes."
    )


def show_profile_page() -> None:
    st.header("Profile and Settings")

    st.text_input("Display name")

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
    )

    st.checkbox("Enable a daily logging reminder")

    st.warning(
        "This project is an educational prototype and is not a medical device."
    )


def main() -> None:
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