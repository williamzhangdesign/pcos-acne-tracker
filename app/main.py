from __future__ import annotations

import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to find the ml folder.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.lesion_detector import detect_lesions
from ml.severity import classify_acne_severity
from ml.severity_classifier import (
    format_severity,
    predict_severity,
    severity_from_description,
)


# Create the data folder if it does not already exist.
DATA_FOLDER = PROJECT_ROOT / "data"
DATA_FOLDER.mkdir(exist_ok=True)

# Uploaded face photos are saved here with unique filenames.
IMAGES_FOLDER = DATA_FOLDER / "images"
IMAGES_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATA_FOLDER / "tracker.db"

# Bound what the vision pipeline ever processes. Cost grows faster than
# linearly with image dimensions, because the detector's baseline blur
# radius scales with the image, so an unbounded upload is a denial of
# service: measured, a 64 MP photo takes ~3.7 minutes of pinned CPU versus
# 40 ms at 640px. Modern phone cameras reach that size without anyone
# attacking anything. The models were trained and tuned on 640px images,
# so bounding here also keeps inference near the training distribution.
MAX_WORKING_DIMENSION = 1024

# Refuse to even decode absurd images, which guards against a small file
# that decompresses to an enormous bitmap.
Image.MAX_IMAGE_PIXELS = 50_000_000

# The deployed instance has finite ephemeral disk; keep only recent images.
MAX_STORED_IMAGES = 200

# Minimum gap between saves from one session, to blunt scripted spamming.
MIN_SECONDS_BETWEEN_SAVES = 3.0

# Hard ceilings for the whole deployment. Once either is reached the
# uploader is disabled outright rather than merely throttled, so no amount
# of persistence can grow the database or the disk any further. These are
# global, not per-session, because sessions are trivially reset.
MAX_TOTAL_ACNE_RECORDS = 500
MAX_TOTAL_FOOD_RECORDS = 500
MAX_TOTAL_IMAGE_BYTES = 100 * 1024 * 1024  # 100 MB


class ImageTooLargeError(Exception):
    """Raised for an image whose pixel count is refused before decoding."""


def prepare_image(uploaded_image) -> Image.Image:
    """Decode an upload and clamp it to a size the CV pipeline can afford."""
    uploaded_image.seek(0)
    image = Image.open(uploaded_image)

    # Image.open only reads the header, so the dimensions are known before
    # any pixels are decoded. Checking here means an oversized image is
    # refused without ever allocating its bitmap - relying on Pillow alone
    # is not enough, since it only warns at MAX_IMAGE_PIXELS and does not
    # raise until twice that.
    width, height = image.size
    if width * height > Image.MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(f"{width}x{height} exceeds the pixel limit")

    image.load()
    image = image.convert("RGB")
    if max(image.size) > MAX_WORKING_DIMENSION:
        image.thumbnail(
            (MAX_WORKING_DIMENSION, MAX_WORKING_DIMENSION),
            Image.LANCZOS,
        )
    return image


def prune_stored_images() -> None:
    """Delete all but the most recent MAX_STORED_IMAGES uploads."""
    files = sorted(
        (p for p in IMAGES_FOLDER.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    for stale in files[:-MAX_STORED_IMAGES]:
        stale.unlink(missing_ok=True)


def stored_image_bytes() -> int:
    """Total bytes currently held in the uploads folder."""
    return sum(
        p.stat().st_size for p in IMAGES_FOLDER.iterdir() if p.is_file()
    )


def count_records(table: str) -> int:
    """Row count for one of the two known tables."""
    if table not in ("acne_records", "food_records"):
        raise ValueError(f"unknown table: {table}")
    with connect_to_database() as connection:
        return int(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        )


def capture_quota_reason() -> str | None:
    """
    Why new captures are refused, or None if they are still allowed.

    This is the hard stop: the caller disables the uploader entirely, so a
    visitor cannot submit anything further once a ceiling is reached.
    """
    if count_records("acne_records") >= MAX_TOTAL_ACNE_RECORDS:
        return (
            f"This demo has reached its limit of "
            f"{MAX_TOTAL_ACNE_RECORDS} saved acne records."
        )
    if stored_image_bytes() >= MAX_TOTAL_IMAGE_BYTES:
        return (
            f"This demo has reached its image storage limit of "
            f"{MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    return None


def save_is_rate_limited() -> bool:
    """True if this session saved too recently."""
    last = st.session_state.get("last_save_time")
    now = time.monotonic()
    if last is not None and now - last < MIN_SECONDS_BETWEEN_SAVES:
        return True
    st.session_state["last_save_time"] = now
    return False


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


def save_uploaded_image(image: Image.Image) -> Path:
    """
    Save the already-bounded image under a unique generated filename.

    The resized image is stored rather than the raw upload: it is what was
    actually analysed, and it keeps disk use predictable on an ephemeral
    host (the raw file may be orders of magnitude larger). The filename is
    generated rather than derived from the upload, so a hostile filename
    cannot influence the write path.
    """
    unique_name = (
        f"{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    )
    destination = IMAGES_FOLDER / unique_name
    image.save(destination, format="JPEG", quality=88)
    return destination


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


def show_capture_page() -> None:
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

    quota_reason = capture_quota_reason()
    if quota_reason:
        st.error(
            f"{quota_reason} Uploads are closed until the demo is reset. "
            "The Dashboard still works."
        )

    uploaded_image = st.file_uploader(
        "Choose a facial image",
        type=["jpg", "jpeg", "png"],
        disabled=bool(quota_reason),
    )

    if uploaded_image is not None and not quota_reason:
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

        try:
            working_image = prepare_image(uploaded_image)
        except (ImageTooLargeError, Image.DecompressionBombError):
            st.error(
                "That image is too large to process. Please upload a "
                "smaller photo."
            )
            return
        except Exception:
            st.error("That file could not be read as an image.")
            return

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

            prediction = predict_severity(working_image)
            severity = format_severity(prediction.grade)
            confidence = prediction.confidence

            detection = detect_lesions(working_image)
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

        if st.button("Save acne result", type="primary"):
            blocked = capture_quota_reason()
            if blocked:
                st.error(f"{blocked} This entry was not saved.")
                return

            if save_is_rate_limited():
                st.warning("Please wait a moment before saving again.")
                return

            saved_image_path = save_uploaded_image(working_image)
            prune_stored_images()
            save_acne_record(
                lesion_count=int(lesion_count),
                severity=severity,
                image_path=str(saved_image_path.relative_to(PROJECT_ROOT)),
                source=source,
                confidence=confidence,
            )
            st.success("The acne result was saved.")


def show_food_log_page() -> None:
    st.header("Food Log")

    food_description = st.text_input(
        "What did you eat?",
        placeholder="Example: cereal with milk and sweetened coffee",
        max_chars=200,
    )

    high_glycemic = st.checkbox("High-glycemic food")
    dairy = st.checkbox("Contains dairy")
    refined_sugar = st.checkbox("Contains refined sugar")

    if st.button("Save food entry", type="primary"):
        if not food_description.strip():
            st.warning("Please type a food or meal first.")
            return

        if count_records("food_records") >= MAX_TOTAL_FOOD_RECORDS:
            st.error(
                f"This demo has reached its limit of "
                f"{MAX_TOTAL_FOOD_RECORDS} food entries."
            )
            return

        if save_is_rate_limited():
            st.warning("Please wait a moment before saving again.")
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

        st.dataframe(
            acne_data.rename(columns={"source": "Source"}),
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