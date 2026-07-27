def classify_acne_severity(lesion_count: int) -> str:
    """
    Convert a lesion count into a simple acne severity category.
    """

    if lesion_count < 0:
        raise ValueError("Lesion count cannot be negative.")

    if lesion_count < 30:
        return "Mild"

    if lesion_count <= 125:
        return "Moderate"

    return "Severe"