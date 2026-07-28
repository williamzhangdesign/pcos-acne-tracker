from food.trigger_detector import detect_dietary_triggers


test_meals = [
    "Cereal with milk and sweetened coffee",
    "Grilled chicken with broccoli",
    "White rice with butter chicken",
    "Ice cream and cookies",
    "Salmon with mixed vegetables",
]


for meal in test_meals:
    result = detect_dietary_triggers(meal)

    print("\nMeal:", meal)
    print("Detected labels:", result.detected_labels)
    print("High-glycemic matches:", result.matched_high_glycemic)
    print("Dairy matches:", result.matched_dairy)
    print("Refined-sugar matches:", result.matched_refined_sugar)