from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    DEBUG = False
    GROUNDING_PROMPT = (
        "food, meal, cooked food, dish, plate, bowl, wooden plate, ceramic plate, "
        "cup, knife, spoon, sauce, meat, cheese, "
        "garnish, accent, leaf, herb, fresh, verde, aroma, botanical, greenery, "
        "sprig, essence, finish, touch, crown, detail"
    )
    FOOD_PROMPTS = ["food", "meal", "cooked food"]
    CONTAINER_PROMPTS = ["plate", "bowl", "dish", "wooden plate", "ceramic plate", 'cup']
    garnish_names = [
        "Garnish",
        "Accent",
        "Leaf",
        "Herb",
        "Fresh",
        "Verde",
        "Aroma",
        "Botanical",
        "Greenery",
        "Sprig",
        "Essence",
        "Finish",
        "Touch",
        "Crown",
        "Detail"
    ]
