from dataclasses import dataclass

@dataclass(frozen=True)
class Constants:
    DEBUG = False
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