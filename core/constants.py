from dataclasses import dataclass

@dataclass(frozen=True)
class Constants:
    FOOD_PROMPTS = ["food", "meal", "cooked food"]
    CONTAINER_PROMPTS = ["plate", "bowl", "dish", "wooden plate", "ceramic plate", 'cup']
