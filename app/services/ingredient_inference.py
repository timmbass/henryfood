"""
Ingredient inference service.

Provides an abstract IngredientInferencer interface and a rule-based
implementation using a built-in knowledge base of common composite foods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from app.schemas import IngredientSchema
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Knowledge base: normalized food name -> list of (ingredient_name, confidence, source_method)
INGREDIENT_KNOWLEDGE_BASE: dict[str, list[tuple[str, float, str]]] = {
    # Egg dishes
    "scrambled eggs": [
        ("eggs", 0.95, "rule_based"),
        ("butter", 0.80, "rule_based"),
        ("salt", 0.70, "rule_based"),
        ("milk", 0.50, "rule_based"),
    ],
    "scrambled eggs with butter": [
        ("eggs", 0.95, "rule_based"),
        ("butter", 0.95, "rule_based"),
        ("salt", 0.70, "rule_based"),
    ],
    "fried eggs": [
        ("eggs", 0.95, "rule_based"),
        ("butter", 0.70, "rule_based"),
        ("oil", 0.60, "rule_based"),
    ],
    "boiled eggs": [
        ("eggs", 0.95, "rule_based"),
    ],
    "poached eggs": [
        ("eggs", 0.95, "rule_based"),
        ("vinegar", 0.40, "rule_based"),
    ],
    "omelette": [
        ("eggs", 0.95, "rule_based"),
        ("butter", 0.70, "rule_based"),
        ("salt", 0.70, "rule_based"),
        ("pepper", 0.60, "rule_based"),
    ],
    # Toast and bread
    "toast": [
        ("bread", 0.95, "rule_based"),
    ],
    "sourdough toast": [
        ("sourdough bread", 0.95, "rule_based"),
    ],
    "toast with butter": [
        ("bread", 0.95, "rule_based"),
        ("butter", 0.90, "rule_based"),
    ],
    "sourdough toast with butter": [
        ("sourdough bread", 0.95, "rule_based"),
        ("butter", 0.90, "rule_based"),
    ],
    "garlic bread": [
        ("bread", 0.90, "rule_based"),
        ("butter", 0.85, "rule_based"),
        ("garlic", 0.90, "rule_based"),
        ("parsley", 0.50, "rule_based"),
    ],
    "avocado toast": [
        ("bread", 0.90, "rule_based"),
        ("avocado", 0.95, "rule_based"),
        ("lemon juice", 0.60, "rule_based"),
        ("salt", 0.70, "rule_based"),
    ],
    # Sandwiches and wraps
    "chicken salad wrap": [
        ("tortilla wrap", 0.90, "rule_based"),
        ("chicken", 0.90, "rule_based"),
        ("lettuce", 0.80, "rule_based"),
        ("tomato", 0.75, "rule_based"),
        ("mayo", 0.65, "rule_based"),
        ("dressing", 0.60, "rule_based"),
    ],
    "blt sandwich": [
        ("bread", 0.90, "rule_based"),
        ("bacon", 0.90, "rule_based"),
        ("lettuce", 0.90, "rule_based"),
        ("tomato", 0.90, "rule_based"),
        ("mayo", 0.70, "rule_based"),
    ],
    "cheese sandwich": [
        ("bread", 0.90, "rule_based"),
        ("cheese", 0.90, "rule_based"),
        ("butter", 0.60, "rule_based"),
    ],
    "ham sandwich": [
        ("bread", 0.90, "rule_based"),
        ("ham", 0.90, "rule_based"),
        ("butter", 0.60, "rule_based"),
    ],
    # Pasta dishes
    "spaghetti bolognese": [
        ("spaghetti", 0.95, "rule_based"),
        ("minced beef", 0.90, "rule_based"),
        ("tomato sauce", 0.85, "rule_based"),
        ("onion", 0.80, "rule_based"),
        ("garlic", 0.80, "rule_based"),
        ("olive oil", 0.75, "rule_based"),
        ("herbs", 0.70, "rule_based"),
        ("parmesan", 0.50, "rule_based"),
    ],
    "pasta bolognese": [
        ("pasta", 0.95, "rule_based"),
        ("minced beef", 0.90, "rule_based"),
        ("tomato sauce", 0.85, "rule_based"),
        ("onion", 0.80, "rule_based"),
        ("garlic", 0.80, "rule_based"),
    ],
    "carbonara": [
        ("pasta", 0.90, "rule_based"),
        ("bacon", 0.85, "rule_based"),
        ("eggs", 0.85, "rule_based"),
        ("parmesan", 0.80, "rule_based"),
        ("black pepper", 0.75, "rule_based"),
    ],
    "pasta carbonara": [
        ("pasta", 0.90, "rule_based"),
        ("bacon", 0.85, "rule_based"),
        ("eggs", 0.85, "rule_based"),
        ("parmesan", 0.80, "rule_based"),
        ("black pepper", 0.75, "rule_based"),
    ],
    "mac and cheese": [
        ("macaroni pasta", 0.90, "rule_based"),
        ("cheddar cheese", 0.85, "rule_based"),
        ("milk", 0.75, "rule_based"),
        ("butter", 0.70, "rule_based"),
        ("flour", 0.65, "rule_based"),
    ],
    "macaroni and cheese": [
        ("macaroni pasta", 0.90, "rule_based"),
        ("cheddar cheese", 0.85, "rule_based"),
        ("milk", 0.75, "rule_based"),
        ("butter", 0.70, "rule_based"),
        ("flour", 0.65, "rule_based"),
    ],
    # Rice dishes
    "chicken fried rice": [
        ("rice", 0.90, "rule_based"),
        ("chicken", 0.90, "rule_based"),
        ("eggs", 0.80, "rule_based"),
        ("soy sauce", 0.80, "rule_based"),
        ("spring onions", 0.70, "rule_based"),
        ("oil", 0.75, "rule_based"),
    ],
    "egg fried rice": [
        ("rice", 0.90, "rule_based"),
        ("eggs", 0.90, "rule_based"),
        ("soy sauce", 0.80, "rule_based"),
        ("spring onions", 0.70, "rule_based"),
        ("oil", 0.75, "rule_based"),
    ],
    # Salads
    "caesar salad": [
        ("romaine lettuce", 0.85, "rule_based"),
        ("croutons", 0.80, "rule_based"),
        ("parmesan", 0.80, "rule_based"),
        ("caesar dressing", 0.85, "rule_based"),
    ],
    "green salad": [
        ("mixed leaves", 0.80, "rule_based"),
        ("cucumber", 0.70, "rule_based"),
        ("tomato", 0.70, "rule_based"),
        ("dressing", 0.65, "rule_based"),
    ],
    "chicken salad": [
        ("chicken", 0.90, "rule_based"),
        ("mixed leaves", 0.80, "rule_based"),
        ("cucumber", 0.70, "rule_based"),
        ("tomato", 0.70, "rule_based"),
        ("dressing", 0.65, "rule_based"),
    ],
    # Breakfast items
    "cereal with milk": [
        ("cereal", 0.90, "rule_based"),
        ("milk", 0.90, "rule_based"),
    ],
    "porridge": [
        ("oats", 0.90, "rule_based"),
        ("milk", 0.80, "rule_based"),
        ("water", 0.60, "rule_based"),
        ("salt", 0.50, "rule_based"),
    ],
    "porridge with honey": [
        ("oats", 0.90, "rule_based"),
        ("milk", 0.80, "rule_based"),
        ("honey", 0.90, "rule_based"),
    ],
    "granola": [
        ("oats", 0.85, "rule_based"),
        ("honey", 0.75, "rule_based"),
        ("nuts", 0.70, "rule_based"),
    ],
    "pancakes": [
        ("flour", 0.85, "rule_based"),
        ("eggs", 0.85, "rule_based"),
        ("milk", 0.85, "rule_based"),
        ("butter", 0.75, "rule_based"),
        ("baking powder", 0.60, "rule_based"),
    ],
    # Coffee and tea
    "coffee": [
        ("coffee", 0.95, "rule_based"),
    ],
    "coffee with milk": [
        ("coffee", 0.95, "rule_based"),
        ("milk", 0.90, "rule_based"),
    ],
    "latte": [
        ("espresso", 0.90, "rule_based"),
        ("steamed milk", 0.90, "rule_based"),
    ],
    "cappuccino": [
        ("espresso", 0.90, "rule_based"),
        ("steamed milk", 0.85, "rule_based"),
        ("milk foam", 0.80, "rule_based"),
    ],
    "tea": [
        ("tea", 0.95, "rule_based"),
    ],
    "tea with milk": [
        ("tea", 0.95, "rule_based"),
        ("milk", 0.90, "rule_based"),
    ],
    # Snacks
    "apple and peanut butter": [
        ("apple", 0.95, "rule_based"),
        ("peanut butter", 0.95, "rule_based"),
    ],
    "peanut butter on toast": [
        ("bread", 0.90, "rule_based"),
        ("peanut butter", 0.95, "rule_based"),
    ],
    "hummus and crackers": [
        ("hummus", 0.95, "rule_based"),
        ("crackers", 0.90, "rule_based"),
        ("chickpeas", 0.80, "rule_based"),
        ("tahini", 0.75, "rule_based"),
        ("lemon juice", 0.70, "rule_based"),
        ("garlic", 0.65, "rule_based"),
    ],
    # Meat and fish mains
    "roast chicken": [
        ("chicken", 0.95, "rule_based"),
        ("olive oil", 0.75, "rule_based"),
        ("herbs", 0.70, "rule_based"),
        ("garlic", 0.65, "rule_based"),
    ],
    "grilled salmon": [
        ("salmon", 0.95, "rule_based"),
        ("olive oil", 0.70, "rule_based"),
        ("lemon", 0.65, "rule_based"),
    ],
    "fish and chips": [
        ("fish", 0.90, "rule_based"),
        ("potatoes", 0.90, "rule_based"),
        ("batter", 0.80, "rule_based"),
        ("oil", 0.80, "rule_based"),
    ],
    # Soups
    "tomato soup": [
        ("tomatoes", 0.90, "rule_based"),
        ("onion", 0.75, "rule_based"),
        ("garlic", 0.70, "rule_based"),
        ("vegetable stock", 0.65, "rule_based"),
        ("olive oil", 0.65, "rule_based"),
    ],
    "chicken soup": [
        ("chicken", 0.90, "rule_based"),
        ("vegetables", 0.80, "rule_based"),
        ("stock", 0.80, "rule_based"),
        ("noodles", 0.60, "rule_based"),
    ],
    # Pizza and burgers
    "pizza": [
        ("pizza dough", 0.85, "rule_based"),
        ("tomato sauce", 0.85, "rule_based"),
        ("mozzarella", 0.85, "rule_based"),
    ],
    "margherita pizza": [
        ("pizza dough", 0.90, "rule_based"),
        ("tomato sauce", 0.90, "rule_based"),
        ("mozzarella", 0.90, "rule_based"),
        ("basil", 0.75, "rule_based"),
    ],
    "burger": [
        ("beef patty", 0.85, "rule_based"),
        ("burger bun", 0.85, "rule_based"),
        ("lettuce", 0.75, "rule_based"),
        ("tomato", 0.75, "rule_based"),
        ("ketchup", 0.70, "rule_based"),
    ],
    # Beverages
    "orange juice": [
        ("orange juice", 0.95, "rule_based"),
    ],
    "smoothie": [
        ("fruit", 0.80, "rule_based"),
        ("milk or yogurt", 0.70, "rule_based"),
    ],
    "red wine": [
        ("red wine", 0.95, "rule_based"),
    ],
    "white wine": [
        ("white wine", 0.95, "rule_based"),
    ],
    "beer": [
        ("beer", 0.95, "rule_based"),
    ],
    "diet coke": [
        ("diet cola", 0.95, "rule_based"),
    ],
    "cola": [
        ("cola", 0.95, "rule_based"),
    ],
    "water": [
        ("water", 0.99, "rule_based"),
    ],
}

# Partial-match rules: (keyword_substring, [(ingredient_name, confidence, source_method)])
# Used as a fallback when no exact or word-overlap match is found in INGREDIENT_KNOWLEDGE_BASE.
PARTIAL_MATCH_RULES: list[tuple[str, list[tuple[str, float, str]]]] = [
    ("egg", [("eggs", 0.85, "rule_based")]),
    ("chicken", [("chicken", 0.85, "rule_based")]),
    ("beef", [("beef", 0.85, "rule_based")]),
    ("pork", [("pork", 0.85, "rule_based")]),
    ("fish", [("fish", 0.85, "rule_based")]),
    ("pasta", [("pasta", 0.85, "rule_based")]),
    ("rice", [("rice", 0.85, "rule_based")]),
    ("bread", [("bread", 0.85, "rule_based")]),
    ("potato", [("potatoes", 0.85, "rule_based")]),
    ("salad", [("mixed leaves", 0.70, "rule_based"), ("dressing", 0.60, "rule_based")]),
    ("soup", [("vegetables", 0.65, "rule_based"), ("stock", 0.65, "rule_based")]),
    ("sandwich", [("bread", 0.85, "rule_based"), ("filling", 0.70, "rule_based")]),
    ("wrap", [("tortilla wrap", 0.85, "rule_based"), ("filling", 0.70, "rule_based")]),
    ("cereal", [("cereal", 0.85, "rule_based"), ("milk", 0.75, "rule_based")]),
    ("yogurt", [("yogurt", 0.90, "rule_based")]),
    ("yoghurt", [("yogurt", 0.90, "rule_based")]),
    ("cheese", [("cheese", 0.90, "rule_based")]),
    ("fruit", [("fruit", 0.80, "rule_based")]),
    ("apple", [("apple", 0.95, "rule_based")]),
    ("banana", [("banana", 0.95, "rule_based")]),
    ("orange", [("orange", 0.95, "rule_based")]),
    ("nuts", [("nuts", 0.85, "rule_based")]),
    ("crisps", [("crisps", 0.90, "rule_based"), ("potato", 0.80, "rule_based"), ("salt", 0.70, "rule_based")]),
    ("chips", [("chips", 0.90, "rule_based")]),
    ("biscuit", [("biscuit", 0.90, "rule_based"), ("flour", 0.60, "rule_based"), ("butter", 0.60, "rule_based")]),
    ("cookie", [("cookie", 0.90, "rule_based"), ("flour", 0.60, "rule_based"), ("butter", 0.65, "rule_based"), ("sugar", 0.65, "rule_based")]),
    ("cake", [("cake", 0.85, "rule_based"), ("flour", 0.70, "rule_based"), ("butter", 0.70, "rule_based"), ("eggs", 0.70, "rule_based"), ("sugar", 0.75, "rule_based")]),
    ("chocolate", [("chocolate", 0.90, "rule_based"), ("cocoa", 0.80, "rule_based"), ("sugar", 0.70, "rule_based")]),
    ("milk", [("milk", 0.95, "rule_based")]),
    ("butter", [("butter", 0.95, "rule_based")]),
    ("coffee", [("coffee", 0.95, "rule_based")]),
    ("tea", [("tea", 0.95, "rule_based")]),
    ("wine", [("wine", 0.90, "rule_based")]),
    ("beer", [("beer", 0.95, "rule_based")]),
    ("juice", [("juice", 0.90, "rule_based")]),
    ("water", [("water", 0.99, "rule_based")]),
]


class IngredientInferencer(ABC):
    """Abstract ingredient inferencer."""

    @abstractmethod
    def infer(self, normalized_item_name: str) -> list[IngredientSchema]:
        """Infer ingredients for the given normalized food item name."""
        ...


class RuleBasedInferencer(IngredientInferencer):
    """
    Rule-based ingredient inferencer using a built-in knowledge base.
    
    Performs exact lookup first, then partial matching.
    """

    def infer(self, normalized_item_name: str) -> list[IngredientSchema]:
        """Infer ingredients using the knowledge base."""
        name = normalized_item_name.strip().lower()

        # Exact match
        if name in INGREDIENT_KNOWLEDGE_BASE:
            return [
                IngredientSchema(
                    ingredient_name=ing_name,
                    normalized_ingredient_name=ing_name.lower(),
                    confidence=conf,
                    source_method=source,
                )
                for ing_name, conf, source in INGREDIENT_KNOWLEDGE_BASE[name]
            ]

        # Partial match: find the best matching key
        best_match: str | None = None
        best_score = 0
        for key in INGREDIENT_KNOWLEDGE_BASE:
            # Count overlapping words
            key_words = set(key.split())
            name_words = set(name.split())
            overlap = len(key_words & name_words)
            if overlap > best_score and overlap >= min(len(key_words), 2):
                best_score = overlap
                best_match = key

        if best_match:
            logger.debug("Partial knowledge base match: %r -> %r", name, best_match)
            return [
                IngredientSchema(
                    ingredient_name=ing_name,
                    normalized_ingredient_name=ing_name.lower(),
                    confidence=conf * 0.85,  # Slightly reduced confidence for partial match
                    source_method=source,
                )
                for ing_name, conf, source in INGREDIENT_KNOWLEDGE_BASE[best_match]
            ]

        # Fallback: partial string matching
        results: list[IngredientSchema] = []
        seen: set[str] = set()
        for keyword, ingredients in PARTIAL_MATCH_RULES:
            if keyword in name:
                for ing_name, conf, source in ingredients:
                    if ing_name not in seen:
                        seen.add(ing_name)
                        results.append(
                            IngredientSchema(
                                ingredient_name=ing_name,
                                normalized_ingredient_name=ing_name.lower(),
                                confidence=conf * 0.75,  # Lower confidence for generic match
                                source_method=source,
                            )
                        )

        return results


def get_inferencer() -> IngredientInferencer:
    """Return the default ingredient inferencer."""
    return RuleBasedInferencer()
