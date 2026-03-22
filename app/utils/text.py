"""Text utility functions."""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Lowercase and strip excess whitespace from text."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_quantity_and_unit(text: str) -> tuple[float | None, str | None]:
    """
    Extract a leading quantity and optional unit from a phrase.
    
    Examples:
        "two eggs" -> (2.0, None)
        "1 cup milk" -> (1.0, "cup")
        "half an avocado" -> (0.5, None)
    """
    word_numbers = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5, "quarter": 0.25, "couple": 2, "few": 3, "some": None,
    }
    units = {
        "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon",
        "teaspoons", "ml", "millilitre", "millilitres", "milliliter", "milliliters",
        "g", "gram", "grams", "kg", "kilogram", "oz", "ounce", "ounces",
        "lb", "pound", "pounds", "slice", "slices", "piece", "pieces",
        "portion", "portions", "handful", "handfuls", "glass", "glasses",
        "bowl", "bowls", "plate", "plates", "serving", "servings",
    }
    text = text.strip()
    # Try numeric first
    numeric_pattern = r"^(\d+(?:\.\d+)?(?:/\d+)?)\s*([a-z]+)?\s*"
    m = re.match(numeric_pattern, text, re.IGNORECASE)
    if m:
        qty_str = m.group(1)
        # Handle fractions like 1/2
        if "/" in qty_str:
            parts = qty_str.split("/")
            try:
                qty = float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                qty = None
        else:
            try:
                qty = float(qty_str)
            except ValueError:
                qty = None
        unit_candidate = m.group(2)
        unit = unit_candidate.lower() if unit_candidate and unit_candidate.lower() in units else None
        return qty, unit

    # Try word numbers
    words = text.lower().split()
    if words and words[0] in word_numbers:
        qty = word_numbers[words[0]]
        unit = None
        if len(words) > 1 and words[1] in units:
            unit = words[1]
        return qty, unit

    return None, None


def clean_food_text(text: str) -> str:
    """Remove filler words and clean up a food diary entry."""
    fillers = [
        r"\bI had\b", r"\bI ate\b", r"\bI drank\b", r"\bI've had\b",
        r"\bwe had\b", r"\bwas\b", r"\bwere\b", r"\bsome\b", r"\bjust\b",
        r"\bprobably\b", r"\bmaybe\b", r"\baround\b", r"\babout\b",
    ]
    result = text
    for filler in fillers:
        result = re.sub(filler, "", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip()


def split_food_items(text: str) -> list[str]:
    """
    Split a comma/and-separated list of food items into individual items.
    
    Handles: "eggs, toast, and coffee with milk" -> ["eggs", "toast", "coffee with milk"]
    """
    # Split on ", and ", " and ", or ", "
    # But be careful not to split compound items like "mac and cheese"
    compound_foods = {
        "mac and cheese", "macaroni and cheese", "fish and chips",
        "bangers and mash", "bread and butter", "salt and pepper",
        "beans on toast", "ham and eggs",
    }

    # First check for compound foods and protect them
    protected = text
    placeholders: dict[str, str] = {}
    for i, compound in enumerate(compound_foods):
        placeholder = f"__COMPOUND_{i}__"
        if compound.lower() in protected.lower():
            pattern = re.compile(re.escape(compound), re.IGNORECASE)
            protected = pattern.sub(placeholder, protected)
            placeholders[placeholder] = compound

    # Split on ", and ", ", ", or bare " and " (with word boundaries via (?<!\w)/(?!\w)
    # to avoid splitting compound foods like "mac and cheese" that weren't caught above).
    parts = re.split(r",\s*(?:and\s+)?|(?<!\w)\s+and\s+(?!\w)", protected, flags=re.IGNORECASE)

    # Restore placeholders
    result = []
    for part in parts:
        restored = part.strip()
        for placeholder, original in placeholders.items():
            restored = restored.replace(placeholder, original)
        if restored:
            result.append(restored)

    return result
