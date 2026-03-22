"""
Food diary parser service.

Provides an abstract FoodParser interface plus a rule-based implementation
that works without any external models or APIs. An optional Ollama-backed
parser is also provided as an extension point.
"""
from __future__ import annotations

import datetime
import re
from abc import ABC, abstractmethod
from typing import Optional

from app.schemas import (
    FoodItemSchema,
    IngredientSchema,
    ParsedMealSchema,
)
from app.services.ingredient_inference import get_inferencer, IngredientInferencer
from app.utils.text import (
    clean_food_text,
    extract_quantity_and_unit,
    normalize_text,
    split_food_items,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Meal type keywords
MEAL_TYPE_PATTERNS: dict[str, list[str]] = {
    "breakfast": ["breakfast", "morning", "brunch"],
    "lunch": ["lunch", "midday", "noon"],
    "dinner": ["dinner", "supper", "evening meal", "tea"],
    "snack": ["snack", "snacked", "bite", "nibble"],
}

# Quantity word patterns in food descriptions
QUANTITY_PATTERNS = [
    (r"\btwo\b", 2.0),
    (r"\bthree\b", 3.0),
    (r"\bfour\b", 4.0),
    (r"\ba\b|\ban\b|\bone\b", 1.0),
    (r"\bhalf\b", 0.5),
    (r"\bcouple\b", 2.0),
    (r"\bfew\b", 3.0),
]


def _detect_meal_type(text: str) -> Optional[str]:
    """Detect meal type from text."""
    lowered = text.lower()
    for meal_type, keywords in MEAL_TYPE_PATTERNS.items():
        for kw in keywords:
            if kw in lowered:
                return meal_type
    return None


def _detect_date_time(text: str) -> tuple[Optional[datetime.date], Optional[datetime.time]]:
    """Extract date and time from text if present (basic heuristic)."""
    today = datetime.date.today()
    entry_date = today
    entry_time: Optional[datetime.time] = None

    time_match = re.search(r"\b(\d{1,2})[:\.](\d{2})\s*(am|pm)?\b", text, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        meridiem = time_match.group(3)
        if meridiem and meridiem.lower() == "pm" and hour < 12:
            hour += 12
        elif meridiem and meridiem.lower() == "am" and hour == 12:
            hour = 0
        try:
            entry_time = datetime.time(hour, minute)
        except ValueError:
            entry_time = None

    # Simple relative date detection
    lowered = text.lower()
    if "yesterday" in lowered:
        entry_date = today - datetime.timedelta(days=1)
    elif "today" in lowered:
        entry_date = today

    return entry_date, entry_time


def _extract_portion_text(text: str) -> str:
    """Extract the portion description prefix from an item phrase."""
    quantity_prefixes = [
        r"^\d+(?:\.\d+)?\s*(?:cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|"
        r"teaspoons|ml|g|gram|grams|kg|oz|ounce|ounces|lb|pound|pounds|"
        r"slice|slices|piece|pieces|portion|portions|handful|handfuls|"
        r"glass|glasses|bowl|bowls|plate|plates|serving|servings)s?\s+of\s+",
        r"^(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"half|quarter|couple|few|some)\s+",
        r"^\d+\s+",
    ]
    for pattern in quantity_prefixes:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


class FoodParser(ABC):
    """Abstract food diary parser."""

    @abstractmethod
    def parse(self, raw_text: str) -> ParsedMealSchema:
        """Parse raw food diary text into a structured meal record."""
        ...


class RuleBasedParser(FoodParser):
    """
    Rule-based food diary parser.
    
    Works without cloud APIs or local ML models.
    Uses heuristics for meal type detection, item splitting, and portion parsing.
    """

    def __init__(self, inferencer: Optional[IngredientInferencer] = None):
        self.inferencer = inferencer or get_inferencer()

    def parse(self, raw_text: str) -> ParsedMealSchema:
        """Parse a raw food diary entry using rules."""
        logger.info("Parsing entry (rule-based): %r", raw_text[:60])

        # Detect metadata
        meal_type = _detect_meal_type(raw_text)
        entry_date, entry_time = _detect_date_time(raw_text)

        # Clean text for item extraction
        cleaned = clean_food_text(raw_text)

        # Remove meal type prefix like "For breakfast I had..." or "Breakfast: ..."
        cleaned_for_split = re.sub(
            r"^(for\s+)?(breakfast|lunch|dinner|supper|snack)[:\s]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Split into individual food items
        raw_items = split_food_items(cleaned_for_split)
        if not raw_items:
            raw_items = [raw_text.strip()]

        food_items: list[FoodItemSchema] = []
        confidences: list[float] = []

        for order, raw_item in enumerate(raw_items, start=1):
            raw_item = raw_item.strip()
            if not raw_item:
                continue

            # Extract portion prefix
            portion_text = _extract_portion_text(raw_item)
            qty, unit = extract_quantity_and_unit(raw_item)

            # The item name is what remains after the quantity prefix
            item_name = raw_item
            if portion_text:
                item_name = raw_item[len(portion_text):].strip()
                # Remove "of" connector
                item_name = re.sub(r"^of\s+", "", item_name, flags=re.IGNORECASE)

            if not item_name:
                item_name = raw_item

            normalized = normalize_text(item_name)
            item_confidence = 0.7

            # Infer ingredients
            ingredients = self.inferencer.infer(normalized)
            if not ingredients:
                item_confidence = 0.5

            food_item = FoodItemSchema(
                item_name=item_name,
                normalized_item_name=normalized,
                portion_text=portion_text or None,
                estimated_quantity=qty,
                estimated_unit=unit,
                confidence=item_confidence,
                item_order=order,
                ingredients=ingredients,
            )
            food_items.append(food_item)
            confidences.append(item_confidence)

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        notes: list[str] = []
        if overall_confidence < 0.6:
            notes.append("Some items have low confidence — please review.")
        if not meal_type:
            notes.append("Meal type could not be determined from text.")

        return ParsedMealSchema(
            raw_entry_text=raw_text,
            cleaned_entry_text=cleaned,
            entry_date=entry_date,
            entry_time=entry_time,
            meal_type=meal_type,
            overall_confidence=overall_confidence,
            processing_notes="; ".join(notes) if notes else None,
            food_items=food_items,
        )


class OllamaParser(FoodParser):
    """
    Ollama-backed food diary parser.
    
    Extension point for local LLM parsing. Requires Ollama running locally.
    Falls back to RuleBasedParser if Ollama is unavailable.
    """

    DEFAULT_PROMPT = (
        "You are a food diary assistant. Parse the following food diary entry "
        "and return a JSON object with keys: meal_type (breakfast/lunch/dinner/snack/null), "
        "food_items (list of objects with: item_name, portion_text, ingredients list). "
        "Only return JSON, no explanation.\n\nEntry: {text}\n\nJSON:"
    )

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        fallback_parser: Optional[FoodParser] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.fallback = fallback_parser or RuleBasedParser()

    def parse(self, raw_text: str) -> ParsedMealSchema:
        """Parse using Ollama, falling back to rule-based on failure."""
        try:
            import json
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "model": self.model,
                "prompt": self.DEFAULT_PROMPT.format(text=raw_text),
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            response_text = data.get("response", "")
            parsed_json = json.loads(response_text)
            return self._json_to_schema(raw_text, parsed_json)
        except (
            urllib.error.URLError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("Ollama parse failed (%s), falling back to rule-based", exc)
            return self.fallback.parse(raw_text)

    def _json_to_schema(self, raw_text: str, data: dict) -> ParsedMealSchema:
        """Convert Ollama JSON response to ParsedMealSchema."""
        meal_type = data.get("meal_type")
        food_items_raw = data.get("food_items", [])
        food_items: list[FoodItemSchema] = []
        for order, item in enumerate(food_items_raw, start=1):
            ingredients = [
                IngredientSchema(
                    ingredient_name=ing if isinstance(ing, str) else ing.get("name", ""),
                    source_method="model_inferred",
                    confidence=0.6,
                )
                for ing in item.get("ingredients", [])
            ]
            food_items.append(
                FoodItemSchema(
                    item_name=item.get("item_name", ""),
                    normalized_item_name=normalize_text(item.get("item_name", "")),
                    portion_text=item.get("portion_text"),
                    confidence=0.75,
                    item_order=order,
                    ingredients=ingredients,
                )
            )
        today = datetime.date.today()
        return ParsedMealSchema(
            raw_entry_text=raw_text,
            cleaned_entry_text=clean_food_text(raw_text),
            entry_date=today,
            meal_type=meal_type,
            overall_confidence=0.75,
            food_items=food_items,
        )


def get_parser(use_ollama: bool = False, **kwargs) -> FoodParser:
    """Return the appropriate food parser based on configuration."""
    if use_ollama:
        return OllamaParser(**kwargs)
    return RuleBasedParser()
