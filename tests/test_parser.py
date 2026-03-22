"""Tests for the rule-based food parser."""
from __future__ import annotations

import datetime
import pytest

from app.services.parser import RuleBasedParser, _detect_meal_type, _detect_date_time
from app.services.ingredient_inference import RuleBasedInferencer


@pytest.fixture
def parser():
    return RuleBasedParser(inferencer=RuleBasedInferencer())


class TestMealTypeDetection:
    def test_breakfast_detected(self):
        assert _detect_meal_type("For breakfast I had eggs") == "breakfast"

    def test_lunch_detected(self):
        assert _detect_meal_type("Lunch was a chicken salad wrap") == "lunch"

    def test_dinner_detected(self):
        assert _detect_meal_type("Dinner was spaghetti bolognese") == "dinner"

    def test_snack_detected(self):
        assert _detect_meal_type("Snack: apple and peanut butter") == "snack"

    def test_no_meal_type(self):
        assert _detect_meal_type("I had eggs and toast") is None


class TestDateTimeDetection:
    def test_today_detected(self):
        date, _ = _detect_date_time("Today I had soup")
        assert date == datetime.date.today()

    def test_yesterday_detected(self):
        date, _ = _detect_date_time("Yesterday for lunch I had pizza")
        assert date == datetime.date.today() - datetime.timedelta(days=1)

    def test_time_detected(self):
        _, time = _detect_date_time("At 7:30 I had breakfast")
        assert time == datetime.time(7, 30)

    def test_pm_time_detected(self):
        _, time = _detect_date_time("Dinner at 6:30 pm")
        assert time == datetime.time(18, 30)


class TestParserSplitting:
    def test_splits_multiple_items(self, parser):
        result = parser.parse("Lunch was a chicken salad wrap, some crisps, and a Diet Coke")
        assert len(result.food_items) >= 2

    def test_preserves_raw_text(self, parser):
        raw = "Dinner was spaghetti bolognese, garlic bread, and a glass of red wine"
        result = parser.parse(raw)
        assert result.raw_entry_text == raw

    def test_scrambled_eggs_breakfast(self, parser):
        result = parser.parse(
            "For breakfast I had two scrambled eggs with butter, one piece of sourdough toast, and coffee with milk"
        )
        assert result.meal_type == "breakfast"
        item_names = [item.item_name.lower() for item in result.food_items]
        assert any("egg" in name or "scrambled" in name for name in item_names)

    def test_dinner_entry(self, parser):
        result = parser.parse("Dinner was spaghetti bolognese, garlic bread, and a glass of red wine")
        assert result.meal_type == "dinner"
        assert len(result.food_items) >= 2

    def test_snack_entry(self, parser):
        result = parser.parse("Snack: apple and peanut butter")
        assert result.meal_type == "snack"

    def test_cereal_entry(self, parser):
        result = parser.parse("I had cereal, probably Cheerios, with semi skimmed milk")
        assert len(result.food_items) >= 1

    def test_confidence_is_between_0_and_1(self, parser):
        result = parser.parse("Lunch was a sandwich")
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_no_raw_text_modification(self, parser):
        """Raw transcription must never be modified."""
        raw = "I had cereal, probably Cheerios, with semi skimmed milk"
        result = parser.parse(raw)
        assert result.raw_entry_text == raw
