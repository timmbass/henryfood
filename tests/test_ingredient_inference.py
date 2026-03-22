"""Tests for the rule-based ingredient inferencer."""
from __future__ import annotations

import pytest

from app.services.ingredient_inference import RuleBasedInferencer


@pytest.fixture
def inferencer():
    return RuleBasedInferencer()


class TestExactMatches:
    def test_scrambled_eggs(self, inferencer):
        ingredients = inferencer.infer("scrambled eggs")
        names = [i.ingredient_name for i in ingredients]
        assert "eggs" in names
        assert "butter" in names

    def test_spaghetti_bolognese(self, inferencer):
        ingredients = inferencer.infer("spaghetti bolognese")
        names = [i.ingredient_name for i in ingredients]
        assert "spaghetti" in names
        assert "minced beef" in names
        assert "tomato sauce" in names

    def test_garlic_bread(self, inferencer):
        ingredients = inferencer.infer("garlic bread")
        names = [i.ingredient_name for i in ingredients]
        assert "bread" in names
        assert "garlic" in names
        assert "butter" in names

    def test_chicken_salad_wrap(self, inferencer):
        ingredients = inferencer.infer("chicken salad wrap")
        names = [i.ingredient_name for i in ingredients]
        assert "chicken" in names
        assert "tortilla wrap" in names

    def test_apple_and_peanut_butter(self, inferencer):
        ingredients = inferencer.infer("apple and peanut butter")
        names = [i.ingredient_name for i in ingredients]
        assert "apple" in names
        assert "peanut butter" in names

    def test_coffee_with_milk(self, inferencer):
        ingredients = inferencer.infer("coffee with milk")
        names = [i.ingredient_name for i in ingredients]
        assert "coffee" in names
        assert "milk" in names


class TestPartialMatches:
    def test_partial_egg_match(self, inferencer):
        """Unknown egg dish should still detect eggs."""
        ingredients = inferencer.infer("baked eggs")
        names = [i.ingredient_name for i in ingredients]
        assert "eggs" in names

    def test_partial_pasta_match(self, inferencer):
        ingredients = inferencer.infer("pasta arrabiata")
        names = [i.ingredient_name for i in ingredients]
        assert len(names) > 0

    def test_unknown_food_returns_list(self, inferencer):
        """Unknown food should return an empty list, not raise."""
        ingredients = inferencer.infer("xyzzy flurble")
        assert isinstance(ingredients, list)

    def test_confidence_between_0_and_1(self, inferencer):
        ingredients = inferencer.infer("scrambled eggs")
        for ing in ingredients:
            assert 0.0 <= ing.confidence <= 1.0

    def test_source_method_set(self, inferencer):
        ingredients = inferencer.infer("garlic bread")
        for ing in ingredients:
            assert ing.source_method in {"rule_based", "model_inferred", "user_confirmed"}
