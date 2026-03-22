"""Pydantic schemas for data validation and transfer."""
from __future__ import annotations

import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class IngredientSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ingredient_name: str
    normalized_ingredient_name: Optional[str] = None
    estimated_amount: Optional[float] = None
    estimated_unit: Optional[str] = None
    confidence: float = 0.5
    source_method: str = "rule_based"


class NutritionEstimateSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    calories_est: Optional[float] = None
    protein_g_est: Optional[float] = None
    carbs_g_est: Optional[float] = None
    fat_g_est: Optional[float] = None
    fiber_g_est: Optional[float] = None
    confidence: float = 0.3
    estimation_method: Optional[str] = None


class FoodItemSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_name: str
    normalized_item_name: Optional[str] = None
    portion_text: Optional[str] = None
    estimated_quantity: Optional[float] = None
    estimated_unit: Optional[str] = None
    confidence: float = 0.5
    item_order: Optional[int] = None
    ingredients: List[IngredientSchema] = Field(default_factory=list)
    nutrition_estimate: Optional[NutritionEstimateSchema] = None


class ParsedMealSchema(BaseModel):
    """Result of parsing a raw food diary entry."""
    model_config = ConfigDict(from_attributes=True)

    raw_entry_text: str
    cleaned_entry_text: Optional[str] = None
    entry_date: Optional[datetime.date] = None
    entry_time: Optional[datetime.time] = None
    meal_type: Optional[str] = None
    overall_confidence: float = 0.5
    processing_notes: Optional[str] = None
    food_items: List[FoodItemSchema] = Field(default_factory=list)


class FoodLogEntrySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime.datetime
    entry_date: Optional[datetime.date] = None
    entry_time: Optional[datetime.time] = None
    meal_type: Optional[str] = None
    raw_entry_text: str
    cleaned_entry_text: Optional[str] = None
    status: str
    overall_confidence: Optional[float] = None
    processing_notes: Optional[str] = None
    food_items: List[FoodItemSchema] = Field(default_factory=list)
