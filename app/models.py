"""SQLAlchemy ORM models for the food diary."""
from __future__ import annotations

import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Date, Time
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime.datetime:
    """Return current UTC time as a naive datetime (SQLite compatible)."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class FoodLogEntry(Base):
    """Top-level food diary entry (one per meal/eating occasion)."""

    __tablename__ = "food_log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    entry_date = Column(Date, nullable=True)
    entry_time = Column(Time, nullable=True)
    meal_type = Column(String(50), nullable=True)  # breakfast/lunch/dinner/snack
    raw_entry_text = Column(Text, nullable=False)
    cleaned_entry_text = Column(Text, nullable=True)
    status = Column(String(20), default="draft", nullable=False)  # draft/confirmed/edited
    overall_confidence = Column(Float, nullable=True)
    processing_notes = Column(Text, nullable=True)

    food_items = relationship(
        "FoodItem", back_populates="food_log_entry", cascade="all, delete-orphan"
    )


class FoodItem(Base):
    """Individual food item within a diary entry."""

    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    food_log_entry_id = Column(Integer, ForeignKey("food_log_entries.id"), nullable=False)
    item_name = Column(String(255), nullable=False)
    normalized_item_name = Column(String(255), nullable=True)
    portion_text = Column(String(255), nullable=True)
    estimated_quantity = Column(Float, nullable=True)
    estimated_unit = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    item_order = Column(Integer, nullable=True)

    food_log_entry = relationship("FoodLogEntry", back_populates="food_items")
    ingredients = relationship(
        "Ingredient", back_populates="food_item", cascade="all, delete-orphan"
    )
    nutrition_estimate = relationship(
        "NutritionEstimate", back_populates="food_item",
        cascade="all, delete-orphan", uselist=False
    )


class Ingredient(Base):
    """Individual ingredient within a food item."""

    __tablename__ = "ingredients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    food_item_id = Column(Integer, ForeignKey("food_items.id"), nullable=False)
    ingredient_name = Column(String(255), nullable=False)
    normalized_ingredient_name = Column(String(255), nullable=True)
    estimated_amount = Column(Float, nullable=True)
    estimated_unit = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    source_method = Column(String(50), nullable=True)  # rule_based/model_inferred/user_confirmed

    food_item = relationship("FoodItem", back_populates="ingredients")


class NutritionEstimate(Base):
    """Optional nutritional estimate for a food item."""

    __tablename__ = "nutrition_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    food_item_id = Column(Integer, ForeignKey("food_items.id"), nullable=False, unique=True)
    calories_est = Column(Float, nullable=True)
    protein_g_est = Column(Float, nullable=True)
    carbs_g_est = Column(Float, nullable=True)
    fat_g_est = Column(Float, nullable=True)
    fiber_g_est = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    estimation_method = Column(String(100), nullable=True)

    food_item = relationship("FoodItem", back_populates="nutrition_estimate")
