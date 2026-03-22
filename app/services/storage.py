"""
Storage service for food diary entries.

Provides an abstract StorageBackend interface and a SQLite implementation.
"""
from __future__ import annotations

import csv
import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.models import FoodLogEntry, FoodItem, Ingredient, NutritionEstimate
from app.schemas import (
    FoodItemSchema,
    FoodLogEntrySchema,
    IngredientSchema,
    NutritionEstimateSchema,
    ParsedMealSchema,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def save_entry(self, parsed: ParsedMealSchema, status: str = "confirmed") -> int:
        """Save a parsed meal entry and return the new entry ID."""
        ...

    @abstractmethod
    def get_entry(self, entry_id: int) -> Optional[FoodLogEntrySchema]:
        """Retrieve an entry by ID."""
        ...

    @abstractmethod
    def list_entries(self, status: Optional[str] = None, limit: int = 50) -> list[FoodLogEntrySchema]:
        """List entries, optionally filtered by status."""
        ...

    @abstractmethod
    def update_status(self, entry_id: int, status: str) -> bool:
        """Update the status of an entry."""
        ...

    @abstractmethod
    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry."""
        ...

    @abstractmethod
    def export_csv(self, output_path: Path) -> int:
        """Export all entries to CSV and return the number of rows written."""
        ...


class SQLiteStorageBackend(StorageBackend):
    """SQLite-backed storage implementation using SQLAlchemy."""

    def __init__(self, session: Session):
        self.session = session

    def save_entry(self, parsed: ParsedMealSchema, status: str = "confirmed") -> int:
        """Persist a ParsedMealSchema to the database."""
        entry = FoodLogEntry(
            entry_date=parsed.entry_date,
            entry_time=parsed.entry_time,
            meal_type=parsed.meal_type,
            raw_entry_text=parsed.raw_entry_text,
            cleaned_entry_text=parsed.cleaned_entry_text,
            status=status,
            overall_confidence=parsed.overall_confidence,
            processing_notes=parsed.processing_notes,
        )
        self.session.add(entry)
        self.session.flush()  # Get the ID before adding children

        for item_schema in parsed.food_items:
            item = FoodItem(
                food_log_entry_id=entry.id,
                item_name=item_schema.item_name,
                normalized_item_name=item_schema.normalized_item_name,
                portion_text=item_schema.portion_text,
                estimated_quantity=item_schema.estimated_quantity,
                estimated_unit=item_schema.estimated_unit,
                confidence=item_schema.confidence,
                item_order=item_schema.item_order,
            )
            self.session.add(item)
            self.session.flush()

            for ing_schema in item_schema.ingredients:
                ingredient = Ingredient(
                    food_item_id=item.id,
                    ingredient_name=ing_schema.ingredient_name,
                    normalized_ingredient_name=ing_schema.normalized_ingredient_name,
                    estimated_amount=ing_schema.estimated_amount,
                    estimated_unit=ing_schema.estimated_unit,
                    confidence=ing_schema.confidence,
                    source_method=ing_schema.source_method,
                )
                self.session.add(ingredient)

            if item_schema.nutrition_estimate:
                nu = item_schema.nutrition_estimate
                nutrition = NutritionEstimate(
                    food_item_id=item.id,
                    calories_est=nu.calories_est,
                    protein_g_est=nu.protein_g_est,
                    carbs_g_est=nu.carbs_g_est,
                    fat_g_est=nu.fat_g_est,
                    fiber_g_est=nu.fiber_g_est,
                    confidence=nu.confidence,
                    estimation_method=nu.estimation_method,
                )
                self.session.add(nutrition)

        self.session.commit()
        logger.info("Saved food log entry id=%d (status=%s)", entry.id, status)
        return entry.id

    def get_entry(self, entry_id: int) -> Optional[FoodLogEntrySchema]:
        """Retrieve a single entry by ID."""
        entry = self.session.get(FoodLogEntry, entry_id)
        if entry is None:
            return None
        return self._to_schema(entry)

    def list_entries(self, status: Optional[str] = None, limit: int = 50) -> list[FoodLogEntrySchema]:
        """List all entries, optionally filtered by status."""
        query = self.session.query(FoodLogEntry).order_by(FoodLogEntry.created_at.desc())
        if status:
            query = query.filter(FoodLogEntry.status == status)
        entries = query.limit(limit).all()
        return [self._to_schema(e) for e in entries]

    def update_status(self, entry_id: int, status: str) -> bool:
        """Update entry status."""
        entry = self.session.get(FoodLogEntry, entry_id)
        if entry is None:
            return False
        entry.status = status
        self.session.commit()
        return True

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry and all its children."""
        entry = self.session.get(FoodLogEntry, entry_id)
        if entry is None:
            return False
        self.session.delete(entry)
        self.session.commit()
        logger.info("Deleted food log entry id=%d", entry_id)
        return True

    def export_csv(self, output_path: Path) -> int:
        """Export all confirmed entries to CSV."""
        entries = self.session.query(FoodLogEntry).order_by(FoodLogEntry.entry_date).all()
        rows_written = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_id", "created_at", "entry_date", "entry_time", "meal_type",
                "status", "raw_entry_text", "overall_confidence",
                "food_item", "portion_text", "estimated_quantity", "estimated_unit",
                "item_confidence", "ingredient", "ingredient_confidence", "ingredient_source",
            ])
            for entry in entries:
                for item in entry.food_items:
                    if item.ingredients:
                        for ing in item.ingredients:
                            writer.writerow([
                                entry.id, entry.created_at, entry.entry_date,
                                entry.entry_time, entry.meal_type, entry.status,
                                entry.raw_entry_text, entry.overall_confidence,
                                item.item_name, item.portion_text,
                                item.estimated_quantity, item.estimated_unit,
                                item.confidence, ing.ingredient_name,
                                ing.confidence, ing.source_method,
                            ])
                            rows_written += 1
                    else:
                        writer.writerow([
                            entry.id, entry.created_at, entry.entry_date,
                            entry.entry_time, entry.meal_type, entry.status,
                            entry.raw_entry_text, entry.overall_confidence,
                            item.item_name, item.portion_text,
                            item.estimated_quantity, item.estimated_unit,
                            item.confidence, None, None, None,
                        ])
                        rows_written += 1
        logger.info("Exported %d rows to %s", rows_written, output_path)
        return rows_written

    def _to_schema(self, entry: FoodLogEntry) -> FoodLogEntrySchema:
        """Convert ORM model to Pydantic schema."""
        food_items = []
        for item in entry.food_items:
            ingredients = [
                IngredientSchema(
                    ingredient_name=ing.ingredient_name,
                    normalized_ingredient_name=ing.normalized_ingredient_name,
                    estimated_amount=ing.estimated_amount,
                    estimated_unit=ing.estimated_unit,
                    confidence=ing.confidence or 0.5,
                    source_method=ing.source_method or "rule_based",
                )
                for ing in item.ingredients
            ]
            nu = None
            if item.nutrition_estimate:
                n = item.nutrition_estimate
                nu = NutritionEstimateSchema(
                    calories_est=n.calories_est,
                    protein_g_est=n.protein_g_est,
                    carbs_g_est=n.carbs_g_est,
                    fat_g_est=n.fat_g_est,
                    fiber_g_est=n.fiber_g_est,
                    confidence=n.confidence or 0.3,
                    estimation_method=n.estimation_method,
                )
            food_items.append(
                FoodItemSchema(
                    item_name=item.item_name,
                    normalized_item_name=item.normalized_item_name,
                    portion_text=item.portion_text,
                    estimated_quantity=item.estimated_quantity,
                    estimated_unit=item.estimated_unit,
                    confidence=item.confidence or 0.5,
                    item_order=item.item_order,
                    ingredients=ingredients,
                    nutrition_estimate=nu,
                )
            )
        return FoodLogEntrySchema(
            id=entry.id,
            created_at=entry.created_at,
            entry_date=entry.entry_date,
            entry_time=entry.entry_time,
            meal_type=entry.meal_type,
            raw_entry_text=entry.raw_entry_text,
            cleaned_entry_text=entry.cleaned_entry_text,
            status=entry.status,
            overall_confidence=entry.overall_confidence,
            processing_notes=entry.processing_notes,
            food_items=food_items,
        )
