"""Tests for the SQLite storage backend."""
from __future__ import annotations

import datetime
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.schemas import FoodItemSchema, IngredientSchema, ParsedMealSchema
from app.services.storage import SQLiteStorageBackend


@pytest.fixture
def session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    
    # Enable foreign keys
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_pragmas(conn, _):
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    yield s
    s.close()


@pytest.fixture
def storage(session):
    return SQLiteStorageBackend(session)


@pytest.fixture
def sample_parsed() -> ParsedMealSchema:
    return ParsedMealSchema(
        raw_entry_text="Dinner was spaghetti bolognese, garlic bread, and a glass of red wine",
        cleaned_entry_text="dinner spaghetti bolognese garlic bread glass red wine",
        entry_date=datetime.date(2026, 1, 15),
        meal_type="dinner",
        overall_confidence=0.75,
        food_items=[
            FoodItemSchema(
                item_name="spaghetti bolognese",
                normalized_item_name="spaghetti bolognese",
                item_order=1,
                confidence=0.9,
                ingredients=[
                    IngredientSchema(ingredient_name="spaghetti", confidence=0.95, source_method="rule_based"),
                    IngredientSchema(ingredient_name="minced beef", confidence=0.90, source_method="rule_based"),
                ],
            ),
            FoodItemSchema(
                item_name="garlic bread",
                normalized_item_name="garlic bread",
                item_order=2,
                confidence=0.85,
                ingredients=[
                    IngredientSchema(ingredient_name="bread", confidence=0.90, source_method="rule_based"),
                    IngredientSchema(ingredient_name="garlic", confidence=0.90, source_method="rule_based"),
                ],
            ),
            FoodItemSchema(
                item_name="red wine",
                normalized_item_name="red wine",
                item_order=3,
                confidence=0.95,
                ingredients=[
                    IngredientSchema(ingredient_name="red wine", confidence=0.95, source_method="rule_based"),
                ],
            ),
        ],
    )


class TestSaveAndRetrieve:
    def test_save_entry_returns_id(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_get_entry_returns_correct_data(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        assert entry is not None
        assert entry.id == entry_id
        assert entry.raw_entry_text == sample_parsed.raw_entry_text

    def test_raw_text_preserved_exactly(self, storage, sample_parsed):
        """Raw transcription must be stored exactly as provided."""
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        assert entry.raw_entry_text == sample_parsed.raw_entry_text

    def test_food_items_saved_relationally(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        assert len(entry.food_items) == 3
        item_names = [item.item_name for item in entry.food_items]
        assert "spaghetti bolognese" in item_names
        assert "garlic bread" in item_names
        assert "red wine" in item_names

    def test_ingredients_saved_relationally(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        bol = next(item for item in entry.food_items if item.item_name == "spaghetti bolognese")
        ing_names = [ing.ingredient_name for ing in bol.ingredients]
        assert "spaghetti" in ing_names
        assert "minced beef" in ing_names

    def test_meal_type_saved(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        assert entry.meal_type == "dinner"

    def test_status_defaults_to_confirmed(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        entry = storage.get_entry(entry_id)
        assert entry.status == "confirmed"

    def test_save_as_draft(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed, status="draft")
        entry = storage.get_entry(entry_id)
        assert entry.status == "draft"


class TestListEntries:
    def test_list_returns_all_entries(self, storage, sample_parsed):
        storage.save_entry(sample_parsed)
        storage.save_entry(sample_parsed)
        entries = storage.list_entries()
        assert len(entries) >= 2

    def test_list_filtered_by_status(self, storage, sample_parsed):
        storage.save_entry(sample_parsed, status="confirmed")
        storage.save_entry(sample_parsed, status="draft")
        confirmed = storage.list_entries(status="confirmed")
        drafts = storage.list_entries(status="draft")
        assert all(e.status == "confirmed" for e in confirmed)
        assert all(e.status == "draft" for e in drafts)


class TestUpdateAndDelete:
    def test_update_status(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed, status="draft")
        storage.update_status(entry_id, "confirmed")
        entry = storage.get_entry(entry_id)
        assert entry.status == "confirmed"

    def test_delete_entry(self, storage, sample_parsed):
        entry_id = storage.save_entry(sample_parsed)
        result = storage.delete_entry(entry_id)
        assert result is True
        assert storage.get_entry(entry_id) is None

    def test_get_nonexistent_entry(self, storage):
        assert storage.get_entry(99999) is None

    def test_delete_nonexistent_entry(self, storage):
        assert storage.delete_entry(99999) is False


class TestExportCsv:
    def test_export_creates_file(self, storage, sample_parsed, tmp_path):
        storage.save_entry(sample_parsed)
        output = tmp_path / "test_export.csv"
        rows = storage.export_csv(output)
        assert output.exists()
        assert rows > 0

    def test_export_contains_header(self, storage, sample_parsed, tmp_path):
        storage.save_entry(sample_parsed)
        output = tmp_path / "test_export.csv"
        storage.export_csv(output)
        content = output.read_text()
        assert "entry_id" in content
        assert "food_item" in content
        assert "ingredient" in content
