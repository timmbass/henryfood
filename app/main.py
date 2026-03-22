"""
Food Diary CLI — main entry point.

Usage:
    python -m app.main enter-food-diary
    python -m app.main list-entries
    python -m app.main show-entry --id 1
    python -m app.main export-csv
    python -m app.main review-drafts
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from app.config import get_db_path
from app.db import get_session
from app.schemas import FoodItemSchema, FoodLogEntrySchema, IngredientSchema, ParsedMealSchema
from app.services.parser import get_parser
from app.services.storage import SQLiteStorageBackend
from app.services.transcription import get_stt_provider
from app.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()
app = typer.Typer(
    name="food-diary",
    help="Local-first food diary assistant — enter, review, and export your meals.",
    add_completion=False,
)


def _get_storage() -> SQLiteStorageBackend:
    """Create and return a storage backend."""
    db_path = get_db_path()
    session = get_session(db_path)
    return SQLiteStorageBackend(session)


def _print_parsed_entry(parsed: ParsedMealSchema) -> None:
    """Print a parsed meal entry in a human-readable format."""
    console.print(Panel(
        f"[bold]Raw transcription:[/bold] {parsed.raw_entry_text}\n"
        f"[bold]Meal type:[/bold] {parsed.meal_type or 'unknown'}\n"
        f"[bold]Date:[/bold] {parsed.entry_date or 'today'}\n"
        f"[bold]Time:[/bold] {parsed.entry_time or 'not specified'}\n"
        f"[bold]Confidence:[/bold] {(parsed.overall_confidence or 0) * 100:.0f}%\n"
        + (f"[yellow]Notes: {parsed.processing_notes}[/yellow]" if parsed.processing_notes else ""),
        title="📋 Parsed Meal Entry",
        border_style="blue",
    ))

    for item in parsed.food_items:
        qty_str = ""
        if item.estimated_quantity is not None:
            qty_str = f" ({item.estimated_quantity}"
            if item.estimated_unit:
                qty_str += f" {item.estimated_unit}"
            qty_str += ")"
        console.print(f"\n  🍽️  [bold]{item.item_name}[/bold]{qty_str} — {int((item.confidence or 0.5) * 100)}% confidence")
        if item.portion_text:
            console.print(f"      Portion: {item.portion_text}")
        if item.ingredients:
            console.print("      Inferred ingredients:")
            for ing in item.ingredients:
                source_label = {"rule_based": "📖", "model_inferred": "🤖", "user_confirmed": "✅"}.get(
                    ing.source_method, "?"
                )
                console.print(
                    f"        {source_label} {ing.ingredient_name} "
                    f"({int((ing.confidence or 0.5) * 100)}%)"
                )
        else:
            console.print("      [dim]No ingredients inferred[/dim]")


def _print_entry_summary(entry: FoodLogEntrySchema) -> None:
    """Print a summary table for an entry."""
    status_styles = {"confirmed": "green", "draft": "yellow", "edited": "cyan"}
    style = status_styles.get(entry.status, "white")

    console.print(Panel(
        f"[bold]ID:[/bold] {entry.id}   "
        f"[bold]Date:[/bold] {entry.entry_date}   "
        f"[bold]Meal:[/bold] {entry.meal_type or 'unknown'}   "
        f"[bold]Status:[/bold] [{style}]{entry.status}[/{style}]\n"
        f"[bold]Raw text:[/bold] {entry.raw_entry_text}\n"
        f"[bold]Confidence:[/bold] {(entry.overall_confidence or 0) * 100:.0f}%"
        + (f"\n[yellow]Notes: {entry.processing_notes}[/yellow]" if entry.processing_notes else ""),
        title=f"Entry #{entry.id}",
        border_style=style,
    ))
    for item in entry.food_items:
        console.print(f"  🍽️  [bold]{item.item_name}[/bold]")
        for ing in item.ingredients:
            console.print(f"      • {ing.ingredient_name} ({int((ing.confidence or 0.5)*100)}%)")


def _edit_parsed_entry(parsed: ParsedMealSchema) -> ParsedMealSchema:
    """Interactive edit loop for a parsed meal entry before saving."""
    while True:
        console.print("\n[bold]Edit options:[/bold]")
        console.print("  1. Rename a food item")
        console.print("  2. Delete a food item")
        console.print("  3. Add an ingredient to an item")
        console.print("  4. Delete an ingredient from an item")
        console.print("  5. Change meal type")
        console.print("  6. Done editing")
        choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6"], default="6")

        if choice == "1":
            # Rename food item
            for i, item in enumerate(parsed.food_items):
                console.print(f"  {i + 1}. {item.item_name}")
            idx_str = Prompt.ask("Item number to rename", default="1")
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(parsed.food_items):
                    new_name = Prompt.ask("New name", default=parsed.food_items[idx].item_name)
                    items = list(parsed.food_items)
                    old_item = items[idx]
                    items[idx] = FoodItemSchema(
                        item_name=new_name,
                        normalized_item_name=new_name.lower(),
                        portion_text=old_item.portion_text,
                        estimated_quantity=old_item.estimated_quantity,
                        estimated_unit=old_item.estimated_unit,
                        confidence=old_item.confidence,
                        item_order=old_item.item_order,
                        ingredients=old_item.ingredients,
                    )
                    parsed = ParsedMealSchema(
                        raw_entry_text=parsed.raw_entry_text,
                        cleaned_entry_text=parsed.cleaned_entry_text,
                        entry_date=parsed.entry_date,
                        entry_time=parsed.entry_time,
                        meal_type=parsed.meal_type,
                        overall_confidence=parsed.overall_confidence,
                        processing_notes=parsed.processing_notes,
                        food_items=items,
                    )
                    console.print(f"[green]Renamed to '{new_name}'[/green]")
            except ValueError:
                console.print("[red]Invalid number[/red]")

        elif choice == "2":
            # Delete food item
            for i, item in enumerate(parsed.food_items):
                console.print(f"  {i + 1}. {item.item_name}")
            idx_str = Prompt.ask("Item number to delete", default="1")
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(parsed.food_items):
                    items = list(parsed.food_items)
                    removed = items.pop(idx)
                    parsed = ParsedMealSchema(
                        raw_entry_text=parsed.raw_entry_text,
                        cleaned_entry_text=parsed.cleaned_entry_text,
                        entry_date=parsed.entry_date,
                        entry_time=parsed.entry_time,
                        meal_type=parsed.meal_type,
                        overall_confidence=parsed.overall_confidence,
                        processing_notes=parsed.processing_notes,
                        food_items=items,
                    )
                    console.print(f"[green]Removed '{removed.item_name}'[/green]")
            except ValueError:
                console.print("[red]Invalid number[/red]")

        elif choice == "3":
            # Add ingredient
            for i, item in enumerate(parsed.food_items):
                console.print(f"  {i + 1}. {item.item_name}")
            idx_str = Prompt.ask("Add ingredient to item number", default="1")
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(parsed.food_items):
                    ing_name = Prompt.ask("Ingredient name")
                    items = list(parsed.food_items)
                    old_item = items[idx]
                    new_ings = list(old_item.ingredients) + [
                        IngredientSchema(
                            ingredient_name=ing_name,
                            normalized_ingredient_name=ing_name.lower(),
                            confidence=1.0,
                            source_method="user_confirmed",
                        )
                    ]
                    items[idx] = FoodItemSchema(
                        item_name=old_item.item_name,
                        normalized_item_name=old_item.normalized_item_name,
                        portion_text=old_item.portion_text,
                        estimated_quantity=old_item.estimated_quantity,
                        estimated_unit=old_item.estimated_unit,
                        confidence=old_item.confidence,
                        item_order=old_item.item_order,
                        ingredients=new_ings,
                    )
                    parsed = ParsedMealSchema(
                        raw_entry_text=parsed.raw_entry_text,
                        cleaned_entry_text=parsed.cleaned_entry_text,
                        entry_date=parsed.entry_date,
                        entry_time=parsed.entry_time,
                        meal_type=parsed.meal_type,
                        overall_confidence=parsed.overall_confidence,
                        processing_notes=parsed.processing_notes,
                        food_items=items,
                    )
                    console.print(f"[green]Added ingredient '{ing_name}'[/green]")
            except ValueError:
                console.print("[red]Invalid number[/red]")

        elif choice == "4":
            # Delete ingredient
            for i, item in enumerate(parsed.food_items):
                console.print(f"  {i + 1}. {item.item_name}")
            idx_str = Prompt.ask("From item number", default="1")
            try:
                item_idx = int(idx_str) - 1
                if 0 <= item_idx < len(parsed.food_items):
                    old_item = parsed.food_items[item_idx]
                    for j, ing in enumerate(old_item.ingredients):
                        console.print(f"    {j + 1}. {ing.ingredient_name}")
                    ing_idx_str = Prompt.ask("Ingredient number to delete", default="1")
                    ing_idx = int(ing_idx_str) - 1
                    items = list(parsed.food_items)
                    new_ings = list(old_item.ingredients)
                    if 0 <= ing_idx < len(new_ings):
                        removed_ing = new_ings.pop(ing_idx)
                        items[item_idx] = FoodItemSchema(
                            item_name=old_item.item_name,
                            normalized_item_name=old_item.normalized_item_name,
                            portion_text=old_item.portion_text,
                            estimated_quantity=old_item.estimated_quantity,
                            estimated_unit=old_item.estimated_unit,
                            confidence=old_item.confidence,
                            item_order=old_item.item_order,
                            ingredients=new_ings,
                        )
                        parsed = ParsedMealSchema(
                            raw_entry_text=parsed.raw_entry_text,
                            cleaned_entry_text=parsed.cleaned_entry_text,
                            entry_date=parsed.entry_date,
                            entry_time=parsed.entry_time,
                            meal_type=parsed.meal_type,
                            overall_confidence=parsed.overall_confidence,
                            processing_notes=parsed.processing_notes,
                            food_items=items,
                        )
                        console.print(f"[green]Removed '{removed_ing.ingredient_name}'[/green]")
            except ValueError:
                console.print("[red]Invalid number[/red]")

        elif choice == "5":
            # Change meal type
            meal_type = Prompt.ask(
                "Meal type",
                choices=["breakfast", "lunch", "dinner", "snack", "other", ""],
                default=parsed.meal_type or "",
            )
            parsed = ParsedMealSchema(
                raw_entry_text=parsed.raw_entry_text,
                cleaned_entry_text=parsed.cleaned_entry_text,
                entry_date=parsed.entry_date,
                entry_time=parsed.entry_time,
                meal_type=meal_type or None,
                overall_confidence=parsed.overall_confidence,
                processing_notes=parsed.processing_notes,
                food_items=parsed.food_items,
            )
            console.print(f"[green]Meal type set to '{meal_type}'[/green]")

        elif choice == "6":
            break

    return parsed


@app.command("enter-food-diary")
def enter_food_diary(
    use_stt: bool = typer.Option(False, "--stt", help="Use speech-to-text (requires faster-whisper)"),
    use_ollama: bool = typer.Option(False, "--ollama", help="Use Ollama for parsing"),
    ollama_model: str = typer.Option("llama3", "--model", help="Ollama model to use"),
    draft: bool = typer.Option(False, "--draft", help="Save as draft without review"),
) -> None:
    """Enter a new food diary entry via typed or spoken input."""
    console.print(Panel(
        "[bold blue]🥗 Food Diary — New Entry[/bold blue]\n"
        "I'll help you log what you ate.",
        border_style="blue",
    ))

    # Get input (typed or STT)
    stt_provider = get_stt_provider(use_stt=use_stt)
    raw_text = stt_provider.transcribe("Tell me what you ate: ")

    if not raw_text.strip():
        console.print("[yellow]No input received. Exiting.[/yellow]")
        raise typer.Exit()

    # Parse
    parser = get_parser(use_ollama=use_ollama, model=ollama_model if use_ollama else "llama3")
    with console.status("Parsing your entry..."):
        parsed = parser.parse(raw_text)

    # Review
    _print_parsed_entry(parsed)

    if draft:
        # Save as draft without review
        storage = _get_storage()
        entry_id = storage.save_entry(parsed, status="draft")
        console.print(f"\n[green]✅ Saved as draft (entry #{entry_id})[/green]")
        return

    # Review/edit/confirm loop
    while True:
        console.print("\n[bold]What would you like to do?[/bold]")
        action = Prompt.ask(
            "Action",
            choices=["confirm", "edit", "discard"],
            default="confirm",
        )
        if action == "confirm":
            storage = _get_storage()
            entry_id = storage.save_entry(parsed, status="confirmed")
            console.print(f"\n[green]✅ Entry saved (#{entry_id})[/green]")
            break
        elif action == "edit":
            parsed = _edit_parsed_entry(parsed)
            _print_parsed_entry(parsed)
        elif action == "discard":
            console.print("[yellow]Entry discarded.[/yellow]")
            break


@app.command("list-entries")
def list_entries(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status: draft/confirmed/edited"),
    limit: int = typer.Option(20, "--limit", help="Maximum entries to show"),
) -> None:
    """List recent food diary entries."""
    storage = _get_storage()
    entries = storage.list_entries(status=status, limit=limit)

    if not entries:
        console.print("[yellow]No entries found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Food Diary Entries ({len(entries)} shown)")
    table.add_column("ID", style="dim")
    table.add_column("Date")
    table.add_column("Meal")
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Items")
    table.add_column("Raw Text (truncated)")

    status_styles = {"confirmed": "green", "draft": "yellow", "edited": "cyan"}
    for entry in entries:
        style = status_styles.get(entry.status, "white")
        table.add_row(
            str(entry.id),
            str(entry.entry_date or ""),
            entry.meal_type or "—",
            f"[{style}]{entry.status}[/{style}]",
            f"{(entry.overall_confidence or 0) * 100:.0f}%",
            str(len(entry.food_items)),
            entry.raw_entry_text[:50] + ("…" if len(entry.raw_entry_text) > 50 else ""),
        )

    console.print(table)


@app.command("show-entry")
def show_entry(
    entry_id: int = typer.Option(..., "--id", help="Entry ID to show"),
) -> None:
    """Show details of a specific food diary entry."""
    storage = _get_storage()
    entry = storage.get_entry(entry_id)
    if entry is None:
        console.print(f"[red]Entry #{entry_id} not found.[/red]")
        raise typer.Exit(1)
    _print_entry_summary(entry)


@app.command("export-csv")
def export_csv(
    output: Path = typer.Option(Path("data/food_diary_export.csv"), "--output", help="Output CSV path"),
) -> None:
    """Export all food diary entries to CSV."""
    storage = _get_storage()
    rows = storage.export_csv(output)
    console.print(f"[green]✅ Exported {rows} rows to {output}[/green]")


@app.command("review-drafts")
def review_drafts() -> None:
    """Review and confirm or discard draft entries."""
    storage = _get_storage()
    drafts = storage.list_entries(status="draft")

    if not drafts:
        console.print("[yellow]No draft entries to review.[/yellow]")
        raise typer.Exit()

    console.print(f"[bold]Found {len(drafts)} draft(s) to review.[/bold]\n")

    for entry in drafts:
        _print_entry_summary(entry)
        action = Prompt.ask(
            f"Action for entry #{entry.id}",
            choices=["confirm", "edit", "discard", "skip"],
            default="skip",
        )
        if action == "confirm":
            storage.update_status(entry.id, "confirmed")
            console.print(f"[green]✅ Entry #{entry.id} confirmed.[/green]")
        elif action == "discard":
            storage.delete_entry(entry.id)
            console.print(f"[yellow]Entry #{entry.id} discarded.[/yellow]")
        elif action == "edit":
            console.print("[dim]Full editing not available in review mode. Use enter-food-diary for a new entry.[/dim]")
        elif action == "skip":
            console.print(f"[dim]Skipped entry #{entry.id}.[/dim]")


if __name__ == "__main__":
    app()
