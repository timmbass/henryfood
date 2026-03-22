"""
Optional Streamlit UI for the food diary app.

Run with: streamlit run app/ui/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from app.config import get_db_path
from app.db import get_session
from app.services.parser import get_parser
from app.services.storage import SQLiteStorageBackend


def main():
    st.set_page_config(page_title="HenryFood Diary", page_icon="🥗", layout="wide")
    st.title("🥗 HenryFood — Food Diary")

    db_path = get_db_path()
    session = get_session(db_path)
    storage = SQLiteStorageBackend(session)
    parser = get_parser()

    tab1, tab2, tab3 = st.tabs(["📝 Add Entry", "📋 View Entries", "📊 Export"])

    with tab1:
        st.header("Add Food Diary Entry")
        raw_text = st.text_area(
            "Tell me what you ate:",
            placeholder="e.g. For breakfast I had two scrambled eggs with butter, sourdough toast, and coffee with milk",
            height=100,
        )
        if st.button("Parse Entry", type="primary"):
            if raw_text.strip():
                with st.spinner("Parsing..."):
                    parsed = parser.parse(raw_text)

                st.subheader("Parsed Result")
                col1, col2, col3 = st.columns(3)
                col1.metric("Meal Type", parsed.meal_type or "Unknown")
                col2.metric("Date", str(parsed.entry_date or "Today"))
                col3.metric("Confidence", f"{(parsed.overall_confidence or 0) * 100:.0f}%")

                if parsed.processing_notes:
                    st.warning(parsed.processing_notes)

                st.markdown("**Raw text (preserved):**")
                st.code(parsed.raw_entry_text)

                st.markdown("**Food Items:**")
                for item in parsed.food_items:
                    with st.expander(f"🍽️ {item.item_name} ({item.portion_text or 'no portion'})"):
                        if item.ingredients:
                            for ing in item.ingredients:
                                conf_pct = int((ing.confidence or 0.5) * 100)
                                st.write(f"• **{ing.ingredient_name}** — {conf_pct}% confidence ({ing.source_method})")
                        else:
                            st.write("No ingredients inferred.")

                col_save, col_discard = st.columns(2)
                with col_save:
                    if st.button("✅ Confirm & Save", key="save_btn"):
                        entry_id = storage.save_entry(parsed, status="confirmed")
                        st.success(f"Saved entry #{entry_id}!")
                        st.rerun()
                with col_discard:
                    if st.button("🗑️ Discard", key="discard_btn"):
                        st.info("Entry discarded.")
            else:
                st.error("Please enter some text first.")

    with tab2:
        st.header("Recent Entries")
        entries = storage.list_entries(limit=20)
        if not entries:
            st.info("No entries yet. Add one in the 'Add Entry' tab.")
        for entry in entries:
            status_emoji = {"confirmed": "✅", "draft": "📝", "edited": "✏️"}.get(entry.status, "❓")
            with st.expander(
                f"{status_emoji} #{entry.id} — {entry.entry_date} {entry.meal_type or ''} "
                f"({len(entry.food_items)} items)"
            ):
                st.markdown(f"**Raw text:** {entry.raw_entry_text}")
                st.markdown(f"**Status:** {entry.status} | **Confidence:** {(entry.overall_confidence or 0) * 100:.0f}%")
                for item in entry.food_items:
                    st.markdown(f"- **{item.item_name}**")
                    for ing in item.ingredients:
                        st.markdown(f"  - {ing.ingredient_name} ({int((ing.confidence or 0.5)*100)}%)")

    with tab3:
        st.header("Export to CSV")
        export_path = Path("data/food_diary_export.csv")
        if st.button("Export CSV"):
            rows = storage.export_csv(export_path)
            st.success(f"Exported {rows} rows to `{export_path}`")
            if export_path.exists():
                with open(export_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download CSV", data=f,
                        file_name="food_diary_export.csv", mime="text/csv"
                    )


if __name__ == "__main__":
    main()
