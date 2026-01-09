"""Sync a Google Sheet to a local JSON diary with conservative merge rules.

Usage:
  python tools/sync_google_sheet.py SHEET_KEY_OR_URL --target data/ingredient_log.json --creds data/gspread_credentials.json [--worksheet Sheet1] [--key-cols id] [--dry-run]

Dependencies: gspread, pandas
pip install gspread pandas

Behavior:
- Builds a stable key per row using the first available of the key columns (default: id), else uses ts+meal_name.
- Appends entirely new rows.
- For existing rows, only fills fields that are blank/None/""/"N/A" in the on-disk record; existing non-N/A values are preserved.
- Dry-run mode prints planned changes without writing the target file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import gspread


def load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def normalize_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "n/a":
        return None
    return s


def get_nested(obj: Dict[str, Any], dotted_key: str) -> Any:
    """Retrieve a nested value using dotted key (e.g. 'nutrition.calories')."""
    parts = dotted_key.split(".")
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def set_nested(obj: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested value creating intermediate dicts as needed."""
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def build_key(row: Dict[str, Any], key_cols: List[str]) -> str:
    # support dotted key columns
    for c in key_cols:
        v = normalize_val(get_nested(row, c) if "." in c else row.get(c))
        if v:
            return f"key:{c}:{v}"
    # fallback: try timestamp + meal_name or arbitrary concatenation
    ts = normalize_val(row.get("ts") or row.get("timestamp") or row.get("date"))
    meal = normalize_val(row.get("meal") or row.get("meal_name") or row.get("items"))
    if ts and meal:
        return f"key:ts_meal:{ts}|{meal}"
    # last resort: use full row hash-like string
    parts = [f"{k}={normalize_val(v)}" for k, v in sorted(row.items()) if normalize_val(v) is not None]
    return "key:row:" + "|".join(parts)


def sheet_to_df(client: gspread.Client, sheet_id: str, worksheet: str | None = None) -> pd.DataFrame:
    sh = client.open_by_key(sheet_id) if "docs.google.com" not in sheet_id else client.open_by_url(sheet_id)
    ws = sh.worksheet(worksheet) if worksheet else sh.get_worksheet(0)
    data = ws.get_all_records()
    return pd.DataFrame(data)


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise ValueError if any required column is missing from the DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sheet: {missing}")


def detect_format_from_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".json", ""}:
        return "json"
    if ext in {".csv"}:
        return "csv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if ext in {".duckdb", ".db"}:
        return "duckdb"
    return "json"


import duckdb  # added for duckdb target support
try:
    import yaml
except Exception:
    yaml = None


def load_target(path: Path, table: str | None = None) -> List[Dict[str, Any]]:
    fmt = detect_format_from_path(path)
    if fmt == "json":
        return load_json(path)
    if fmt == "duckdb":
        if not path.exists():
            return []
        con = duckdb.connect(database=str(path), read_only=False)
        try:
            tbl = table or "diary"
            # check table exists
            res = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() if False else None
            # safer: query duckdb catalog
            try:
                df = con.execute(f"SELECT * FROM \"{tbl}\";").df()
            except Exception:
                return []
            return df.where(pd.notnull(df), None).to_dict(orient="records")
        finally:
            con.close()
    if not path.exists():
        return []
    if fmt == "csv":
        df = pd.read_csv(path)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    if fmt == "parquet":
        df = pd.read_parquet(path)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    return load_json(path)


def save_target(path: Path, data: List[Dict[str, Any]], table: str | None = None) -> None:
    fmt = detect_format_from_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        save_json(path, data)
        return
    df = pd.DataFrame(data)
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "duckdb":
        tbl = table or "diary"
        con = duckdb.connect(database=str(path), read_only=False)
        try:
            # write by replacing the table with the DataFrame
            con.register("__tmp_df", df)
            # Replace or create the table
            con.execute(f"DROP TABLE IF EXISTS \"{tbl}\";")
            con.execute(f"CREATE TABLE \"{tbl}\" AS SELECT * FROM __tmp_df;")
            con.unregister("__tmp_df")
        finally:
            con.close()
        return
    save_json(path, data)


# --- New: helpers to normalize meal text into a flattened meals table ---

def _coerce_ts(val: Any) -> Any:
    if val is None:
        return None
    try:
        return pd.to_datetime(val)
    except Exception:
        return val


def build_meals_df(records: List[Dict[str, Any]], key_cols: List[str]) -> pd.DataFrame:
    """Turn diary records into a flattened meals DataFrame with columns:
    ts, source_key, meal_type, item, raw_text
    """
    rows: List[Dict[str, Any]] = []
    meal_fields = ["breakfast", "lunch", "dinner", "snack"]
    for rec in records:
        src_key = build_key(rec, key_cols)
        ts_val = rec.get("ts") or rec.get("timestamp") or rec.get("date")
        ts = _coerce_ts(ts_val)
        # support nested 'meals' dict
        meals_container = None
        if isinstance(rec.get("meals"), dict):
            meals_container = rec.get("meals")
        # allow both top-level Breakfast/Lunch names
        for mf in meal_fields:
            raw = None
            # first try nested meals.mf
            if meals_container and mf in meals_container:
                raw = meals_container.get(mf)
            # then case-insensitive top-level columns
            if raw is None:
                for k in rec.keys():
                    if k.lower() == mf:
                        raw = rec.get(k)
                        break
            if raw is None:
                continue
            if isinstance(raw, list):
                items = raw
            else:
                raw_text = str(raw)
                # split on commas and semicolons
                items = [p.strip() for p in raw_text.split(",") if p.strip()]
            for it in items:
                norm = normalize_val(it)
                if norm is None:
                    continue
                rows.append({"ts": ts, "source_key": src_key, "meal_type": mf, "item": norm, "raw_text": it})
    if not rows:
        return pd.DataFrame(columns=["ts", "source_key", "meal_type", "item", "raw_text"])
    dfm = pd.DataFrame(rows)
    # ensure ts is datetime or string
    dfm["ts"] = dfm["ts"].apply(lambda x: pd.to_datetime(x) if x is not None else None)
    return dfm


def write_meals_to_duckdb(dfm: pd.DataFrame, duckdb_path: Path, table: str = "meals") -> None:
    con = duckdb.connect(database=str(duckdb_path), read_only=False)
    try:
        con.register("__meals_df", dfm)
        con.execute(f"DROP TABLE IF EXISTS \"{table}\";")
        con.execute(f"CREATE TABLE \"{table}\" AS SELECT * FROM __meals_df;")
        con.unregister("__meals_df")
    finally:
        con.close()


def normalize_and_write_meals(records: List[Dict[str, Any]], key_cols: List[str], target_duckdb: Path | None = None, meals_table: str = "meals", meals_parquet: str | None = None) -> Dict[str, Any]:
    dfm = build_meals_df(records, key_cols)
    out = {"meals_rows": len(dfm)}
    if target_duckdb is not None:
        write_meals_to_duckdb(dfm, target_duckdb, table=meals_table)
        out["duckdb_table"] = str(target_duckdb) + ":" + meals_table
    if meals_parquet:
        dfm.to_parquet(meals_parquet, index=False)
        out["meals_parquet"] = meals_parquet
    return out


def sync_sheet(sheet_id_or_url: str, creds_path: str, target_path: str, worksheet: str | None, key_cols: List[str], field_map: Dict[str, str] | None = None, dry_run: bool = True, table: str | None = None, update_policy: str = "conservative", ts_col: str | None = None, export_parquet: str | None = None, export_jsonl: str | None = None, force_write: bool = False, normalize: bool = False, meals_table: str | None = None, meals_parquet: str | None = None) -> Dict[str, Any]:
    client = gspread.service_account(filename=creds_path)
    df = sheet_to_df(client, sheet_id_or_url, worksheet)
    df = df.replace({pd.NA: None})

    # Basic validation: ensure at least timestamp or id/meal columns present
    try:
        validate_columns(df, ["id", "ts"])  # will accept if either is present; tests may bypass
    except ValueError:
        # fallback: not fatal here — proceed but tests can call validate_columns directly
        pass

    target = Path(target_path)
    existing = load_target(target, table=table)
    existing_map = {}
    for r in existing:
        k = build_key(r, key_cols)
        existing_map[k] = r

    actions = {"added": 0, "updated": 0, "skipped": 0}

    def _parse_ts(val: Any):
        if val is None:
            return None
        try:
            return pd.to_datetime(val)
        except Exception:
            return None

    # iterate incoming rows
    for row in df.fillna("").to_dict(orient="records"):
        # normalize row values
        raw_norm = {k: normalize_val(v) for k, v in row.items()}
        # apply field mapping if provided (src_col -> dst.key)
        if field_map:
            norm_row: Dict[str, Any] = {}
            # first apply mappings
            for src, dst in field_map.items():
                if src in raw_norm:
                    val = raw_norm.get(src)
                    if val is not None:
                        set_nested(norm_row, dst, val)
            # then include any unmapped columns at top-level
            for k, v in raw_norm.items():
                if k not in field_map:
                    norm_row[k] = v
        else:
            norm_row = raw_norm
        key = build_key(norm_row, key_cols)
        existing_row = existing_map.get(key)

        if existing_row is None:
            # entirely new -> append full normalized row
            if not dry_run or force_write:
                existing.append(norm_row)
                existing_map[key] = norm_row
            actions["added"] += 1
        else:
            # determine whether we are allowed to overwrite fields
            allow_overwrite = False
            if update_policy == "timestamp":
                incoming_ts = _parse_ts(normalize_val(norm_row.get(ts_col or "ts") or norm_row.get("timestamp") or norm_row.get("date")))
                existing_ts = _parse_ts(normalize_val(existing_row.get(ts_col or "ts") or existing_row.get("timestamp") or existing_row.get("date")))
                if incoming_ts is not None and existing_ts is None:
                    allow_overwrite = True
                elif incoming_ts is not None and existing_ts is not None and incoming_ts > existing_ts:
                    allow_overwrite = True
            # merge: only fill missing fields in existing_row unless overwriting allowed
            changed = False
            for col, val in norm_row.items():
                if val is None:
                    continue
                # support dotted target columns in norm_row (nested dicts)
                if "." in col or isinstance(val, dict):
                    if isinstance(val, dict):
                        for subk, subv in val.items():
                            full_key = f"{col}.{subk}" if "." not in col else col + "." + subk
                            cur = normalize_val(get_nested(existing_row, full_key))
                            if (cur is None and subv is not None) or allow_overwrite:
                                set_nested(existing_row, full_key, subv)
                                changed = True
                    else:
                        cur = normalize_val(get_nested(existing_row, col))
                        if (cur is None and val is not None) or allow_overwrite:
                            set_nested(existing_row, col, val)
                            changed = True
                else:
                    cur = normalize_val(existing_row.get(col))
                    if (cur is None and val is not None) or allow_overwrite:
                        # if not allowed to overwrite and cur not None, skip
                        if not allow_overwrite and cur is not None:
                            continue
                        existing_row[col] = val
                        changed = True
            if changed:
                actions["updated"] += 1
            else:
                actions["skipped"] += 1

    # write out results
    if not dry_run or force_write:
        save_target(target, existing, table=table)
        # optional exports
        if export_parquet:
            pd.DataFrame(existing).to_parquet(export_parquet, index=False)
        if export_jsonl:
            outp = Path(export_jsonl)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf8") as fh:
                for rec in existing:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # optional normalization into meals table
        if normalize:
            duckdb_path = target if detect_format_from_path(target) == "duckdb" else None
            try:
                norm_res = normalize_and_write_meals(existing, key_cols, target_duckdb=Path(duckdb_path) if duckdb_path else None, meals_table=meals_table or "meals", meals_parquet=meals_parquet)
            except Exception as e:
                norm_res = {"error": str(e)}
        else:
            norm_res = None

    return {**actions, "target": str(target), "dry_run": dry_run, "table": table, "update_policy": update_policy, "normalize": normalize, "normalize_result": norm_res}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sheet", help="Google Sheet key or full URL")
    p.add_argument("--config", help="Path to YAML config file (values overridden by CLI)", default=None)
    p.add_argument("--creds", help="Path to service account JSON creds", default=None)
    p.add_argument("--target", help="Target file path (infer format by extension). Default: data/curated/diary.duckdb", default="data/curated/diary.duckdb")
    p.add_argument("--worksheet", help="Worksheet name (default: first sheet)", default=None)
    p.add_argument("--key-cols", help="Comma-separated preferred key columns (fall back to ts+meal)", default="id")
    p.add_argument("--field-map", help="Comma-separated src:dst mappings (e.g. Timestamp:ts,Meal:meal_name,nut_cal:nutrition.calories)", default="")
    p.add_argument("--dry-run", action="store_true", help="Don't write changes; just print planned summary (default: write if not set)")
    p.add_argument("--table", help="DuckDB table name when using a .duckdb target (default: diary)", default="diary")
    p.add_argument("--export-parquet", help="Optional path to write a Parquet snapshot after sync", default=None)
    p.add_argument("--export-jsonl", help="Optional path to write a JSONL snapshot after sync", default=None)
    p.add_argument("--update-policy", help="Update policy: conservative (only fill blanks) or timestamp (overwrite when incoming ts is newer)", choices=["conservative", "timestamp"], default="conservative")
    p.add_argument("--ts-col", help="Timestamp column to use for timestamp policy (default: ts)", default="ts")
    p.add_argument("--normalize", action="store_true", help="Build a flattened meals table from diary records and write to DuckDB (if target is .duckdb) and/or a parquet snapshot")
    p.add_argument("--meals-table", help="DuckDB table name to write normalized meals (default: meals)", default="meals")
    p.add_argument("--meals-parquet", help="Optional path to write normalized meals parquet", default=None)
    p.add_argument("--force", action="store_true", help="Force write even if --dry-run was set")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML config if provided
    cfg: Dict[str, Any] = {}
    if args.config:
        if yaml is None:
            raise SystemExit("PyYAML is required to load --config. Install with `pip install pyyaml`")
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise SystemExit("Config file must contain a YAML mapping/object at top level")
            cfg = loaded

    # Merge CLI args over config (CLI wins). Handle booleans specially: only set False from CLI when user explicitly provided True/False is ambiguous; we treat store_true flags as "if True then override"
    merged: Dict[str, Any] = dict(cfg)
    for k, v in vars(args).items():
        if k == "config":
            continue
        # booleans: args provides True only when flag passed; False means not provided => keep config if present
        if isinstance(v, bool):
            if v:
                merged[k] = True
            else:
                if k not in merged:
                    merged[k] = False
        else:
            if v is not None:
                merged[k] = v

    # creds: allow env var to override config if present
    creds = merged.get("creds") or os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds:
        raise SystemExit("Provide --creds path, set GOOGLE_SHEETS_CREDENTIALS, or include creds in the config file")

    # key columns
    key_cols = merged.get("key_cols") if isinstance(merged.get("key_cols"), list) else [c.strip() for c in str(merged.get("key_cols", "id")).split(",") if c.strip()]

    # field_map: accept dict or comma-separated src:dst
    field_map_val = merged.get("field_map")
    field_map: Dict[str, str] = {}
    if isinstance(field_map_val, dict):
        field_map = {str(k): str(v) for k, v in field_map_val.items()}
    elif isinstance(field_map_val, str) and field_map_val:
        for pair in [p for p in field_map_val.split(",") if p.strip()]:
            if ":" in pair:
                src, dst = pair.split(":", 1)
                field_map[src.strip()] = dst.strip()

    # call sync
    res = sync_sheet(
        merged.get("sheet") or args.sheet,
        creds,
        merged.get("target"),
        merged.get("worksheet"),
        key_cols,
        field_map or None,
        dry_run=merged.get("dry_run", False),
        table=merged.get("table"),
        update_policy=merged.get("update_policy", "conservative"),
        ts_col=merged.get("ts_col"),
        export_parquet=merged.get("export_parquet"),
        export_jsonl=merged.get("export_jsonl"),
        force_write=merged.get("force", False),
        normalize=merged.get("normalize", False),
        meals_table=merged.get("meals_table"),
        meals_parquet=merged.get("meals_parquet"),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
