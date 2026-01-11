#!/usr/bin/env python3
"""Inspect normalized meals in DuckDB or Parquet.

Usage:
  python scripts/tools/inspect_meals.py --db scripts/data/curated/diary.duckdb --limit 20
  python scripts/tools/inspect_meals.py --parquet scripts/data/raw/meals.parquet --limit 50
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="DuckDB file to open", default="scripts/data/curated/diary.duckdb")
    p.add_argument("--parquet", help="Parquet file to inspect instead of DuckDB", default=None)
    p.add_argument("--limit", help="Number of rows to show", type=int, default=20)
    args = p.parse_args()

    if args.parquet:
        try:
            import pandas as pd
        except Exception as e:
            print("pandas is required to read parquet:", e, file=sys.stderr)
            return 2
        df = pd.read_parquet(args.parquet)
        print(df.head(args.limit).to_string(index=False))
        return 0

    try:
        import duckdb
    except Exception as e:
        print("duckdb python package is required:", e, file=sys.stderr)
        return 2

    con = duckdb.connect(database=args.db)
    try:
        print("Tables:")
        print(con.execute("SHOW TABLES").df())
        try:
            print("\nmeals count:")
            print(con.execute("SELECT COUNT(*) AS cnt FROM meals").df())
            print("\nTop rows:")
            df = con.execute(f"SELECT * FROM meals ORDER BY ts DESC LIMIT {args.limit}").df()
            print(df.to_string(index=False))
        except Exception as qerr:
            print("Error querying meals table:", qerr, file=sys.stderr)
    finally:
        con.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
