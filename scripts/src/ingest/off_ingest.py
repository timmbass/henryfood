"""Open Food Facts ingest stub"""

import argparse


def ingest_off(path: str):
    print(f"Stub: ingest_off called for {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/external/off.parquet")
    args = parser.parse_args()
    ingest_off(args.path)


if __name__ == "__main__":
    main()
