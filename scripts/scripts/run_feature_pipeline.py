"""Run feature engineering pipeline (CLI)"""

import argparse
from src.features import lag_features, rolling_loads, fuzzy_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/diary.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()
    print("Would run feature pipeline on", args.input)


if __name__ == "__main__":
    main()
