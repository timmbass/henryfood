"""Train Random Forest stub"""

import argparse


def train_rf(features_path: str, model_out: str):
    print(f"Stub: train_rf called for {features_path} -> {model_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--out", default="models/rf.pkl")
    args = parser.parse_args()
    train_rf(args.features, args.out)


if __name__ == "__main__":
    main()
