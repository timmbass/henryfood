"""Clustering stub"""

import argparse


def run_clustering(features_path: str):
    print(f"Stub: run_clustering called for {features_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    args = parser.parse_args()
    run_clustering(args.features)


if __name__ == "__main__":
    main()
