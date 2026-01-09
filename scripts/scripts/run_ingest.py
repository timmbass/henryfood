"""Run ingest pipeline (CLI)"""

import argparse
from src.ingest import usda_client, off_ingest, liljebo, sighi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all")
    args = parser.parse_args()
    print("Would run ingest step:", args.step)


if __name__ == "__main__":
    main()
