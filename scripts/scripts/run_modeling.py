"""Run modeling tasks (CLI)"""

import argparse
from src.models import ccf_analysis, train_rf, clustering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ccf")
    args = parser.parse_args()
    print("Would run modeling task:", args.task)


if __name__ == "__main__":
    main()
