"""Liljebo FODMAP mapping stub"""

import argparse


def load_liljebo(path: str):
    print(f"Stub: load_liljebo called for {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/external/liljebo.csv")
    args = parser.parse_args()
    load_liljebo(args.path)


if __name__ == "__main__":
    main()
