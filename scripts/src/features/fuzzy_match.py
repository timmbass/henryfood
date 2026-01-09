"""Fuzzy matching stub"""

import argparse
from thefuzz import process


def fuzzy_map(text: str, choices=None):
    if choices is None:
        choices = []
    print(f"Stub: fuzzy_map called for {text}")
    return process.extract(text, choices, limit=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(fuzzy_map(args.text))


if __name__ == "__main__":
    main()
