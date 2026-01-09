"""SIGHI parser stub"""

import argparse


def parse_sighi(pdf_path: str):
    print(f"Stub: parse_sighi called for {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/external/sighi.pdf")
    args = parser.parse_args()
    parse_sighi(args.pdf)


if __name__ == "__main__":
    main()
