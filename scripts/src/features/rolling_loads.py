"""Rolling loads stub"""

import argparse


def compute_rolling(df_path: str, out_path: str):
    print(f"Stub: compute_rolling called for {df_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/features_lags.csv")
    parser.add_argument("--out", default="data/processed/features_rolling.csv")
    args = parser.parse_args()
    compute_rolling(args.input, args.out)


if __name__ == "__main__":
    main()
