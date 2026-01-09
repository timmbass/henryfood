"""I/O helpers
"""

import pandas as pd
import pickle


def read_csv(path: str):
    return pd.read_csv(path)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
