from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_adult_full(data_home: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Adult dataset via OpenML and apply the SAME preprocessing pipeline as data_adult.py:
      - OneHotEncoder(handle_unknown="ignore", sparse_output=False)  -> dense output required by HGBDT
      - passthrough numeric columns
    Returns:
      X (dense numpy array), y (0/1 numpy array)
    """
    data = fetch_openml("adult", version=2, as_frame=True, data_home=data_home)

    X_df = data.data
    y_raw = data.target
    y = (y_raw == ">50K").astype(int).values

    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    num_cols = X_df.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(X_df)
    return np.asarray(X), np.asarray(y)


def stratified_subsample(X: np.ndarray, y: np.ndarray, n_total: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified subsample to exactly n_total points (or full if n_total >= len(y)).
    For n_total=50 this is what you want.
    """
    n_total = int(n_total)
    if n_total >= len(y):
        return X, y

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    p1 = float(np.mean(y))
    n1 = int(round(n_total * p1))
    n0 = n_total - n1

    n0 = min(n0, len(idx0))
    n1 = min(n1, len(idx1))

    # fill remainder if one class clipped
    rem = n_total - (n0 + n1)
    if rem > 0:
        if len(idx0) - n0 >= rem:
            n0 += rem
        else:
            n1 += rem

    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y[sel]


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
