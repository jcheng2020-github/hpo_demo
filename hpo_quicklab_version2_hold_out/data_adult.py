# data_adult.py
from __future__ import annotations

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_adult(test_size: float = 0.25, seed: int = 42, data_home: str | None = None):
    """
    Loads the UCI Adult dataset via OpenML, preprocesses with one-hot encoding,
    returns a train/test split.

    IMPORTANT: HistGradientBoostingClassifier requires DENSE input.
    So we force OneHotEncoder to output dense arrays.
    """
    data = fetch_openml("adult", version=2, as_frame=True, data_home=data_home)

    X_df = data.data
    y_raw = data.target

    # Binary label: 1 if >50K, else 0
    y = (y_raw == ">50K").astype(int).values

    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    num_cols = X_df.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            # dense output to satisfy HistGradientBoostingClassifier
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
