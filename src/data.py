"""Data loading and splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, TARGET, POSITIVE_CLASS, DROP_COLS,
                    RANDOM_STATE, TEST_SIZE)


def load_data(path=DATA_PATH):
    """Load the CSV and return features and binary target."""
    df = pd.read_csv(path)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    return X, y


def get_splits(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Return a stratified train/test split."""
    X, y = load_data()
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,            # preserves the 67/33 balance in both splits
        random_state=random_state,
    )


def get_feature_types(X):
    """Split column names into categorical and numeric."""
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric