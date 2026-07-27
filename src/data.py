import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, TARGET, POSITIVE_CLASS, DROP_COLS,
                    RANDOM_STATE, TEST_SIZE)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    return X, y


def get_splits(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X, y = load_data()
    return train_test_split(X, y,
        test_size=test_size,
        stratify=y,            # preserves the 67/33 balance in both splits
        random_state=random_state,
    )


def get_feature_types(X):
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return categorical, numeric