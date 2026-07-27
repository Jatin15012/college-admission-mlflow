"""Train an improved challenger model with engineered features."""
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import EXPERIMENT_NAME, MODEL_NAME, RANDOM_STATE, TRACKING_URI
from data import get_feature_types, get_splits

# Max observed score per exam — from EDA (notebooks/01_eda.ipynb)
EXAM_MAX = {"cet": 199, "jee": 299, "neet": 634, "none": 1}


def add_features(X):
    """Engineered features. Stateless, so it works on one row or 20,000."""
    X = X.copy()
    X["has_entrance_exam"] = (X["entrance_exam"] != "none").astype(int)
    X["entrance_score_norm"] = X["entrance_score"] / X["entrance_exam"].map(EXAM_MAX).fillna(1)
    return X


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    X_train, X_test, y_train, y_test = get_splits()
    categorical, numeric = get_feature_types(X_train)
    numeric_plus = numeric + ["has_entrance_exam", "entrance_score_norm"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric_plus),
    ])
    pipe = Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    with mlflow.start_run(run_name="challenger_engineered_features"):
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

        mlflow.log_params({
            "model": "logistic_regression",
            "features": "has_entrance_exam + entrance_score_norm",
        })
        mlflow.log_metrics({"cv_roc_auc_mean": cv.mean(), "test_roc_auc": auc})

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
            serialization_format="cloudpickle",
        )
        print(f"challenger cv_auc={cv.mean():.4f}  test_auc={auc:.4f}  (champion was 0.6953)")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "challenger", latest.version)
    print(f"alias @challenger -> version {latest.version}")


if __name__ == "__main__":
    main()