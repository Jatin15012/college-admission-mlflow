"""Train and compare models, logging everything to MLflow."""
import matplotlib
matplotlib.use("Agg")            # no GUI needed when running as a script
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import EXPERIMENT_NAME, RANDOM_STATE, TRACKING_URI
from data import get_feature_types, get_splits

MODELS = {
    "dummy_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}


def build_pipeline(model, categorical, numeric):
    """Preprocessing + model as one object, so fitting stays leakage-free."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = get_splits()
    categorical, numeric = get_feature_types(X_train)
    print(f"categorical: {categorical}\nnumeric: {numeric}\n")

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(model, categorical, numeric)

            # cross-validate on TRAIN only — the test set stays untouched
            cv = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="roc_auc", n_jobs=-1)

            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            mlflow.log_params({"model": name, **model.get_params()})
            mlflow.log_metrics({
                "cv_roc_auc_mean": cv.mean(),
                "cv_roc_auc_std": cv.std(),
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision": precision_score(y_test, pred, zero_division=0),
                "test_recall": recall_score(y_test, pred, zero_division=0),
                "test_f1": f1_score(y_test, pred, zero_division=0),
            })

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
            ax.set_title(name)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            mlflow.sklearn.log_model(pipe, name="model", input_example=X_test.head(3))

            print(f"{name:22s} cv_auc={cv.mean():.4f}  test_auc={roc_auc_score(y_test, proba):.4f}  acc={accuracy_score(y_test, pred):.4f}")


if __name__ == "__main__":
    main()