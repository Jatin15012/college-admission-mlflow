"""Train the champion model, register it, and set the @champion alias."""
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import EXPERIMENT_NAME, MODEL_NAME, RANDOM_STATE, TRACKING_URI
from data import get_feature_types, get_splits
from train import build_pipeline


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    X_train, X_test, y_train, y_test = get_splits()
    categorical, numeric = get_feature_types(X_train)

    with mlflow.start_run(run_name="champion_logistic_regression"):
        pipe = build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            categorical, numeric,
        )
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

        mlflow.log_params({"model": "logistic_regression", "C": 1.0, "max_iter": 1000})
        mlflow.log_metric("test_roc_auc", auc)

        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
        )
        print(f"registered -> {info.model_uri}   test_roc_auc={auc:.4f}")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    print(f"alias @champion -> version {latest.version}")


if __name__ == "__main__":
    main()