"""Register the random forest as champion, representing an earlier deployment.

This gives a realistic starting point for the promotion demo: an initial
production model that a later challenger legitimately beats.
"""
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
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

    with mlflow.start_run(run_name="initial_deployment_random_forest"):
        pipe = build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            categorical, numeric,
        )
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

        mlflow.log_params({"model": "random_forest", "role": "initial_deployment"})
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
            serialization_format="cloudpickle",
        )
        print(f"initial deployment model  test_auc={auc:.4f}")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    print(f"alias @champion -> version {latest.version} (random forest)")


if __name__ == "__main__":
    main()