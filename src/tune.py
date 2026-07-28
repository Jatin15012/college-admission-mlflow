"""Hyperparameter tuning for the champion model."""
import numpy as np
import mlflow
import mlflow.sklearn
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from config import EXPERIMENT_NAME, RANDOM_STATE, TRACKING_URI
from data import get_feature_types, get_splits
from train import build_pipeline

# Baseline to beat: untuned logistic regression scored cv_auc 0.6786, test_auc 0.6953
BASELINE_CV_AUC = 0.6786


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = get_splits()
    categorical, numeric = get_feature_types(X_train)

    pipe = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        categorical, numeric,
    )

    # C controls regularisation: small C = simpler model, large C = more flexible
    param_dist = {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="tuned_logistic_regression"):
        search.fit(X_train, y_train)

        best_cv = search.best_score_
        test_auc = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])

        mlflow.log_params({
            "model": "logistic_regression_tuned",
            "search": "RandomizedSearchCV",
            "n_iter": 15,
            "best_C": round(search.best_params_["model__C"], 5),
            "best_class_weight": str(search.best_params_["model__class_weight"]),
        })
        mlflow.log_metrics({
            "cv_roc_auc_mean": best_cv,
            "test_roc_auc": test_auc,
            "cv_gain_over_default": best_cv - BASELINE_CV_AUC,
        })

        print("\n" + "=" * 55)
        print(f"best params      : {search.best_params_}")
        print(f"tuned   cv_auc   : {best_cv:.4f}")
        print(f"default cv_auc   : {BASELINE_CV_AUC:.4f}")
        print(f"gain             : {best_cv - BASELINE_CV_AUC:+.4f}")
        print(f"tuned   test_auc : {test_auc:.4f}   (default was 0.6953)")
        print("=" * 55)


if __name__ == "__main__":
    main()