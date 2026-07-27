"""Compare challenger against champion and promote only if it genuinely wins."""
from datetime import date

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

from config import MODEL_NAME, TRACKING_URI
from data import get_splits

MARGIN = 0.005   # required improvement; anything smaller is noise


def score(alias, X, y):
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    _, X_test, _, y_test = get_splits()

    champ_auc = score("champion", X_test, y_test)
    chal_auc = score("challenger", X_test, y_test)

    champ_v = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    chal_v = client.get_model_version_by_alias(MODEL_NAME, "challenger").version

    print(f"\nchampion   v{champ_v}: roc_auc = {champ_auc:.4f}")
    print(f"challenger v{chal_v}: roc_auc = {chal_auc:.4f}")
    print(f"required margin: {MARGIN}")
    print(f"difference: {chal_auc - champ_auc:+.4f}\n")

    if chal_auc > champ_auc + MARGIN:
        client.set_registered_model_alias(MODEL_NAME, "champion", chal_v)
        client.set_model_version_tag(MODEL_NAME, chal_v, "promoted_on", str(date.today()))
        client.set_model_version_tag(MODEL_NAME, chal_v, "promoted_roc_auc", f"{chal_auc:.4f}")
        print(f"PROMOTED — @champion now points to version {chal_v}")
        print("Restart the serving process to pick up the new champion.")
    else:
        print(f"REJECTED — improvement of {chal_auc - champ_auc:+.4f} does not exceed the "
              f"{MARGIN} noise margin. @champion stays at version {champ_v}.")


if __name__ == "__main__":
    main()