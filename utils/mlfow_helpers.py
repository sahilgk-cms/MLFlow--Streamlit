import mlflow
from mlflow.tracking import MlflowClient
from typing import List
import pandas as pd

def initiate_client(mlflow_uri: str) -> mlflow.tracking.client.MlflowClient:
    client = MlflowClient(tracking_uri=mlflow_uri)
    return client

def get_all_models(client: mlflow.tracking.client.MlflowClient) -> List[str]:
    models = client.search_registered_models()
    return sorted([m.name for m in models])

def get_training_context(client: mlflow.tracking.client.MlflowClient, 
                         registered_model_name: str, version: int) -> dict:
    mv = client.get_model_version(registered_model_name, version)
    training_run = client.get_run(mv.run_id)

    pipeline_root_run_id = mv.tags.get("pipeline_root_run_id")
    pipeline_run = client.get_run(pipeline_root_run_id)

    test_metrics = {
        k:v 
        for k, v in mv.tags.items()
        if k.startswith("test_") and k != "test_data_hash"
    }

    context = {
        "training_run_id": mv.run_id,
        "pipeline_root_run_id": pipeline_root_run_id,
        "experiment_id": training_run.info.experiment_id,
        "model_tags": mv.tags,
        "params": training_run.data.params,
        "metrics": training_run.data.metrics,
        "test_metrics": test_metrics,
        "training_tags": training_run.data.tags,
        "pipeline_tags": pipeline_run.data.tags

    }
    return context


def get_model_versions(client: mlflow.tracking.client.MlflowClient,
                       registered_model_name: str) -> List[str]:
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    versions = sorted([v.version for v in versions])
    return versions


def get_multiple_versions_context(client: mlflow.tracking.client.MlflowClient,
                                    registered_model_name: str,
                                    versions: list) -> pd.DataFrame:
    data = []
    for v in versions:
        ctx = get_training_context(client, registered_model_name, v)
        data.append({
            "version": v,
            "test_rmse": float(ctx["test_metrics"].get("test_rmse", 0)),
            "cv_rmse": float(ctx["metrics"].get("best_cv_rmse", 0)),
            "precision": float(ctx["model_tags"].get("precision", 0)),
            "recall": float(ctx["model_tags"].get("recall", 0)),
            "high_risk_limit": float(ctx["model_tags"].get("high_risk_limit")),
            "train_date_min": ctx["pipeline_tags"]["train_date_min"],
            "train_date_max": ctx["pipeline_tags"]["train_date_max"],
            "test_date_min": ctx["pipeline_tags"]["test_date_min"],
            "test_date_max": ctx["pipeline_tags"]["test_date_max"],
            "window": ctx["model_tags"].get("window"),
            "shift_by": ctx["model_tags"].get("shift_by"),
            "lags_weather": ctx["model_tags"].get("lags_weather"),
            "lags_cases": ctx["model_tags"].get("lags_cases"),
            "interaction_lag": ctx["model_tags"].get("interaction_lag"),
            "humidity_threshold": ctx["model_tags"].get("humidity_threshold"),
            "precip_threshold": ctx["model_tags"].get("precip_threshold"),
            "temp_threshold": ctx["model_tags"].get("temp_threshold"),
            "diurnal_threshold": ctx["model_tags"].get("diurnal_threshold"),
            "rolling_windows": ctx["model_tags"].get("rolling_windows"),
            "cutoff_week": ctx["model_tags"].get("cutoff_week"),
            "n_trials": ctx["params"].get("n_trials"),
            "n_splits": ctx["params"].get("n_splits"),
            "iterations": ctx["params"].get("iterations"),
            "learning_rate": ctx["params"].get("learning_rate"),
            "l2_leaf_reg": ctx["params"].get("l2_leaf_reg"),
            "subsample": ctx["params"].get("subsample"),
        })
    return pd.DataFrame(data)


def load_model(mlflow_uri: str, registered_model_name: str, version: int):
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri = f"models:/{registered_model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model
