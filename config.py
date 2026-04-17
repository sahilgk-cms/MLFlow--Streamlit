import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI")

FEATURES_DATA_COLS = [
    "window",
    "shift_by",
    "lags_weather",
    "lags_cases",
    "interaction_lag",
    "humidity_threshold",
    "precip_threshold",
    "temp_threshold",
    "diurnal_threshold",
    "rolling_windows",
    "cutoff_week"
]

ML_PARAMS_COLS = [
    "n_trials",
    "n_splits",
    "iterations",
    "learning_rate",
    "l2_leaf_reg",
    "subsample"
]