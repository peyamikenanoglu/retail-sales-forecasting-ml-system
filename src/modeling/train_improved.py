import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/GitHub/retail-sales-forecasting-ml-system")
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from src.features.build_features import *
from lightgbm import LGBMRegressor

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Load data
df = pd.read_csv(RAW_DATA_DIR / "train.csv")
stores = pd.read_csv(RAW_DATA_DIR / "stores.csv")
holidays = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv")
oil = pd.read_csv(RAW_DATA_DIR / "oil.csv")


# Feature engineering
df = create_date_features(df)
df = merge_store_features(df, stores)
df = create_holiday_feature(df, holidays)
df = create_oil_feature(df, oil)
df = create_lag_features(df)
df = create_rolling_features(df)
df = create_promotion_features(df)


# Time-based validation split
split_date = "2017-07-01"

train_df = df[df["date"] < split_date].copy()
valid_df = df[df["date"] >= split_date].copy()


# Feature selection
target = "sales"

cat_features = [
    "family",
    "city",
    "state",
    "type"
]

num_features = [
    "onpromotion",
    "has_promotion",
    "is_holiday",
    "is_weekend",
    "month",
    "day_of_week",
    "week_of_year",
    "year",
    "cluster",
    "dcoilwtico_filled",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_56",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_mean_60",
    "rolling_mean_90",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_30",
    "rolling_std_60",

    "promotion_lag_1",
    "promotion_lag_7",
    "promotion_lag_14",

    "promotion_rolling_mean_7",
    "promotion_rolling_mean_14",
    "promotion_rolling_mean_30"
]

features = cat_features + num_features

X_train = train_df[features].fillna(0)
y_train = train_df[target]

X_valid = valid_df[features].fillna(0)
y_valid = valid_df[target]


# Categorical feature indices for CatBoost
cat_feature_indices = [
    X_train.columns.get_loc(col)
    for col in cat_features
]


# Naive baseline
naive_pred = X_valid["lag_1"]

naive_pred = np.maximum(naive_pred, 0)

naive_rmsle = np.sqrt(
    mean_squared_log_error(
        y_valid,
        naive_pred
    )
)
naive_mae = mean_absolute_error(y_valid, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_valid, naive_pred))


# Log target model
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.025,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train_log,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid_log),
    use_best_model=True,
    early_stopping_rounds=150
)


# Prediction and inverse transform
y_pred_log = model.predict(X_valid)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

model_mae = mean_absolute_error(y_valid, y_pred)
model_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
model_rmsle = np.sqrt(
    mean_squared_log_error(
        y_valid,
        y_pred
    )
)

################################################
# LightGBM Model
################################################

X_train_lgbm = X_train.copy()
X_valid_lgbm = X_valid.copy()

for col in cat_features:
    X_train_lgbm[col] = X_train_lgbm[col].astype("category")
    X_valid_lgbm[col] = X_valid_lgbm[col].astype("category")

lightgbm_model = LGBMRegressor(
    objective="regression",
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

lightgbm_model.fit(
    X_train_lgbm,
    y_train_log,
    eval_set=[(X_valid_lgbm, y_valid_log)],
    eval_metric="rmse",
    categorical_feature=cat_features,
    callbacks=[]
)

y_pred_lgbm_log = lightgbm_model.predict(X_valid_lgbm)
y_pred_lgbm = np.expm1(y_pred_lgbm_log)
y_pred_lgbm = np.maximum(y_pred_lgbm, 0)

lgbm_mae = mean_absolute_error(y_valid, y_pred_lgbm)
lgbm_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_lgbm))
lgbm_rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred_lgbm))

lightgbm_model_path = MODEL_DIR / "lightgbm_sales_forecaster.pkl"

joblib.dump(
    lightgbm_model,
    lightgbm_model_path
)

print("\n##################### LightGBM Model Saved #####################")
print(lightgbm_model_path)

################################################
# Ensemble: CatBoost + LightGBM
################################################

ensemble_pred = (
    0.5 * y_pred
    + 0.5 * y_pred_lgbm
)

ensemble_pred = np.maximum(ensemble_pred, 0)

ensemble_mae = mean_absolute_error(y_valid, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_valid, ensemble_pred))
ensemble_rmsle = np.sqrt(mean_squared_log_error(y_valid, ensemble_pred))



# Save model
model_path = MODEL_DIR / "catboost_sales_forecaster.cbm"
model.save_model(str(model_path))


# Feature importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.to_csv(
    METRICS_DIR / "catboost_feature_importance.csv",
    index=False
)


# Save model comparison
model_comparison = pd.DataFrame({
    "model": [
        "Naive Forecast",
        "CatBoost Log Target",
        "LightGBM Log Target",
        "CatBoost + LightGBM Ensemble"
    ],
    "mae": [
        naive_mae,
        model_mae,
        lgbm_mae,
        ensemble_mae
    ],
    "rmse": [
        naive_rmse,
        model_rmse,
        lgbm_rmse,
        ensemble_rmse
    ],
    "rmsle": [
        naive_rmsle,
        model_rmsle,
        lgbm_rmsle,
        ensemble_rmsle
    ]
})

model_comparison.to_csv(
    METRICS_DIR / "model_comparison.csv",
    index=False
)


print("\n##################### Train Baseline Completed #####################")
print("Model saved to:", model_path)

print("\n##################### Model Comparison #####################")
print(model_comparison)

print("\n##################### Feature Importance #####################")
print(feature_importance)

print("\n##################### LightGBM Metrics #####################")
print(f"LightGBM MAE   : {lgbm_mae:.4f}")
print(f"LightGBM RMSE  : {lgbm_rmse:.4f}")
print(f"LightGBM RMSLE : {lgbm_rmsle:.6f}")

print("\n##################### Ensemble Metrics #####################")
print(f"Ensemble MAE   : {ensemble_mae:.4f}")
print(f"Ensemble RMSE  : {ensemble_rmse:.4f}")
print(f"Ensemble RMSLE : {ensemble_rmsle:.6f}")

champion_metadata = {
    "champion_model": "LightGBM Log Target",
    "validation_strategy": "Time-based split: train before 2017-07-01, validation from 2017-07-01 to 2017-08-15",
    "target": "sales",
    "target_transform": "log1p",
    "inverse_transform": "expm1",
    "main_metric": "RMSLE",
    "mae": lgbm_mae,
    "rmse": lgbm_rmse,
    "rmsle": lgbm_rmsle,
    "catboost_rmsle": model_rmsle,
    "ensemble_rmsle": ensemble_rmsle,
    "naive_rmsle": naive_rmsle,
    "features": features
}

import json

metadata_path = METRICS_DIR / "champion_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(champion_metadata, f, indent=4)

print("\n##################### Champion Metadata Saved #####################")
print(metadata_path)