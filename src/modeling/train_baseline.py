import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/GitHub/retail-sales-forecasting-ml-system")
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.build_features import *


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
    "is_holiday",
    "is_weekend",
    "month",
    "day_of_week",
    "year",
    "cluster",
    "dcoilwtico_filled",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_30"
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

naive_mae = mean_absolute_error(y_valid, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_valid, naive_pred))


# Log target model
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train_log,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid_log),
    use_best_model=True
)


# Prediction and inverse transform
y_pred_log = model.predict(X_valid)
y_pred = np.expm1(y_pred_log)

model_mae = mean_absolute_error(y_valid, y_pred)
model_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))


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
        "CatBoost Log Target"
    ],
    "mae": [
        naive_mae,
        model_mae
    ],
    "rmse": [
        naive_rmse,
        model_rmse
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