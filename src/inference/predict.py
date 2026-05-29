import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("D:/GitHub/retail-sales-forecasting-ml-system")
sys.path.append(str(PROJECT_ROOT))

from src.features.build_features import *

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_sales_forecaster.pkl"

model = joblib.load(MODEL_PATH)
print("Model Loaded Successfully")

train = pd.read_csv(RAW_DATA_DIR / "train.csv")
test = pd.read_csv(RAW_DATA_DIR / "test.csv")
stores = pd.read_csv(RAW_DATA_DIR / "stores.csv")
holidays = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv")
oil = pd.read_csv(RAW_DATA_DIR / "oil.csv")

full_df = pd.concat(
    [train, test],
    axis=0,
    ignore_index=True
)

full_df = create_date_features(full_df)
full_df = merge_store_features(full_df, stores)
full_df = create_holiday_feature(full_df, holidays)
full_df = create_oil_feature(full_df, oil)
full_df = create_lag_features(full_df)
full_df = create_rolling_features(full_df)
full_df = create_promotion_features(full_df)

test_features_df = full_df[full_df["id"].isin(test["id"])].copy()

cat_features = ["family", "city", "state", "type"]

features = [
    "family", "city", "state", "type",
    "onpromotion", "has_promotion", "is_holiday", "is_weekend",
    "month", "day_of_week", "week_of_year", "year", "cluster",
    "dcoilwtico_filled",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28", "lag_56",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_mean_60", "rolling_mean_90",
    "rolling_std_7", "rolling_std_14", "rolling_std_30", "rolling_std_60",
    "promotion_lag_1", "promotion_lag_7", "promotion_lag_14",
    "promotion_rolling_mean_7", "promotion_rolling_mean_14",
    "promotion_rolling_mean_30"
]

X_test = test_features_df[features].fillna(0)

for col in cat_features:
    X_test[col] = X_test[col].astype("category")

pred_log = model.predict(X_test)
pred_sales = np.expm1(pred_log)
pred_sales = np.maximum(pred_sales, 0)

submission = pd.DataFrame({
    "id": test_features_df["id"].astype(int).values,
    "sales": pred_sales
})

submission = submission.sort_values("id").reset_index(drop=True)

submission_path = OUTPUT_DIR / "submission_lightgbm.csv"
submission.to_csv(submission_path, index=False)

print("Prediction file saved to:", submission_path)
print(submission.head())
print(submission.shape)