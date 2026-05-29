# Retail Store Sales Forecasting with Time-Aware Machine Learning

A production-style machine learning project for forecasting retail store sales using historical sales, store metadata, promotions, holidays, oil prices, lag features, rolling statistics, and time-aware validation.

This project was built as a serious GitHub portfolio project, not as a quick notebook experiment. The main goal is to demonstrate a complete machine learning workflow for a realistic retail forecasting problem: data inspection, exploratory analysis, leakage-safe feature engineering, time-based validation, model comparison, inference pipeline, and clean project organization.

---

## 1. Project Overview

Retail demand forecasting is a core business problem for supermarkets, supply chain teams, inventory planners, and commercial analytics departments. Accurate sales forecasting helps businesses reduce stockouts, avoid overstocking, plan promotions, and allocate inventory across stores.

This project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle. The task is to predict future sales for different product families across multiple stores.

The project focuses on:

- Time-series forecasting with tabular machine learning
- Leakage-safe validation
- Store-family level lag and rolling features
- Promotion, holiday, oil price, and calendar-based feature engineering
- Strong baseline comparison
- CatBoost and LightGBM modeling
- Champion model selection using RMSLE
- Production-oriented prediction pipeline

---

## 2. Business Problem

The business question is:

> Can we predict future sales for each store and product family using historical sales patterns, promotions, calendar effects, store metadata, holidays, and macroeconomic signals?

The final prediction unit is:

```text
date + store_nbr + family
```

For each future date, store, and product family, the model predicts:

```text
sales
```

---

## 3. Dataset

The project uses the Kaggle **Store Sales - Time Series Forecasting** dataset.

Main files used:

| File | Description |
|---|---|
| `train.csv` | Historical sales data |
| `test.csv` | Future rows for prediction |
| `stores.csv` | Store metadata such as city, state, type, and cluster |
| `holidays_events.csv` | National, regional, and local holidays/events |
| `oil.csv` | Daily oil prices |
| `transactions.csv` | Store-level transaction counts, reserved for future improvement |
| `sample_submission.csv` | Kaggle submission format |

Raw Kaggle data is not committed to GitHub. Users must download it separately and place it under:

```text
data/raw/
```

---

## 4. Problem Type

This is a supervised machine learning problem with a time-series structure.

| Item | Value |
|---|---|
| Problem type | Time-series forecasting / supervised regression |
| Target variable | `sales` |
| Prediction level | `date + store_nbr + family` |
| Main metric | RMSLE |
| Secondary metrics | MAE, RMSE |
| Main validation method | Time-based validation split |

---

## 5. Validation Strategy

A random split is not appropriate for this project because it would mix future observations into the training process and create temporal leakage.

Instead, the project uses a time-based split:

| Dataset | Period |
|---|---|
| Training | Before `2017-07-01` |
| Validation | From `2017-07-01` to `2017-08-15` |

This simulates the real forecasting setting:

```text
Past data → Future prediction
```

---

## 6. Feature Engineering

The project includes several groups of features.

### 6.1 Calendar Features

Created from the `date` column:

| Feature | Meaning |
|---|---|
| `year` | Year |
| `month` | Month |
| `day` | Day of month |
| `day_of_week` | Weekday number |
| `week_of_year` | ISO week number |
| `is_weekend` | Whether the date is Saturday or Sunday |

### 6.2 Store Features

Merged from `stores.csv`:

| Feature | Meaning |
|---|---|
| `city` | Store city |
| `state` | Store state |
| `type` | Store type |
| `cluster` | Store cluster |

### 6.3 Holiday Features

A binary holiday feature was created using `holidays_events.csv`.

| Feature | Meaning |
|---|---|
| `is_holiday` | Whether the date is associated with a holiday or event |

Transferred holidays were handled carefully to avoid noisy duplicate effects.

### 6.4 Oil Price Feature

Oil prices were merged from `oil.csv`.

Missing oil prices were filled using:

```text
forward fill + backward fill
```

This is appropriate for a time-series economic variable because oil prices are not available for every calendar day.

| Feature | Meaning |
|---|---|
| `dcoilwtico_filled` | Cleaned oil price feature |

### 6.5 Lag Features

Lag features were created at the store-family level:

```text
groupby(["store_nbr", "family"])
```

| Feature | Meaning |
|---|---|
| `lag_1` | Sales from the previous day |
| `lag_7` | Sales from the same weekday one week earlier |
| `lag_14` | Sales from two weeks earlier |
| `lag_21` | Sales from three weeks earlier |
| `lag_28` | Sales from four weeks earlier |
| `lag_56` | Sales from eight weeks earlier |

These features were created using only past values.

### 6.6 Rolling Features

Rolling statistics were also created at the store-family level.

| Feature | Meaning |
|---|---|
| `rolling_mean_7` | Average sales over the previous 7 days |
| `rolling_mean_14` | Average sales over the previous 14 days |
| `rolling_mean_30` | Average sales over the previous 30 days |
| `rolling_mean_60` | Average sales over the previous 60 days |
| `rolling_mean_90` | Average sales over the previous 90 days |
| `rolling_std_7` | Sales volatility over the previous 7 days |
| `rolling_std_14` | Sales volatility over the previous 14 days |
| `rolling_std_30` | Sales volatility over the previous 30 days |
| `rolling_std_60` | Sales volatility over the previous 60 days |

To avoid target leakage, rolling features were calculated after applying `shift(1)`, so the current target value was not included in its own features.

### 6.7 Promotion Features

Promotion-related features were created from `onpromotion`.

| Feature | Meaning |
|---|---|
| `has_promotion` | Whether there is at least one promoted item |
| `promotion_lag_1` | Promotion count from the previous day |
| `promotion_lag_7` | Promotion count from one week earlier |
| `promotion_lag_14` | Promotion count from two weeks earlier |
| `promotion_rolling_mean_7` | Average promotion count over the previous 7 days |
| `promotion_rolling_mean_14` | Average promotion count over the previous 14 days |
| `promotion_rolling_mean_30` | Average promotion count over the previous 30 days |

---

## 7. Exploratory Data Analysis

Several EDA outputs were generated and saved under:

```text
outputs/figures/
```

Key EDA findings:

- Sales show strong time dependency.
- Weekend sales are higher than weekday sales.
- Promotion has a strong relationship with sales.
- High-volume families such as `GROCERY I`, `BEVERAGES`, `PRODUCE`, and `CLEANING` dominate total sales.
- `PRODUCE` showed suspicious zero-sales behavior in early years, suggesting structural changes or store-family availability effects.
- Oil price showed a strong time trend, but its direct predictive contribution was limited compared to lag and rolling sales features.

Important visual outputs include:

| Figure | Purpose |
|---|---|
| `daily_sales_trend.png` | Overall sales trend |
| `monthly_sales_trend.png` | Monthly seasonality |
| `sales_by_day_of_week.png` | Weekday/weekend behavior |
| `top_product_families.png` | Highest-sales product families |
| `promotion_vs_no_promotion.png` | Promotion effect |
| `top_family_sales_trends.png` | Sales trends of top product families |
| `produce_zero_sales_ratio_over_time.png` | Zero-sales behavior for `PRODUCE` |
| `catboost_feature_importance.png` | CatBoost feature importance |
| `relative_error_by_family.png` | Relative error by product family |

---

## 8. Models

The project compares the following models:

| Model | Description |
|---|---|
| Naive Forecast | Predicts sales using `lag_1` |
| CatBoost Log Target | CatBoost trained on `log1p(sales)` |
| LightGBM Log Target | LightGBM trained on `log1p(sales)` |
| CatBoost + LightGBM Ensemble | Simple average of CatBoost and LightGBM predictions |

The target was transformed using:

```python
np.log1p(sales)
```

Predictions were converted back using:

```python
np.expm1(prediction)
```

This improved stability because sales are highly skewed and contain large outliers.

---

## 9. Final Results

Internal time-based validation results:

| Model | MAE | RMSE | RMSLE |
|---|---:|---:|---:|
| Naive Forecast | 122.0717 | 431.9644 | 0.548650 |
| CatBoost Log Target | 63.6831 | 234.9786 | 0.379240 |
| LightGBM Log Target | **59.0045** | 221.5640 | **0.373578** |
| CatBoost + LightGBM Ensemble | 59.0132 | **220.3944** | 0.373673 |

---

## 10. Champion Model

The final champion model is:

```text
LightGBM Log Target
```

Champion validation metrics:

| Metric | Value |
|---|---:|
| MAE | 59.0045 |
| RMSE | 221.5640 |
| RMSLE | 0.373578 |

LightGBM was selected because it achieved the best RMSLE, which is the main competition-aligned metric.

The ensemble achieved slightly better RMSE, but RMSLE was the primary decision metric.

---

## 11. Important Feature Insights

The strongest features were temporal features based on historical sales.

Top features included:

| Feature | Meaning |
|---|---|
| `rolling_mean_7` | Average sales over the previous 7 days for the same store-family series |
| `lag_1` | Sales from the previous day |
| `lag_7` | Sales from the same weekday in the previous week |
| `rolling_mean_14` | Average sales over the previous 14 days |
| `lag_14` | Sales from two weeks earlier |

Main interpretation:

> The model relied primarily on recent historical demand patterns. This is expected in retail forecasting, where store-family level sales history is usually the strongest predictor of near-future demand.

Promotion, holiday, calendar, and oil features provided additional signal, but lag and rolling features dominated model performance.

---

## 12. Error Analysis

Error analysis was performed by product family.

Key findings:

- Absolute error was highest for high-volume categories such as `GROCERY I`, `BEVERAGES`, `CLEANING`, and `PRODUCE`.
- Relative error showed that the model performed well on high-revenue categories.
- Intermittent-demand categories such as `LINGERIE`, `CELEBRATION`, and `GROCERY II` had higher relative error.
- `PRODUCE` initially appeared suspicious because of historical zero-sales periods, but validation relative error was reasonable.

This distinction matters because absolute error naturally increases with sales volume.

---

## 13. Inference Pipeline

The project includes an inference script:

```text
src/inference/predict.py
```

This script:

1. Loads the trained LightGBM model.
2. Loads train/test and supporting datasets.
3. Applies the same feature engineering pipeline.
4. Generates predictions for `test.csv`.
5. Saves the prediction file to:

```text
outputs/predictions/submission_lightgbm.csv
```

Run inference:

```bash
python src/inference/predict.py
```

Expected output:

```text
Prediction file saved to: outputs/predictions/submission_lightgbm.csv
```

---

## 14. Project Structure

```text
retail-sales-forecasting-ml-system/
│
├── README.md
├── requirements.txt
├── .gitignore
├── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── catboost_sales_forecaster.cbm
│   └── lightgbm_sales_forecaster.pkl
│
├── notebooks/
│   └── dataset_notes.md
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   ├── catboost_feature_importance.csv
│   │   └── champion_metadata.json
│   └── predictions/
│       └── submission_lightgbm.csv
│
├── reports/
│   └── final_project_summary.md
│
└── src/
    ├── data_loading_and_inspection.py
    ├── features/
    │   └── build_features.py
    ├── inference/
    │   └── predict.py
    ├── modeling/
    │   ├── train_baseline.py
    │   └── train_improved.py
    └── utils/
        └── helpers.py
```

Note: raw data, trained models, CatBoost logs, and generated prediction files are excluded from GitHub through `.gitignore`.

---

## 15. How to Run the Project

### 15.1 Clone the Repository

```bash
git clone <repository-url>
cd retail-sales-forecasting-ml-system
```

### 15.2 Create and Activate Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 15.3 Install Requirements

```bash
pip install -r requirements.txt
```

### 15.4 Add Kaggle Data

Download the Kaggle dataset and place the following files inside:

```text
data/raw/
```

Required files:

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

### 15.5 Train the Improved Model

```bash
python src/modeling/train_improved.py
```

### 15.6 Generate Predictions

```bash
python src/inference/predict.py
```

---

## 16. Reproducibility Notes

- The project uses a fixed random seed where applicable.
- The validation split is time-based.
- Lag and rolling features are created with past values only.
- Rolling features use `shift(1)` to avoid target leakage.
- Raw Kaggle data is not committed to GitHub.
- Model artifacts are excluded from GitHub by default.

---

## 17. Limitations

This project uses an internal time-based validation split. Its RMSLE score is useful for model comparison inside this project but is not directly equivalent to Kaggle’s official leaderboard score, which is computed on a hidden test set.

Other limitations:

- No advanced holiday distance features were implemented.
- `transactions.csv` was not used in the final model.
- Deep learning models were not tested.
- Hyperparameter optimization was limited to practical, time-efficient model improvement.
- The current Streamlit and FastAPI components are reserved for later productization.

---

## 18. Future Improvements

Possible next steps:

- Add transaction-based features.
- Add local/regional/national holiday-specific features.
- Add days-before-holiday and days-after-holiday features.
- Add model explainability with SHAP.
- Build a Streamlit dashboard for interactive forecasting.
- Build a FastAPI service for prediction.
- Add Docker support.
- Add CI checks for code quality.
- Test horizon-specific models.
- Compare direct multi-step forecasting strategies.
- Add walk-forward validation across multiple time windows.

---

## 19. Portfolio Value

This project demonstrates skills relevant to data science, machine learning engineering, and applied forecasting roles:

- Time-series aware validation
- Leakage-safe feature engineering
- Retail demand forecasting
- Tabular ML with CatBoost and LightGBM
- Model comparison using business-relevant metrics
- Error analysis by product family
- Inference pipeline design
- GitHub-ready project organization
- Professional documentation

---

## 20. Final Takeaway

The final LightGBM model substantially outperformed the naive forecast baseline and achieved a strong internal validation RMSLE of `0.373578`.

The strongest predictive signals came from recent historical sales behavior, especially rolling means and lag features at the store-family level. This confirms that short-term temporal demand patterns are the dominant drivers of forecast accuracy in this retail sales forecasting task.