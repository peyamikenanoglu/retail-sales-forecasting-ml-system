# Final Project Summary

## Project
Retail Store Sales Forecasting with Time-Aware Machine Learning

## Dataset
Kaggle Store Sales - Time Series Forecasting

## Problem Type
Time-series forecasting / supervised regression

## Target
sales

## Validation Strategy
Time-based validation split:
- Train: before 2017-07-01
- Validation: 2017-07-01 to 2017-08-15

## Champion Model
LightGBM trained on log1p(sales)

## Final Metrics
| Model | MAE | RMSE | RMSLE |
|---|---:|---:|---:|
| Naive Forecast | 122.0717 | 431.9644 | 0.548650 |
| CatBoost Log Target | 63.6831 | 234.9786 | 0.379240 |
| LightGBM Log Target | 59.0045 | 221.5640 | 0.373578 |
| CatBoost + LightGBM Ensemble | 59.0132 | 220.3944 | 0.373673 |

## Final Decision
The final champion is LightGBM because it achieved the best RMSLE, which is the main competition-aligned metric.

## Most Important Features
| Feature | Meaning |
|---|---|
| rolling_mean_7 | Average sales over the previous 7 days for the same store-family series |
| lag_1 | Sales from the previous day |
| lag_7 | Sales from the same weekday in the previous week |
| rolling_mean_14 | Average sales over the previous 14 days |
| lag_14 | Sales from two weeks earlier |

## Key Findings
- Temporal lag and rolling features were the strongest predictors.
- LightGBM outperformed CatBoost on the internal validation split.
- The model substantially improved over the naive forecast baseline.
- Promotion, holiday, and calendar features added useful but secondary signal.
- The validation score is not directly comparable to Kaggle’s official leaderboard because it is based on an internal time-based split.