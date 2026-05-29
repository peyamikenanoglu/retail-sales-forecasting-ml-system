import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"


st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


def load_model_comparison():
    path = METRICS_DIR / "model_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


def load_feature_importance():
    path = METRICS_DIR / "catboost_feature_importance.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


def load_champion_metadata():
    path = METRICS_DIR / "champion_metadata.json"

    if path.exists():
        with open(path, "r") as file:
            return json.load(file)

    return None


def show_metric_card(label, value):
    st.metric(label=label, value=value)


def show_figure(filename, caption):
    path = FIGURES_DIR / filename

    if path.exists():
        try:
            st.image(
                str(path),
                caption=caption,
                use_container_width=True
            )
        except TypeError:
            st.image(
                str(path),
                caption=caption,
                use_column_width=True
            )
    else:
        st.warning(f"Figure not found: {filename}")

st.title("Retail Store Sales Forecasting")
st.write(
    "A production-style machine learning project for forecasting retail store sales "
    "using historical sales, store metadata, promotion signals, holidays, oil prices, "
    "lag features, rolling statistics, and time-aware validation."
)


metadata = load_champion_metadata()
model_comparison = load_model_comparison()
feature_importance = load_feature_importance()


st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Model Results",
        "Feature Importance",
        "EDA Figures",
        "Prediction Output",
        "Project Notes"
    ]
)


if section == "Project Overview":

    st.header("Project Overview")

    st.subheader("Business Problem")
    st.write(
        "The goal is to predict future sales for each store and product family. "
        "This type of forecasting is useful for inventory planning, promotion planning, "
        "store operations, and demand management."
    )

    st.subheader("Prediction Unit")
    st.code("date + store_nbr + family")

    st.subheader("Problem Type")
    st.write("Time-series forecasting / supervised regression")

    st.subheader("Validation Strategy")
    st.write(
        "A time-based validation split was used to avoid temporal leakage. "
        "The model was trained on historical data before 2017-07-01 and validated "
        "on future data from 2017-07-01 to 2017-08-15."
    )

    st.subheader("Main Metric")
    st.write("RMSLE was used as the main metric because it is aligned with the Kaggle competition metric.")

    if metadata:
        st.subheader("Champion Model")
        col1, col2, col3 = st.columns(3)

        with col1:
            show_metric_card("MAE", round(metadata["mae"], 4))

        with col2:
            show_metric_card("RMSE", round(metadata["rmse"], 4))

        with col3:
            show_metric_card("RMSLE", round(metadata["rmsle"], 6))


elif section == "Model Results":

    st.header("Model Results")

    if model_comparison is not None:
        st.dataframe(model_comparison, use_container_width=True)

        best_row = model_comparison.sort_values("rmsle").iloc[0]

        st.subheader("Best Model by RMSLE")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            show_metric_card("Model", best_row["model"])

        with col2:
            show_metric_card("MAE", round(best_row["mae"], 4))

        with col3:
            show_metric_card("RMSE", round(best_row["rmse"], 4))

        with col4:
            show_metric_card("RMSLE", round(best_row["rmsle"], 6))

        st.write(
            "LightGBM was selected as the champion model because it achieved the best RMSLE "
            "on the internal time-based validation split."
        )

    else:
        st.warning("model_comparison.csv not found.")


elif section == "Feature Importance":

    st.header("Feature Importance")

    if feature_importance is not None:
        st.dataframe(feature_importance, use_container_width=True)

        st.subheader("Top 15 Features")

        top_features = feature_importance.head(15).copy()
        st.bar_chart(
            top_features.set_index("Feature")["Importance"]
        )

        st.write(
            "The strongest features are lag and rolling sales features, which confirms that "
            "recent historical demand patterns are the dominant signal in this retail forecasting task."
        )

    else:
        st.warning("catboost_feature_importance.csv not found.")


elif section == "EDA Figures":

    st.header("Exploratory Data Analysis Figures")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Sales Trends",
            "Promotion and Calendar",
            "Product Families",
            "Error Analysis"
        ]
    )

    with tab1:
        show_figure(
            "daily_sales_trend.png",
            "Total daily sales over time"
        )

        show_figure(
            "monthly_sales_trend.png",
            "Monthly sales trend"
        )

        show_figure(
            "top_family_sales_trends.png",
            "Daily sales trends for top product families"
        )

    with tab2:
        show_figure(
            "sales_by_day_of_week.png",
            "Average sales by day of week"
        )

        show_figure(
            "promotion_vs_no_promotion.png",
            "Average sales with and without promotion"
        )

        show_figure(
            "correlation_matrix.png",
            "Correlation matrix"
        )

    with tab3:
        show_figure(
            "top_product_families.png",
            "Top product families by total sales"
        )

        show_figure(
            "top_promotion_families.png",
            "Top product families by promotion ratio"
        )

        show_figure(
            "produce_zero_sales_ratio_over_time.png",
            "PRODUCE zero-sales ratio over time"
        )

    with tab4:
        show_figure(
            "family_prediction_error.png",
            "Top families by prediction error"
        )

        show_figure(
            "relative_error_by_family.png",
            "Relative error by product family"
        )

        show_figure(
            "catboost_feature_importance.png",
            "CatBoost feature importance"
        )


elif section == "Prediction Output":

    st.header("Prediction Output")

    full_prediction_path = PREDICTIONS_DIR / "submission_lightgbm.csv"
    sample_prediction_path = METRICS_DIR / "sample_predictions.csv"

    if full_prediction_path.exists():
        predictions = pd.read_csv(full_prediction_path)

        st.success("Full local prediction file found.")
        st.code(str(full_prediction_path))

    elif sample_prediction_path.exists():
        predictions = pd.read_csv(sample_prediction_path)

        st.info(
            "Sample prediction file is shown for the deployed dashboard. "
            "The full prediction file is generated locally and excluded from GitHub."
        )
        st.code(str(sample_prediction_path))

    else:
        predictions = None

        st.warning(
            "No prediction sample was found. Run the inference script locally first:"
        )

        st.code("python src/inference/predict.py")

    if predictions is not None:
        st.subheader("Prediction Preview")
        st.dataframe(predictions.head(50), use_container_width=True)

        st.write("Prediction shape:")
        st.write(predictions.shape)

        st.subheader("Sales Prediction Distribution")
        st.bar_chart(predictions["sales"].head(200))

elif section == "Project Notes":

    st.header("Project Notes")

    st.subheader("Why time-based validation?")
    st.write(
        "Random splitting is not suitable for forecasting because it can mix future observations "
        "into the training process. This project uses a time-based validation split to simulate "
        "real future prediction."
    )

    st.subheader("Why log-transform the target?")
    st.write(
        "Sales are highly skewed and contain large outliers. The model was trained on log1p(sales), "
        "then predictions were transformed back using expm1."
    )

    st.subheader("Why LightGBM?")
    st.write(
        "LightGBM provided the best RMSLE on the internal validation split and outperformed both "
        "CatBoost and the naive lag-based baseline."
    )

    st.subheader("Important Limitation")
    st.write(
        "The reported RMSLE is based on an internal time-based validation split. It is not directly "
        "equivalent to Kaggle's hidden leaderboard score."
    )