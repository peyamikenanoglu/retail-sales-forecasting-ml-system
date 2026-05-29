################################################
# Retail Store Sales Forecasting
# Step 01 - Data Loading and Basic Inspection
################################################

# 1. Import Libraries
# 2. Helper Functions
# 3. Load Main Training Data
# 4. Basic Data Inspection
# 5. Variable Type Analysis
# 6. Target Analysis
# 7. Date Analysis
# 8. First Business Summaries

import sys
from pathlib import Path

PROJECT_ROOT = Path("D:/GitHub/retail-sales-forecasting-ml-system")
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

from src.utils.helpers import *
from src.features.build_features import *

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

################################################
# Figure Save Path
################################################

FIGURES_DIR = Path("D:/GitHub/retail-sales-forecasting-ml-system/outputs/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

################################################
# 1. Helper Functions
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("\n##################### Types #####################")
    print(dataframe.dtypes)

    print("\n##################### Head #####################")
    print(dataframe.head(head))

    print("\n##################### Tail #####################")
    print(dataframe.tail(head))

    print("\n##################### NA #####################")
    print(dataframe.isnull().sum())

    print("\n##################### Duplicate Rows #####################")
    print(dataframe.duplicated().sum())

    print("\n##################### Quantiles #####################")
    numeric_df = dataframe.select_dtypes(include=["number"])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [
        col for col in dataframe.columns
        if str(dataframe[col].dtype) in ["object", "str", "string", "category"]
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th
        and pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th
        and str(dataframe[col].dtype) in ["object", "str", "string", "category"]
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name, plot=False):
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })

    print(summary)
    print("##########################################")

    if plot:
        dataframe[col_name].value_counts().plot(kind="bar")
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.title(col_name)
        plt.tight_layout()
        plt.show(block=True)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=30)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.tight_layout()
        plt.show(block=True)
def target_summary_with_cat(dataframe, target, categorical_col):
    print(
        dataframe.groupby(categorical_col)
        .agg({target: ["count", "mean", "median", "sum"]})
        .sort_values((target, "sum"), ascending=False),
        end="\n\n"
    )
def target_summary_with_num(dataframe, target, numerical_col):
    print(
        dataframe.groupby(numerical_col)
        .agg({target: ["count", "mean", "median", "sum"]}),
        end="\n\n"
    )

def save_plot(filename):
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
def plot_line(dataframe, x_col, y_col, title, xlabel, ylabel, filename, figsize=(18, 6), rotate_xticks=False):
    plt.figure(figsize=figsize)
    plt.plot(dataframe[x_col], dataframe[y_col])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=90)

    save_plot(filename)
def plot_bar(dataframe, x_col, y_col, title, xlabel, ylabel, filename, figsize=(12, 6), rotate_xticks=False):
    plt.figure(figsize=figsize)
    plt.bar(dataframe[x_col], dataframe[y_col])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=45)

    save_plot(filename)

def plot_multiple_lines(dataframe, x_col, y_col, group_col,
                        title, xlabel, ylabel, filename,
                        figsize=(18, 6)):

    plt.figure(figsize=figsize)

    groups = dataframe[group_col].unique()

    for group in groups:
        subset = dataframe[dataframe[group_col] == group]

        plt.plot(
            subset[x_col],
            subset[y_col],
            label=group
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    save_plot(filename)

def plot_heatmap(corr_matrix, title, filename, figsize=(10, 8)):

    plt.figure(figsize=figsize)

    plt.imshow(corr_matrix, aspect="auto")

    plt.colorbar()

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=45
    )

    plt.yticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns
    )

    plt.title(title)

    save_plot(filename)

def plot_feature_importance(feature_imp_df,
                            title,
                            filename,
                            top_n=15):

    plot_df = feature_imp_df.head(top_n)

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(title)

    plt.xlabel("Importance")

    save_plot(filename)

################################################
# 2. Load Main Training Data
################################################

df = pd.read_csv("D:/GitHub/retail-sales-forecasting-ml-system/data/raw/train.csv")

stores = pd.read_csv("D:/GitHub/retail-sales-forecasting-ml-system/data/raw/stores.csv")

holidays  = pd.read_csv("D:/GitHub/retail-sales-forecasting-ml-system/data/raw/holidays_events.csv")

oil  = pd.read_csv("D:/GitHub/retail-sales-forecasting-ml-system/data/raw/oil.csv")

################################################
# 3. Basic Data Inspection
################################################

check_df(df)


################################################
# 4. Variable Type Analysis
################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

print("\n##################### Categorical Columns #####################")
print(cat_cols)

print("\n##################### Numerical Columns #####################")
print(num_cols)

print("\n##################### Categorical but Cardinal Columns #####################")
print(cat_but_car)


################################################
# 5. Categorical Variable Inspection
################################################

print("\n##################### Family Summary #####################")
cat_summary(df, "family", plot=False)


################################################
# 6. Numerical Variable Inspection
################################################

print("\n##################### Sales Summary #####################")
num_summary(df, "sales", plot=False)

print("\n##################### Onpromotion Summary #####################")
num_summary(df, "onpromotion", plot=False)


################################################
# 7. Target Analysis
################################################

print("\n##################### Target: sales #####################")
print("Negative sales count:", (df["sales"] < 0).sum())
print("Zero sales count:", (df["sales"] == 0).sum())
print("Zero sales ratio:", round((df["sales"] == 0).mean() * 100, 2))


################################################
# 8. Date Analysis
################################################

df["date"] = pd.to_datetime(df["date"])

print("\n##################### Date Analysis #####################")
print("Min date:", df["date"].min())
print("Max date:", df["date"].max())
print("Number of unique dates:", df["date"].nunique())


################################################
# 10. Date Feature Extraction for EDA
################################################

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

print("\n##################### Date Feature Check #####################")
print(df[["date", "year", "month", "day", "day_of_week", "week_of_year", "is_weekend"]].head())

print("\n##################### Sales by Year #####################")
print(df.groupby("year")["sales"].agg(["count", "mean", "median", "sum"]))

print("\n##################### Sales by Month #####################")
print(df.groupby("month")["sales"].agg(["count", "mean", "median", "sum"]))

print("\n##################### Sales by Day of Week #####################")
print(df.groupby("day_of_week")["sales"].agg(["count", "mean", "median", "sum"]))

print("\n##################### Sales by Weekend #####################")
print(df.groupby("is_weekend")["sales"].agg(["count", "mean", "median", "sum"]))

################################################
# 9. Key Structure Check
################################################

print("\n##################### Key Structure #####################")
print("Number of stores:", df["store_nbr"].nunique())
print("Number of families:", df["family"].nunique())

print("\nDuplicate date-store-family rows:")
print(df.duplicated(subset=["date", "store_nbr", "family"]).sum())


################################################
# 10. First Business Summaries
################################################

print("\n##################### Sales by Family #####################")
target_summary_with_cat(df, "sales", "family")

print("\n##################### Sales by Store #####################")
target_summary_with_num(df, "sales", "store_nbr")


################################################
# 11. Manual Checks
################################################

# Use these manually in PyCharm console when needed:
# df.head()
# df.shape
# df.info()
# df.describe()
# df["family"].value_counts()
# df.groupby("family")["sales"].sum().sort_values(ascending=False).head(15)


################################################
# 12. Visualization
################################################

daily_sales = df.groupby("date")["sales"].sum().reset_index()

plot_line(
    daily_sales,
    x_col="date",
    y_col="sales",
    title="Total Daily Sales Over Time",
    xlabel="Date",
    ylabel="Total Sales",
    filename="daily_sales_trend.png"
)


monthly_sales = df.groupby(["year", "month"])["sales"].sum().reset_index()

monthly_sales["year_month"] = (
    monthly_sales["year"].astype(str)
    + "-"
    + monthly_sales["month"].astype(str).str.zfill(2)
)

plot_line(
    monthly_sales,
    x_col="year_month",
    y_col="sales",
    title="Monthly Sales Trend",
    xlabel="Year-Month",
    ylabel="Total Sales",
    filename="monthly_sales_trend.png",
    rotate_xticks=True
)


dow_sales = df.groupby("day_of_week")["sales"].mean().reset_index()

day_names = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun"
}

dow_sales["day_name"] = dow_sales["day_of_week"].map(day_names)

plot_bar(
    dow_sales,
    x_col="day_name",
    y_col="sales",
    title="Average Sales by Day of Week",
    xlabel="Day",
    ylabel="Average Sales",
    filename="sales_by_day_of_week.png"
)


top_families = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plot_bar(
    top_families,
    x_col="family",
    y_col="sales",
    title="Top 10 Product Families by Total Sales",
    xlabel="Family",
    ylabel="Total Sales",
    filename="top_product_families.png",
    rotate_xticks=True
)


################################################
# Store Information
################################################

print("\n##################### STORES #####################")
check_df(stores)

print("\n##################### STORES COLUMN TYPES #####################")
grab_col_names(stores)

print("\n##################### Store Type #####################")
cat_summary(stores, "type")

print("\n##################### Store Cluster #####################")
cat_summary(stores, "cluster")


################################################
# Merge Store Information
################################################

df = df.merge(
    stores,
    how="left",
    on="store_nbr"
)

print("\n##################### AFTER STORE MERGE #####################")
print(df.shape)

print("\n##################### Missing Values After Merge #####################")
print(df[["city", "state", "type", "cluster"]].isnull().sum())


################################################
# Sales by Store Type
################################################

sales_by_type = (df.groupby("type")["sales"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False))

print("\n##################### Sales by Store Type #####################")
print(sales_by_type)

################################################
# Sales by Store Type Figure
################################################

sales_by_type_plot = (
    df.groupby("type")["sales"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

plot_bar(
    sales_by_type_plot,
    x_col="type",
    y_col="sales",
    title="Total Sales by Store Type",
    xlabel="Store Type",
    ylabel="Total Sales",
    filename="sales_by_store_type.png"
)

################################################
# Sales by City
################################################

sales_by_city = (df.groupby("city")["sales"].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False))

print("\n##################### Sales by City #####################")
print(sales_by_city.head(15))

################################################
# Top Cities by Sales
################################################

top_cities = (
    df.groupby("city")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plot_bar(
    top_cities,
    x_col="city",
    y_col="sales",
    title="Top Cities by Total Sales",
    xlabel="City",
    ylabel="Total Sales",
    filename="top_cities_sales.png",
    rotate_xticks=True
)


################################################
# Promotion Analysis
################################################

df["has_promotion"] = (df["onpromotion"] > 0).astype(int)

promotion_summary = (
    df.groupby("has_promotion")["sales"]
    .agg(["count", "mean", "median", "sum"])
)

print("\n##################### Promotion Summary #####################")
print(promotion_summary)


################################################
# Sales With vs Without Promotion
################################################

promotion_plot = (
    df.groupby("has_promotion")["sales"]
    .mean()
    .reset_index()
)

promotion_plot["promotion_label"] = (
    promotion_plot["has_promotion"]
    .map({
        0: "No Promotion",
        1: "Promotion"
    })
)

plot_bar(
    promotion_plot,
    x_col="promotion_label",
    y_col="sales",
    title="Average Sales: Promotion vs No Promotion",
    xlabel="Promotion Status",
    ylabel="Average Sales",
    filename="promotion_vs_no_promotion.png"
)


################################################
# Promotion Ratio by Family
################################################

promotion_by_family = (
    df.groupby("family")["has_promotion"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

promotion_by_family["promotion_ratio_percent"] = (
    promotion_by_family["has_promotion"] * 100
)

print("\n##################### Promotion Ratio by Family #####################")
print(promotion_by_family.head(15))

################################################
# Top Families by Promotion Ratio
################################################

top_promo_families = promotion_by_family.head(10)

plot_bar(
    top_promo_families,
    x_col="family",
    y_col="promotion_ratio_percent",
    title="Top Product Families by Promotion Ratio",
    xlabel="Family",
    ylabel="Promotion Ratio (%)",
    filename="top_promotion_families.png",
    rotate_xticks=True
)


################################################
# Family Time Series Analysis
################################################

top_5_families = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
)

family_daily_sales = (
    df[df["family"].isin(top_5_families)]
    .groupby(["date", "family"])["sales"]
    .sum()
    .reset_index()
)

print("\n##################### Top 5 Families #####################")
print(top_5_families)

print("\n##################### Family Daily Sales #####################")
print(family_daily_sales.head())

################################################
# Top Family Sales Trends
################################################

plot_multiple_lines(
    family_daily_sales,
    x_col="date",
    y_col="sales",
    group_col="family",
    title="Daily Sales Trends for Top Product Families",
    xlabel="Date",
    ylabel="Sales",
    filename="top_family_sales_trends.png"
)

################################################
# Rolling Mean Sales Trends
################################################

rolling_family_sales = family_daily_sales.copy()

rolling_family_sales["rolling_mean_30"] = (
    rolling_family_sales
    .groupby("family")["sales"]
    .transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
)

print("\n##################### Rolling Mean Sample #####################")
print(
    rolling_family_sales[
        ["date", "family", "sales", "rolling_mean_30"]
    ].head()
)


print(
    rolling_family_sales[
        rolling_family_sales["family"] == "BEVERAGES"
    ][
        ["date", "family", "sales", "rolling_mean_30"]
    ].head(40)
)


################################################
# Load Holidays Dataset
################################################


print("\n##################### HOLIDAYS #####################")
check_df(holidays)

print("\n##################### HOLIDAY COLUMN TYPES #####################")
grab_col_names(holidays)

print("\n##################### Holiday Type #####################")
cat_summary(holidays, "type")

print("\n##################### Holiday Locale #####################")
cat_summary(holidays, "locale")

print("\n##################### Holiday Transferred #####################")
cat_summary(holidays, "transferred")



holidays["date"] = pd.to_datetime(holidays["date"])

################################################
# Clean Holiday Dataset
################################################

holiday_clean = holidays.copy()

holiday_clean = holiday_clean[
    holiday_clean["transferred"] == False
]

holiday_clean = holiday_clean[
    holiday_clean["type"].isin([
        "Holiday",
        "Additional",
        "Bridge",
        "Event"
    ])
]

print("\n##################### Clean Holiday Dataset #####################")
print(holiday_clean.shape)

print("\n##################### Holiday Types After Cleaning #####################")
print(holiday_clean["type"].value_counts())


################################################
# Create Holiday Feature
################################################

holiday_dates = holiday_clean["date"].unique()

df["is_holiday"] = (
    df["date"].isin(holiday_dates)
).astype(int)

print("\n##################### Holiday Feature #####################")
print(df["is_holiday"].value_counts())

print("\n##################### Sales by Holiday #####################")
print(
    df.groupby("is_holiday")["sales"]
    .agg(["count", "mean", "median", "sum"])
)


################################################
# Load Oil Dataset
################################################


print("\n##################### OIL #####################")
check_df(oil)

print("\n##################### OIL COLUMN TYPES #####################")
grab_col_names(oil)

oil["date"] = pd.to_datetime(oil["date"])

################################################
# Oil Missing Values
################################################

print("\n##################### Oil Missing Values #####################")
print(oil.isnull().sum())

print("\nMissing Ratio (%)")
print(
    (oil.isnull().sum() / len(oil) * 100)
)

plot_line(
    oil,
    x_col="date",
    y_col="dcoilwtico",
    title="Oil Prices Over Time",
    xlabel="Date",
    ylabel="Oil Price",
    filename="oil_prices_over_time.png"
)


################################################
# Fill Missing Oil Prices
################################################

oil["dcoilwtico_filled"] = (
    oil["dcoilwtico"]
    .ffill()
    .bfill()
)

print("\n##################### Missing After Fill #####################")
print(oil.isnull().sum())

################################################
# Filled Oil Prices Plot
################################################

plot_line(
    oil,
    x_col="date",
    y_col="dcoilwtico_filled",
    title="Filled Oil Prices Over Time",
    xlabel="Date",
    ylabel="Oil Price",
    filename="filled_oil_prices.png"
)


################################################
# Prepare Oil Dataset for Merge
################################################

oil_final = oil[
    ["date", "dcoilwtico_filled"]
].copy()

print("\n##################### OIL FINAL #####################")
print(oil_final.head())


################################################
# Merge Oil Prices
################################################

df = df.merge(
    oil_final,
    how="left",
    on="date"
)

print("\n##################### AFTER OIL MERGE #####################")
print(df.shape)

print("\n##################### Oil Missing After Merge #####################")
print(df["dcoilwtico_filled"].isnull().sum())


df["dcoilwtico_filled"] = (
    df["dcoilwtico_filled"]
    .ffill()
    .bfill()
)

print("\n##################### Oil Missing After Final Fill #####################")
print(df["dcoilwtico_filled"].isnull().sum())

################################################
# Sales by Oil Price Levels
################################################

df["oil_price_level"] = pd.qcut(
    df["dcoilwtico_filled"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"]
)

print("\n##################### Sales by Oil Price Level #####################")

print(
    df.groupby("oil_price_level")["sales"]
    .agg(["count", "mean", "median", "sum"])
)


################################################
# Normalize Sales by Year
################################################

yearly_mean_sales = (
    df.groupby("year")["sales"]
    .mean()
)

df["yearly_avg_sales"] = (
    df["year"]
    .map(yearly_mean_sales)
)

df["sales_normalized_by_year"] = (
    df["sales"] / df["yearly_avg_sales"]
)

print("\n##################### Normalized Sales Sample #####################")

print(
    df[
        ["date", "sales", "yearly_avg_sales",
         "sales_normalized_by_year"]
    ].head()
)

################################################
# Normalized Sales by Oil Level
################################################

print("\n##################### Normalized Sales by Oil Level #####################")

print(
    df.groupby("oil_price_level")["sales_normalized_by_year"]
    .agg(["count", "mean", "median"])
)


################################################
# Correlation Analysis
################################################

corr_cols = [
    "sales",
    "onpromotion",
    "dcoilwtico_filled",
    "is_holiday",
    "is_weekend",
    "month",
    "day_of_week",
    "year"
]

corr_df = df[corr_cols].corr()

print("\n##################### Correlation Matrix #####################")
print(corr_df)


plot_heatmap(
    corr_df,
    title="Correlation Matrix",
    filename="correlation_matrix.png"
)


################################################
# Sort Data Before Lag Features
################################################

df = df.sort_values(
    by=["store_nbr", "family", "date"]
)

print("\n##################### Data Sorted #####################")
print(df[["store_nbr", "family", "date"]].head())


################################################
# Lag Features
################################################

df["lag_1"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
)

df["lag_7"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(7)
)

print("\n##################### Lag Features Sample #####################")

print(
    df[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "lag_1",
            "lag_7"
        ]
    ].head(20)
)

################################################
# Rolling Features
################################################

df["rolling_mean_7"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
)

df["rolling_mean_30"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
)

print("\n##################### Rolling Features Sample #####################")

print(
    df[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "lag_1",
            "lag_7",
            "rolling_mean_7",
            "rolling_mean_30"
        ]
    ].head(20)
)


################################################
# Time-Based Train Validation Split
################################################

split_date = "2017-07-01"

train_df = df[
    df["date"] < split_date
].copy()

valid_df = df[
    df["date"] >= split_date
].copy()

print("\n##################### Train Shape #####################")
print(train_df.shape)

print("\n##################### Validation Shape #####################")
print(valid_df.shape)

print("\n##################### Train Date Range #####################")
print(
    train_df["date"].min(),
    " --> ",
    train_df["date"].max()
)

print("\n##################### Validation Date Range #####################")
print(
    valid_df["date"].min(),
    " --> ",
    valid_df["date"].max()
)


################################################
# Feature Selection
################################################

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

print("\n##################### Number of Features #####################")
print(len(features))

print("\n##################### Features #####################")
print(features)


################################################
# Train Validation Matrices
################################################

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]

print("\n##################### X_train Shape #####################")
print(X_train.shape)

print("\n##################### X_valid Shape #####################")
print(X_valid.shape)

print("\n##################### y_train Shape #####################")
print(y_train.shape)

print("\n##################### y_valid Shape #####################")
print(y_valid.shape)



################################################
# Missing Values Before Final Fill
################################################

print("\n##################### Train Missing Values #####################")

print(
    X_train.isnull()
    .sum()
    .sort_values(ascending=False)
)

print("\n##################### Validation Missing Values #####################")

print(
    X_valid.isnull()
    .sum()
    .sort_values(ascending=False)
)

################################################
# Fill Lag and Rolling Missing Values
################################################

X_train = X_train.fillna(0)
X_valid = X_valid.fillna(0)

print("\n##################### Missing After Fill #####################")

print(X_train.isnull().sum().sum())
print(X_valid.isnull().sum().sum())


################################################
# Categorical Feature Indices
################################################

cat_feature_indices = [
    X_train.columns.get_loc(col)
    for col in cat_features
]

print("\n##################### Cat Feature Indices #####################")
print(cat_feature_indices)


################################################
# Baseline CatBoost Model
################################################

catboost_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

catboost_model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

################################################
# Validation Predictions
################################################

y_pred = catboost_model.predict(X_valid)

print("\n##################### Prediction Sample #####################")

print(y_pred[:10])

################################################
# Validation Metrics
################################################

mae = mean_absolute_error(y_valid, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_valid, y_pred)
)

print("\n##################### Validation Metrics #####################")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


################################################
# Naive Forecast Baseline
################################################

naive_pred = X_valid["lag_1"]

naive_mae = mean_absolute_error(
    y_valid,
    naive_pred
)

naive_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        naive_pred
    )
)

print("\n##################### Naive Forecast Metrics #####################")

print(f"Naive MAE  : {naive_mae:.4f}")
print(f"Naive RMSE : {naive_rmse:.4f}")


################################################
# Train Predictions
################################################

y_train_pred = catboost_model.predict(X_train)

################################################
# Train Metrics
################################################

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

print("\n##################### Train Metrics #####################")

print(f"Train MAE  : {train_mae:.4f}")
print(f"Train RMSE : {train_rmse:.4f}")


################################################
# Feature Importance
################################################

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": catboost_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n##################### Feature Importance #####################")

print(feature_importance)


plot_feature_importance(
    feature_importance,
    title="CatBoost Feature Importance",
    filename="catboost_feature_importance.png"
)


################################################
# Log Transformed Targets
################################################

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

print("\n##################### Log Target Sample #####################")

print(y_train_log.head())


################################################
# CatBoost with Log Target
################################################

catboost_log_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100
)

catboost_log_model.fit(
    X_train,
    y_train_log,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid_log),
    use_best_model=True
)

################################################
# Log Model Predictions
################################################

y_pred_log = catboost_log_model.predict(X_valid)

print("\n##################### Log Prediction Sample #####################")

print(y_pred_log[:10])

################################################
# Inverse Transform Predictions
################################################

y_pred_log_inverse = np.expm1(y_pred_log)

print("\n##################### Inverse Prediction Sample #####################")

print(y_pred_log_inverse[:10])

################################################
# Log Model Metrics
################################################

log_mae = mean_absolute_error(
    y_valid,
    y_pred_log_inverse
)

log_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        y_pred_log_inverse
    )
)

print("\n##################### Log Model Metrics #####################")

print(f"Log MAE  : {log_mae:.4f}")
print(f"Log RMSE : {log_rmse:.4f}")



################################################
# Zero Sales Deep Dive
################################################

zero_sales_by_family = (
    df.groupby("family")["sales"]
    .apply(lambda x: (x == 0).mean() * 100)
    .sort_values(ascending=False)
    .reset_index(name="zero_sales_ratio_percent")
)

print("\n##################### Zero Sales Ratio by Family #####################")
print(zero_sales_by_family)


################################################
# Produce Zero Sales Analysis
################################################

produce_df = df[df["family"] == "PRODUCE"].copy()

print("\n##################### PRODUCE Shape #####################")
print(produce_df.shape)

print("\n##################### PRODUCE Sales Summary #####################")
print(produce_df["sales"].describe())

print("\n##################### PRODUCE Zero Sales Ratio #####################")
print(round((produce_df["sales"] == 0).mean() * 100, 2))


produce_zero_by_year = (
    produce_df.groupby("year")["sales"]
    .apply(lambda x: (x == 0).mean() * 100)
    .reset_index(name="zero_sales_ratio_percent")
)

print("\n##################### PRODUCE Zero Sales Ratio by Year #####################")
print(produce_zero_by_year)


produce_zero_by_store = (
    produce_df.groupby("store_nbr")["sales"]
    .apply(lambda x: (x == 0).mean() * 100)
    .sort_values(ascending=False)
    .reset_index(name="zero_sales_ratio_percent")
)

print("\n##################### PRODUCE Zero Sales Ratio by Store #####################")
print(produce_zero_by_store.head(20))


produce_daily_zero = (
    produce_df.groupby("date")["sales"]
    .apply(lambda x: (x == 0).mean() * 100)
    .reset_index(name="zero_sales_ratio_percent")
)

print("\n##################### PRODUCE Daily Zero Ratio Sample #####################")
print(produce_daily_zero.head())


plot_line(
    produce_daily_zero,
    x_col="date",
    y_col="zero_sales_ratio_percent",
    title="PRODUCE Zero Sales Ratio Over Time",
    xlabel="Date",
    ylabel="Zero Sales Ratio (%)",
    filename="produce_zero_sales_ratio_over_time.png"
)


################################################
# Validation Results DataFrame
################################################

valid_results = valid_df.copy()

valid_results["actual_sales"] = y_valid.values

valid_results["predicted_sales"] = y_pred_log_inverse

valid_results["absolute_error"] = (
    np.abs(
        valid_results["actual_sales"]
        - valid_results["predicted_sales"]
    )
)

print("\n##################### Validation Results Sample #####################")

print(
    valid_results[
        [
            "date",
            "store_nbr",
            "family",
            "actual_sales",
            "predicted_sales",
            "absolute_error"
        ]
    ].head()
)

################################################
# Error Analysis by Family
################################################

family_error_analysis = (
    valid_results.groupby("family")["absolute_error"]
    .mean()
    .sort_values(ascending=False)
    .reset_index(name="mae_by_family")
)

print("\n##################### Error Analysis by Family #####################")

print(family_error_analysis)



plot_bar(
    family_error_analysis.head(15),
    x_col="family",
    y_col="mae_by_family",
    title="Top Families by Prediction Error",
    xlabel="Family",
    ylabel="MAE",
    filename="family_prediction_error.png",
    rotate_xticks=True
)


################################################
# Relative Error Analysis
################################################

valid_results["relative_error_percent"] = (
    valid_results["absolute_error"]
    / (valid_results["actual_sales"] + 1)
) * 100


family_relative_error = (
    valid_results.groupby("family")[
        "relative_error_percent"
    ]
    .mean()
    .sort_values(ascending=False)
    .reset_index(name="mean_relative_error_percent")
)

print("\n##################### Relative Error by Family #####################")

print(family_relative_error)


plot_bar(
    family_relative_error.head(15),
    x_col="family",
    y_col="mean_relative_error_percent",
    title="Relative Error by Family",
    xlabel="Family",
    ylabel="Mean Relative Error (%)",
    filename="relative_error_by_family.png",
    rotate_xticks=True
)


################################################
# Save and Load Trained Model
################################################

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "catboost_sales_forecaster.cbm"

catboost_log_model.save_model(str(MODEL_PATH))

print("\n##################### Model Saved #####################")
print(MODEL_PATH)


loaded_model = CatBoostRegressor()
loaded_model.load_model(str(MODEL_PATH))

print("\n##################### Model Loaded Successfully #####################")


################################################
# Save Metrics and Feature Importance
################################################

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

model_comparison = pd.DataFrame({
    "model": [
        "Naive Forecast",
        "CatBoost Raw Target",
        "CatBoost Log Target"
    ],
    "mae": [
        naive_mae,
        mae,
        log_mae
    ],
    "rmse": [
        naive_rmse,
        rmse,
        log_rmse
    ]
})

model_comparison_path = METRICS_DIR / "model_comparison.csv"
model_comparison.to_csv(model_comparison_path, index=False)

feature_importance_path = METRICS_DIR / "catboost_feature_importance.csv"
feature_importance.to_csv(feature_importance_path, index=False)

print("\n##################### Saved Project Outputs #####################")
print("Model comparison saved to:", model_comparison_path)
print("Feature importance saved to:", feature_importance_path)

print("\n##################### Model Comparison #####################")
print(model_comparison)


