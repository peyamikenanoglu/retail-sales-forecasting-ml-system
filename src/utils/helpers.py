import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

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
