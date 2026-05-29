import pandas as pd
import numpy as np


def create_date_features(df):
    """
    Create date-based features.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    return df

def create_holiday_feature(df, holidays_df):

    holidays_df = holidays_df.copy()

    holidays_df["date"] = pd.to_datetime(
        holidays_df["date"]
    )

    holidays_df = holidays_df[
        holidays_df["type"].isin(
            ["Holiday", "Event", "Additional", "Bridge"]
        )
    ]

    holiday_dates = holidays_df["date"].unique()

    df["is_holiday"] = (
        df["date"]
        .isin(holiday_dates)
        .astype(int)
    )

    return df

def create_oil_feature(df, oil_df):

    oil_df = oil_df.copy()

    oil_df["date"] = pd.to_datetime(
        oil_df["date"]
    )

    oil_df["dcoilwtico_filled"] = (
        oil_df["dcoilwtico"]
        .ffill()
        .bfill()
    )

    df = df.merge(
        oil_df[
            [
                "date",
                "dcoilwtico_filled"
            ]
        ],
        on="date",
        how="left"
    )

    df["dcoilwtico_filled"] = (
        df["dcoilwtico_filled"]
        .ffill()
        .bfill()
    )

    return df

def merge_store_features(df, stores_df):

    df = df.merge(
        stores_df,
        on="store_nbr",
        how="left"
    )

    return df

def create_lag_features(df):

    df = df.copy()

    df = df.sort_values(
        ["store_nbr", "family", "date"]
    )

    for lag in [1, 7, 14, 21, 28, 56]:
        df[f"lag_{lag}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(lag)
        )

    return df

def create_rolling_features(df):

    df = df.copy()

    group_cols = ["store_nbr", "family"]

    for window in [7, 14, 30, 60, 90]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols)["sales"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        )

    for window in [7, 14, 30, 60]:
        df[f"rolling_std_{window}"] = (
            df.groupby(group_cols)["sales"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=2)
                .std()
            )
        )

    return df

def create_promotion_features(df):

    df = df.copy()

    group_cols = ["store_nbr", "family"]

    df["has_promotion"] = (
        df["onpromotion"] > 0
    ).astype(int)

    for lag in [1, 7, 14]:
        df[f"promotion_lag_{lag}"] = (
            df.groupby(group_cols)["onpromotion"]
            .shift(lag)
        )

    for window in [7, 14, 30]:
        df[f"promotion_rolling_mean_{window}"] = (
            df.groupby(group_cols)["onpromotion"]
            .transform(
                lambda x: x.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        )

    return df