import os
from pathlib import Path

import pandas as pd


GROUP_COLS = ["store_id", "item_id"]


def load_input_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Loaded input data:", df.shape)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    df["sell_price"] = (
        df.groupby(GROUP_COLS)["sell_price"]
        .transform(lambda x: x.ffill().bfill())
    )

    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.weekday
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_7"] = df.groupby(GROUP_COLS)["sales_qty"].shift(7)
    df["lag_14"] = df.groupby(GROUP_COLS)["sales_qty"].shift(14)
    df["lag_28"] = df.groupby(GROUP_COLS)["sales_qty"].shift(28)
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rolling_mean_7"] = (
        df.groupby(GROUP_COLS)["sales_qty"]
        .transform(lambda x: x.shift(1).rolling(window=7).mean())
    )
    df["rolling_mean_28"] = (
        df.groupby(GROUP_COLS)["sales_qty"]
        .transform(lambda x: x.shift(1).rolling(window=28).mean())
    )
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_lag_7"] = df.groupby(GROUP_COLS)["sell_price"].shift(7)
    df["price_change_7"] = df["sell_price"] - df["price_lag_7"]
    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "price_lag_7",
    ]

    before_shape = df.shape
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after_shape = df.shape

    print("Before dropna:", before_shape)
    print("After dropna:", after_shape)
    return df


def save_output(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved feature dataset to: {output_path}")
    print("Final shape:", df.shape)


def build_forecasting_features(input_path: str, output_path: str) -> None:
    df = load_input_data(input_path)
    df = preprocess_data(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_price_features(df)
    df = drop_na_rows(df)
    save_output(df, output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "daily_sales_chunk_subset.csv"
    output_path = project_root / "data" / "processed" / "forecasting_features.csv"
    build_forecasting_features(str(input_path), str(output_path))
