import os
import pandas as pd


def load_raw_data(base_path: str):
    sales_path = os.path.join(base_path, "sales_train_validation.csv")
    calendar_path = os.path.join(base_path, "calendar.csv")
    prices_path = os.path.join(base_path, "sell_prices.csv")

    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    prices_df = pd.read_csv(prices_path)

    print("Loaded sales_df:", sales_df.shape)
    print("Loaded calendar_df:", calendar_df.shape)
    print("Loaded prices_df:", prices_df.shape)

    return sales_df, calendar_df, prices_df


def melt_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
    ]
    day_columns = [col for col in sales_df.columns if col.startswith("d_")]

    sales_long = sales_df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales_qty"
    )

    print("After melt:", sales_long.shape)
    return sales_long


def merge_calendar(sales_long: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    calendar_use = calendar_df[
        [
            "date", "wm_yr_wk", "d", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"
        ]
    ].copy()

    merged = sales_long.merge(calendar_use, on="d", how="left")

    print("After merging calendar:", merged.shape)
    return merged


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    def get_snap(row):
        if row["state_id"] == "CA":
            return row["snap_CA"]
        elif row["state_id"] == "TX":
            return row["snap_TX"]
        elif row["state_id"] == "WI":
            return row["snap_WI"]
        return 0

    df["snap_flag"] = df.apply(get_snap, axis=1)
    return df


def merge_prices(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    prices_use = prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]].copy()

    merged = df.merge(
        prices_use,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    print("After merging prices:", merged.shape)
    return merged


def reduce_memory_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    categorical_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "d", "weekday", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def build_daily_sales_dataset(raw_dir: str, output_path: str):
    sales_df, calendar_df, prices_df = load_raw_data(raw_dir)

    sales_long = melt_sales_data(sales_df)
    merged_df = merge_calendar(sales_long, calendar_df)
    merged_df = add_snap_flag(merged_df)
    merged_df = merge_prices(merged_df, prices_df)
    merged_df = reduce_memory_and_sort(merged_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved processed dataset to: {output_path}")
    print("Final shape:", merged_df.shape)
    print(merged_df.head())


if __name__ == "__main__":
    raw_dir = "data/raw"
    output_path = "data/processed/daily_sales_full.csv"
    build_daily_sales_dataset(raw_dir, output_path)