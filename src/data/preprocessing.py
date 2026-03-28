import os
from pathlib import Path

import pandas as pd


SELECTED_STORES = ["CA_1", "TX_1", "WI_1"]
LOOKBACK_DAYS = 365


def process_in_chunks(raw_dir: str, output_path: str, chunk_size: int = 2000) -> None:
    sales_path = os.path.join(raw_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_dir, "calendar.csv")
    prices_path = os.path.join(raw_dir, "sell_prices.csv")

    for file_path in [sales_path, calendar_path, prices_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required input file: {file_path}")

    print("Loading calendar and price metadata...")
    calendar_df = pd.read_csv(calendar_path)[["d", "date", "wm_yr_wk"]]
    prices_df = pd.read_csv(prices_path)
    prices_df = prices_df[prices_df["store_id"].isin(SELECTED_STORES)].copy()

    reader = pd.read_csv(sales_path, chunksize=chunk_size)
    first_chunk = True

    for chunk_idx, chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx}...")
        chunk = chunk[chunk["store_id"].isin(SELECTED_STORES)].copy()
        if chunk.empty:
            continue

        id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_columns = [col for col in chunk.columns if col.startswith("d_")][-LOOKBACK_DAYS:]

        sales_long = chunk.melt(
            id_vars=id_columns,
            value_vars=day_columns,
            var_name="d",
            value_name="sales_qty",
        )

        sales_long = sales_long.merge(calendar_df, on="d", how="left")
        sales_long = sales_long.merge(
            prices_df,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if first_chunk:
            sales_long.to_csv(output_path, index=False, mode="w")
            first_chunk = False
        else:
            sales_long.to_csv(output_path, index=False, mode="a", header=False)

    print(f"Saved processed sales data to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "processed" / "daily_sales_chunk_subset.csv"

    print(f"Project root: {project_root}")
    process_in_chunks(str(raw_dir), str(output_path))
