import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CAPACITY_BUFFER = -0.08
MIN_WAREHOUSE_CAPACITY = 3000
MIN_INITIAL_STOCK = 0
MAX_INITIAL_STOCK_COVERAGE = 8


def load_scenario_forecast_summary(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Loaded scenario forecast summary:", df.shape)
    return df


def build_forecast_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "store_id",
        "item_id",
        "forecast_demand",
        "adjusted_demand",
        "demand_low",
        "demand_base",
        "demand_high",
        "scenario_demand",
        "demand_spread",
        "uncertainty_ratio",
        "dynamic_safety_factor",
        "censored_days",
        "avg_censoring_score",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    summary = df[required_cols].copy()
    print("Forecast summary shape:", summary.shape)
    return summary


def build_warehouse_info(
    df: pd.DataFrame,
    capacity_buffer: float = DEFAULT_CAPACITY_BUFFER
) -> pd.DataFrame:
    warehouse_df = (
        df.groupby("store_id", as_index=False)["scenario_demand"]
        .sum()
        .rename(columns={"store_id": "warehouse_id", "scenario_demand": "aggregate_scenario_demand"})
    )

    warehouse_df["capacity"] = (
        warehouse_df["aggregate_scenario_demand"] * (1 + capacity_buffer)
    ).round().astype(int)

    warehouse_df["capacity"] = warehouse_df["capacity"].clip(lower=MIN_WAREHOUSE_CAPACITY)

    return warehouse_df[["warehouse_id", "capacity"]]


def classify_item_category(item_id: str) -> str:
    if isinstance(item_id, str):
        if item_id.startswith("FOODS"):
            return "FOODS"
        if item_id.startswith("HOUSEHOLD"):
            return "HOUSEHOLD"
        if item_id.startswith("HOBBIES"):
            return "HOBBIES"
    return "OTHER"


def build_sku_info(df: pd.DataFrame) -> pd.DataFrame:
    sku_df = df.groupby("item_id", as_index=False).agg(
        avg_forecast_demand=("forecast_demand", "mean"),
        avg_adjusted_demand=("adjusted_demand", "mean"),
        avg_uncertainty_ratio=("uncertainty_ratio", "mean"),
        avg_censoring_score=("avg_censoring_score", "mean"),
        avg_safety_factor=("dynamic_safety_factor", "mean"),
    )

    sku_df["category"] = sku_df["item_id"].apply(classify_item_category)

    scale = sku_df["avg_adjusted_demand"].clip(lower=1.0)
    normalized_scale = (scale / scale.median()).clip(lower=0.3, upper=3.0)

    category_holding_base = {
        "FOODS": 0.14,
        "HOUSEHOLD": 0.20,
        "HOBBIES": 0.26,
        "OTHER": 0.18,
    }
    category_stockout_base = {
        "FOODS": 4.2,
        "HOUSEHOLD": 3.4,
        "HOBBIES": 2.8,
        "OTHER": 3.2,
    }
    category_replenishment_base = {
        "FOODS": 0.95,
        "HOUSEHOLD": 1.05,
        "HOBBIES": 1.15,
        "OTHER": 1.00,
    }

    sku_df["holding_cost"] = sku_df["category"].map(category_holding_base)
    sku_df["holding_cost"] = (
        sku_df["holding_cost"] + 0.03 * normalized_scale
    ).round(4)

    sku_df["stockout_cost"] = sku_df["category"].map(category_stockout_base)
    sku_df["stockout_cost"] = (
        sku_df["stockout_cost"]
        + 1.8 * sku_df["avg_uncertainty_ratio"].clip(0, 1.2)
        + 0.8 * sku_df["avg_censoring_score"].clip(0, 1.0)
    ).round(4)

    sku_df["replenishment_cost"] = sku_df["category"].map(category_replenishment_base)
    sku_df["replenishment_cost"] = (
        sku_df["replenishment_cost"]
        + 0.08 * normalized_scale
        + 0.20 * sku_df["avg_safety_factor"].clip(0, 0.25)
    ).round(4)

    return sku_df[["item_id", "holding_cost", "stockout_cost", "replenishment_cost"]]


def build_initial_inventory(df: pd.DataFrame) -> pd.DataFrame:
    inventory_df = df[
        ["store_id", "item_id", "forecast_demand", "adjusted_demand", "uncertainty_ratio", "dynamic_safety_factor"]
    ].copy()

    base_cover = (
        0.01
        + 0.05 * inventory_df["dynamic_safety_factor"].clip(0, 0.25)
        + 0.02 * inventory_df["uncertainty_ratio"].clip(0, 0.8)
    )

    inventory_df["initial_inventory"] = (
        inventory_df["forecast_demand"] * base_cover
    ).round().astype(int)

    low_demand_mask = inventory_df["adjusted_demand"] <= 3
    inventory_df.loc[low_demand_mask, "initial_inventory"] = (
        inventory_df.loc[low_demand_mask, "initial_inventory"] + 1
    )

    inventory_df["initial_inventory"] = inventory_df["initial_inventory"].clip(
        lower=MIN_INITIAL_STOCK,
        upper=MAX_INITIAL_STOCK_COVERAGE,
    )

    return inventory_df[["store_id", "item_id", "initial_inventory"]]


def save_all(
    forecast_summary: pd.DataFrame,
    warehouse_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    base_path: str
) -> None:
    os.makedirs(base_path, exist_ok=True)
    forecast_summary.to_csv(os.path.join(base_path, "forecast_summary.csv"), index=False)
    warehouse_df.to_csv(os.path.join(base_path, "warehouse_info.csv"), index=False)
    sku_df.to_csv(os.path.join(base_path, "sku_info.csv"), index=False)
    inventory_df.to_csv(os.path.join(base_path, "initial_inventory.csv"), index=False)
    print("Saved all optimization input files.")


def run_prepare_inputs(input_path: str, output_dir: str) -> None:
    df = load_scenario_forecast_summary(input_path)
    forecast_summary = build_forecast_summary(df)
    warehouse_df = build_warehouse_info(df)
    sku_df = build_sku_info(df)
    inventory_df = build_initial_inventory(df)
    save_all(forecast_summary, warehouse_df, sku_df, inventory_df, output_dir)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "scenario_forecast_summary.csv"
    output_dir = project_root / "data" / "processed"
    run_prepare_inputs(str(input_path), str(output_dir))


if __name__ == "__main__":
    main()