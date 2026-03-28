import os
from pathlib import Path

import numpy as np
import pandas as pd
import pulp


DEFAULT_SERVICE_LEVEL = 0.90
DEFAULT_SAFETY_STOCK_FACTOR = 0.05
DEFAULT_RISK_THRESHOLD = 0.22


def load_input_data(base_dir: str):
    forecast_df = pd.read_csv(os.path.join(base_dir, "forecast_summary.csv"))
    warehouse_df = pd.read_csv(os.path.join(base_dir, "warehouse_info.csv"))
    sku_df = pd.read_csv(os.path.join(base_dir, "sku_info.csv"))
    inventory_df = pd.read_csv(os.path.join(base_dir, "initial_inventory.csv"))

    print("Loaded forecast_df:", forecast_df.shape)
    print("Loaded warehouse_df:", warehouse_df.shape)
    print("Loaded sku_df:", sku_df.shape)
    print("Loaded inventory_df:", inventory_df.shape)

    return forecast_df, warehouse_df, sku_df, inventory_df


def prepare_optimization_table(
    forecast_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    inventory_df: pd.DataFrame
) -> pd.DataFrame:
    df = forecast_df.merge(inventory_df, on=["store_id", "item_id"], how="left")
    df = df.merge(sku_df, on="item_id", how="left")
    df["initial_inventory"] = df["initial_inventory"].fillna(0)
    print("Optimization table shape:", df.shape)
    return df


def get_demand_value(row: pd.Series) -> float:
    if "scenario_demand" in row.index and pd.notna(row["scenario_demand"]):
        return float(row["scenario_demand"])
    if "adjusted_demand" in row.index and pd.notna(row["adjusted_demand"]):
        return float(row["adjusted_demand"])
    return float(row["forecast_demand"])


def get_safety_factor(row: pd.Series, default_factor: float) -> float:
    dynamic_part = row["dynamic_safety_factor"] if "dynamic_safety_factor" in row.index else np.nan
    uncertainty_part = row["uncertainty_ratio"] if "uncertainty_ratio" in row.index else np.nan
    censoring_part = row["avg_censoring_score"] if "avg_censoring_score" in row.index else np.nan

    dynamic_part = 0.0 if pd.isna(dynamic_part) else float(dynamic_part)
    uncertainty_part = 0.0 if pd.isna(uncertainty_part) else float(uncertainty_part)
    censoring_part = 0.0 if pd.isna(censoring_part) else float(censoring_part)

    factor = (
        default_factor
        + 0.06 * dynamic_part
        + 0.03 * min(uncertainty_part, 1.0)
        + 0.02 * min(censoring_part, 1.0)
    )

    return float(np.clip(factor, 0.04, 0.10))


def get_risk_score(row: pd.Series) -> float:
    uncertainty_part = row["uncertainty_ratio"] if "uncertainty_ratio" in row.index else np.nan
    dynamic_part = row["dynamic_safety_factor"] if "dynamic_safety_factor" in row.index else np.nan
    censoring_part = row["avg_censoring_score"] if "avg_censoring_score" in row.index else np.nan

    uncertainty_part = 0.0 if pd.isna(uncertainty_part) else float(uncertainty_part)
    dynamic_part = 0.0 if pd.isna(dynamic_part) else float(dynamic_part)
    censoring_part = 0.0 if pd.isna(censoring_part) else float(censoring_part)

    risk_score = (
        0.5 * uncertainty_part
        + 0.3 * dynamic_part
        + 0.2 * censoring_part
    )
    return float(risk_score)


def build_replenishment_model_with_hard_safety_stock(
    opt_df: pd.DataFrame,
    warehouse_df: pd.DataFrame,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    safety_stock_factor: float = DEFAULT_SAFETY_STOCK_FACTOR,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD
):
    model = pulp.LpProblem("Replenishment_Optimization_Hard_Safety_Stock", pulp.LpMinimize)
    keys = list(opt_df[["store_id", "item_id"]].itertuples(index=False, name=None))

    replenishment_vars = {
        key: pulp.LpVariable(f"replenishment_{key[0]}_{key[1]}", lowBound=0)
        for key in keys
    }
    ending_inventory_vars = {
        key: pulp.LpVariable(f"ending_inventory_{key[0]}_{key[1]}", lowBound=0)
        for key in keys
    }
    shortage_vars = {
        key: pulp.LpVariable(f"shortage_{key[0]}_{key[1]}", lowBound=0)
        for key in keys
    }

    model += pulp.lpSum(
        opt_df.loc[i, "replenishment_cost"] * replenishment_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        + opt_df.loc[i, "holding_cost"] * ending_inventory_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        + opt_df.loc[i, "stockout_cost"] * shortage_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        for i in opt_df.index
    )

    protected_count = 0

    for i in opt_df.index:
        store_id = opt_df.loc[i, "store_id"]
        item_id = opt_df.loc[i, "item_id"]
        key = (store_id, item_id)

        initial_inventory = float(opt_df.loc[i, "initial_inventory"])
        demand = get_demand_value(opt_df.loc[i])

        current_safety_factor = get_safety_factor(opt_df.loc[i], safety_stock_factor)
        risk_score = get_risk_score(opt_df.loc[i])

        target_safety_stock = current_safety_factor * demand if risk_score >= risk_threshold else 0.0

        if target_safety_stock > 0:
            protected_count += 1

        model += (
            initial_inventory
            + replenishment_vars[key]
            + shortage_vars[key]
            == demand + ending_inventory_vars[key]
        )

        model += shortage_vars[key] <= (1 - service_level) * demand

        if target_safety_stock > 0:
            model += ending_inventory_vars[key] >= target_safety_stock

    for _, row in warehouse_df.iterrows():
        warehouse_id = row["warehouse_id"]
        relevant_keys = [key for key in keys if key[0] == warehouse_id]

        model += pulp.lpSum(
            ending_inventory_vars[key] for key in relevant_keys
        ) <= float(row["capacity"])

    print(f"Protected SKU-store pairs with hard safety stock: {protected_count}")

    return model, replenishment_vars, ending_inventory_vars, shortage_vars


def solve_model(model) -> None:
    solver = pulp.PULP_CBC_CMD(msg=True)
    model.solve(solver)
    print("Solver Status:", pulp.LpStatus[model.status])
    print("Objective Value:", pulp.value(model.objective))


def extract_results(
    opt_df: pd.DataFrame,
    replenishment_vars,
    ending_inventory_vars,
    shortage_vars,
    default_safety_factor: float,
    risk_threshold: float
) -> pd.DataFrame:
    result_df = opt_df.copy()

    result_df["optimization_demand"] = result_df.apply(get_demand_value, axis=1)
    result_df["risk_score"] = result_df.apply(get_risk_score, axis=1)
    result_df["applied_safety_factor"] = result_df.apply(
        lambda row: get_safety_factor(row, default_safety_factor), axis=1
    )

    result_df["is_protected_sku"] = result_df["risk_score"] >= risk_threshold
    result_df["target_safety_stock"] = np.where(
        result_df["is_protected_sku"],
        result_df["optimization_demand"] * result_df["applied_safety_factor"],
        0.0
    )

    result_df["replenishment_qty"] = result_df.apply(
        lambda row: replenishment_vars[(row["store_id"], row["item_id"])].varValue, axis=1
    )
    result_df["ending_inventory"] = result_df.apply(
        lambda row: ending_inventory_vars[(row["store_id"], row["item_id"])].varValue, axis=1
    )
    result_df["shortage_qty"] = result_df.apply(
        lambda row: shortage_vars[(row["store_id"], row["item_id"])].varValue, axis=1
    )

    result_df["fill_rate"] = 1 - (
        result_df["shortage_qty"] / result_df["optimization_demand"].replace(0, 1)
    )

    result_df["replenishment_cost_component"] = (
        result_df["replenishment_cost"] * result_df["replenishment_qty"]
    )
    result_df["holding_cost_component"] = (
        result_df["holding_cost"] * result_df["ending_inventory"]
    )
    result_df["stockout_cost_component"] = (
        result_df["stockout_cost"] * result_df["shortage_qty"]
    )
    result_df["safety_penalty_cost"] = 0.0

    result_df["total_cost"] = (
        result_df["replenishment_cost_component"]
        + result_df["holding_cost_component"]
        + result_df["stockout_cost_component"]
    )

    return result_df


def save_results(result_df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def print_summary(result_df: pd.DataFrame) -> None:
    print("\n=== OPTIMIZATION SUMMARY (HARD SAFETY STOCK FOR HIGH-RISK SKUS) ===")
    print(f"Total Demand: {result_df['optimization_demand'].sum():.2f}")
    print(f"Total Replenishment: {result_df['replenishment_qty'].sum():.2f}")
    print(f"Total Ending Inventory: {result_df['ending_inventory'].sum():.2f}")
    print(f"Total Shortage: {result_df['shortage_qty'].sum():.2f}")
    print(f"Average Fill Rate: {result_df['fill_rate'].mean():.4f}")
    print(f"Average Applied Safety Factor: {result_df['applied_safety_factor'].mean():.4f}")
    print(f"Protected SKU-store pairs: {int(result_df['is_protected_sku'].sum())}")
    print(f"Total Target Safety Stock: {result_df['target_safety_stock'].sum():.2f}")
    print(f"Replenishment Cost: {result_df['replenishment_cost_component'].sum():.2f}")
    print(f"Holding Cost: {result_df['holding_cost_component'].sum():.2f}")
    print(f"Stockout Cost: {result_df['stockout_cost_component'].sum():.2f}")
    print(f"Total Cost: {result_df['total_cost'].sum():.2f}")


def run_service_level_optimization(
    base_dir: str,
    output_path: str,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    safety_stock_factor: float = DEFAULT_SAFETY_STOCK_FACTOR,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD
) -> None:
    forecast_df, warehouse_df, sku_df, inventory_df = load_input_data(base_dir)
    opt_df = prepare_optimization_table(forecast_df, sku_df, inventory_df)

    model, r_vars, e_vars, s_vars = build_replenishment_model_with_hard_safety_stock(
        opt_df,
        warehouse_df,
        service_level=service_level,
        safety_stock_factor=safety_stock_factor,
        risk_threshold=risk_threshold,
    )

    solve_model(model)

    if pulp.LpStatus[model.status] != "Optimal":
        print("Model did not reach an optimal solution.")
        return

    result_df = extract_results(
        opt_df,
        r_vars,
        e_vars,
        s_vars,
        safety_stock_factor,
        risk_threshold,
    )

    save_results(result_df, output_path)
    print_summary(result_df)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "data" / "processed"
    output_file = project_root / "results" / "replenishment_optimization_soft_safety_stock_results.csv"

    run_service_level_optimization(
        base_dir=str(input_dir),
        output_path=str(output_file),
        service_level=0.90,
        safety_stock_factor=0.05,
        risk_threshold=0.35,
    )


if __name__ == "__main__":
    main()