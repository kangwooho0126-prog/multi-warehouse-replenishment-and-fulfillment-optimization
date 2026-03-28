import os
from pathlib import Path

import pandas as pd
import pulp


def load_input_data(base_dir: str):
    forecast_path = os.path.join(base_dir, "forecast_summary.csv")
    warehouse_path = os.path.join(base_dir, "warehouse_info.csv")
    sku_path = os.path.join(base_dir, "sku_info.csv")
    inventory_path = os.path.join(base_dir, "initial_inventory.csv")

    forecast_df = pd.read_csv(forecast_path)
    warehouse_df = pd.read_csv(warehouse_path)
    sku_df = pd.read_csv(sku_path)
    inventory_df = pd.read_csv(inventory_path)

    print("Loaded forecast_df:", forecast_df.shape)
    print("Loaded warehouse_df:", warehouse_df.shape)
    print("Loaded sku_df:", sku_df.shape)
    print("Loaded inventory_df:", inventory_df.shape)

    return forecast_df, warehouse_df, sku_df, inventory_df


def prepare_optimization_table(
    forecast_df: pd.DataFrame,
    sku_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    df = forecast_df.merge(inventory_df, on=["store_id", "item_id"], how="left")
    df = df.merge(sku_df, on="item_id", how="left")
    df["initial_inventory"] = df["initial_inventory"].fillna(0)

    print("Optimization table shape:", df.shape)
    return df


def get_demand_value(row: pd.Series) -> float:
    if "scenario_demand" in row.index and pd.notna(row["scenario_demand"]):
        return row["scenario_demand"]
    if "adjusted_demand" in row.index and pd.notna(row["adjusted_demand"]):
        return row["adjusted_demand"]
    return row["forecast_demand"]


def build_replenishment_model(
    opt_df: pd.DataFrame,
    warehouse_df: pd.DataFrame,
):
    model = pulp.LpProblem("Replenishment_Optimization", pulp.LpMinimize)

    keys = list(opt_df[["store_id", "item_id"]].itertuples(index=False, name=None))

    replenishment_vars = {
        key: pulp.LpVariable(
            f"replenishment_{key[0]}_{key[1]}",
            lowBound=0,
            cat="Continuous",
        )
        for key in keys
    }

    ending_inventory_vars = {
        key: pulp.LpVariable(
            f"ending_inventory_{key[0]}_{key[1]}",
            lowBound=0,
            cat="Continuous",
        )
        for key in keys
    }

    shortage_vars = {
        key: pulp.LpVariable(
            f"shortage_{key[0]}_{key[1]}",
            lowBound=0,
            cat="Continuous",
        )
        for key in keys
    }

    model += pulp.lpSum(
        opt_df.loc[i, "replenishment_cost"]
        * replenishment_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        + opt_df.loc[i, "holding_cost"]
        * ending_inventory_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        + opt_df.loc[i, "stockout_cost"]
        * shortage_vars[(opt_df.loc[i, "store_id"], opt_df.loc[i, "item_id"])]
        for i in opt_df.index
    )

    for i in opt_df.index:
        store_id = opt_df.loc[i, "store_id"]
        item_id = opt_df.loc[i, "item_id"]
        key = (store_id, item_id)

        initial_inventory = opt_df.loc[i, "initial_inventory"]
        demand = get_demand_value(opt_df.loc[i])

        model += (
            initial_inventory
            + replenishment_vars[key]
            + shortage_vars[key]
            == demand + ending_inventory_vars[key]
        ), f"inventory_balance_{store_id}_{item_id}"

    for _, row in warehouse_df.iterrows():
        warehouse_id = row["warehouse_id"]
        capacity = row["capacity"]

        relevant_keys = [key for key in keys if key[0] == warehouse_id]

        model += (
            pulp.lpSum(ending_inventory_vars[key] for key in relevant_keys) <= capacity
        ), f"capacity_{warehouse_id}"

    return model, replenishment_vars, ending_inventory_vars, shortage_vars


def solve_model(model):
    solver = pulp.PULP_CBC_CMD(msg=True)
    model.solve(solver)

    print("Solver Status:", pulp.LpStatus[model.status])
    print("Objective Value:", pulp.value(model.objective))


def extract_results(
    opt_df: pd.DataFrame,
    replenishment_vars,
    ending_inventory_vars,
    shortage_vars,
) -> pd.DataFrame:
    result_df = opt_df.copy()

    result_df["optimization_demand"] = result_df.apply(get_demand_value, axis=1)

    result_df["replenishment_qty"] = result_df.apply(
        lambda row: replenishment_vars[(row["store_id"], row["item_id"])].varValue,
        axis=1,
    )

    result_df["ending_inventory"] = result_df.apply(
        lambda row: ending_inventory_vars[(row["store_id"], row["item_id"])].varValue,
        axis=1,
    )

    result_df["shortage_qty"] = result_df.apply(
        lambda row: shortage_vars[(row["store_id"], row["item_id"])].varValue,
        axis=1,
    )

    result_df["total_cost"] = (
        result_df["replenishment_cost"] * result_df["replenishment_qty"]
        + result_df["holding_cost"] * result_df["ending_inventory"]
        + result_df["stockout_cost"] * result_df["shortage_qty"]
    )

    return result_df


def save_results(result_df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Saved replenishment results to: {output_path}")
    print(result_df.head())


def print_summary(result_df: pd.DataFrame):
    print("\n=== REPLENISHMENT SUMMARY ===")
    print("Total optimization demand:", result_df["optimization_demand"].sum())
    print("Total replenishment qty:", result_df["replenishment_qty"].sum())
    print("Total ending inventory:", result_df["ending_inventory"].sum())
    print("Total shortage qty:", result_df["shortage_qty"].sum())
    print("Total cost:", result_df["total_cost"].sum())

    warehouse_summary = (
        result_df.groupby("store_id", as_index=False)
        .agg(
            {
                "optimization_demand": "sum",
                "replenishment_qty": "sum",
                "ending_inventory": "sum",
                "shortage_qty": "sum",
                "total_cost": "sum",
            }
        )
    )

    print("\n=== WAREHOUSE SUMMARY ===")
    print(warehouse_summary)


def run_replenishment(base_dir: str, output_path: str):
    forecast_df, warehouse_df, sku_df, inventory_df = load_input_data(base_dir)
    opt_df = prepare_optimization_table(forecast_df, sku_df, inventory_df)

    model, replenishment_vars, ending_inventory_vars, shortage_vars = (
        build_replenishment_model(opt_df, warehouse_df)
    )

    solve_model(model)

    result_df = extract_results(
        opt_df,
        replenishment_vars,
        ending_inventory_vars,
        shortage_vars,
    )
    save_results(result_df, output_path)
    print_summary(result_df)


def main():
    project_root = Path(__file__).resolve().parents[2]

    base_dir = project_root / "data" / "processed"
    output_path = project_root / "results" / "replenishment_optimization_results.csv"

    run_replenishment(
        base_dir=str(base_dir),
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()