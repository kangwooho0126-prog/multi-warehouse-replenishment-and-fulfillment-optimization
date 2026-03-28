import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NO_SERVICE_LABEL = "No_Service_Level"
SERVICE_LABEL = "Service_Level_90%"


def ensure_cost_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "replenishment_cost_component" not in df.columns:
        df["replenishment_cost_component"] = df["replenishment_cost"] * df["replenishment_qty"]

    if "holding_cost_component" not in df.columns:
        df["holding_cost_component"] = df["holding_cost"] * df["ending_inventory"]

    if "stockout_cost_component" not in df.columns:
        df["stockout_cost_component"] = df["stockout_cost"] * df["shortage_qty"]

    if "safety_penalty_cost" not in df.columns:
        df["safety_penalty_cost"] = 0.0

    if "total_cost" not in df.columns:
        df["total_cost"] = (
            df["replenishment_cost_component"]
            + df["holding_cost_component"]
            + df["stockout_cost_component"]
            + df["safety_penalty_cost"]
        )

    return df


def load_results(base_dir: str):
    no_sl_df = pd.read_csv(os.path.join(base_dir, "replenishment_optimization_results.csv"))
    sl_df = pd.read_csv(os.path.join(base_dir, "replenishment_optimization_soft_safety_stock_results.csv"))
    return ensure_cost_components(no_sl_df), ensure_cost_components(sl_df)


def summarize_one(df: pd.DataFrame, scenario_label: str) -> dict:
    demand_col = "optimization_demand" if "optimization_demand" in df.columns else "forecast_demand"

    total_demand = df[demand_col].sum()
    total_ending_inventory = df["ending_inventory"].sum()
    total_shortage = df["shortage_qty"].sum()

    avg_daily_demand_proxy = total_demand / 28 if total_demand > 0 else 1.0
    inventory_days_proxy = total_ending_inventory / avg_daily_demand_proxy if avg_daily_demand_proxy > 0 else 0.0
    inventory_turnover_proxy = total_demand / max(total_ending_inventory, 1e-6)

    return {
        "scenario": scenario_label,
        "total_demand": total_demand,
        "total_replenishment_qty": df["replenishment_qty"].sum(),
        "total_ending_inventory": total_ending_inventory,
        "total_shortage_qty": total_shortage,
        "average_fill_rate": 1 - (total_shortage / max(total_demand, 1e-6)),
        "total_cost": df["total_cost"].sum(),
        "replenishment_cost_component": df["replenishment_cost_component"].sum(),
        "holding_cost_component": df["holding_cost_component"].sum(),
        "stockout_cost_component": df["stockout_cost_component"].sum(),
        "safety_penalty_cost": df["safety_penalty_cost"].sum(),
        "inventory_turnover_proxy": inventory_turnover_proxy,
        "inventory_days_proxy": inventory_days_proxy,
    }


def build_overall_summary(no_sl_df: pd.DataFrame, sl_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = pd.DataFrame([
        summarize_one(no_sl_df, NO_SERVICE_LABEL),
        summarize_one(sl_df, SERVICE_LABEL),
    ])

    baseline = summary_df.iloc[0]
    summary_df["cost_change_pct_vs_baseline"] = (
        (summary_df["total_cost"] - baseline["total_cost"]) / max(baseline["total_cost"], 1e-6)
    ) * 100

    if baseline["total_ending_inventory"] <= 1e-6:
         summary_df["inventory_change_pct_vs_baseline"] = np.where(
             summary_df["scenario"] == baseline["scenario"],
             0.0,
         np.nan
        )
    else:
        summary_df["inventory_change_pct_vs_baseline"] = (
           (summary_df["total_ending_inventory"] - baseline["total_ending_inventory"])
           / baseline["total_ending_inventory"]
    ) * 100

    summary_df["shortage_change_pct_vs_baseline"] = (
        (summary_df["total_shortage_qty"] - baseline["total_shortage_qty"])
        / max(max(baseline["total_shortage_qty"], 1e-6), 1e-6)
    ) * 100

    summary_df["fill_rate_change_vs_baseline"] = (
        summary_df["average_fill_rate"] - baseline["average_fill_rate"]
    )

    return summary_df


def build_warehouse_summary(no_sl_df: pd.DataFrame, sl_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for label, df in [(NO_SERVICE_LABEL, no_sl_df), (SERVICE_LABEL, sl_df)]:
        demand_col = "optimization_demand" if "optimization_demand" in df.columns else "forecast_demand"

        warehouse_summary = df.groupby("store_id", as_index=False).agg(
            replenishment_qty=("replenishment_qty", "sum"),
            ending_inventory=("ending_inventory", "sum"),
            shortage_qty=("shortage_qty", "sum"),
            optimization_demand=(demand_col, "sum"),
            total_cost=("total_cost", "sum"),
            replenishment_cost_component=("replenishment_cost_component", "sum"),
            holding_cost_component=("holding_cost_component", "sum"),
            stockout_cost_component=("stockout_cost_component", "sum"),
            safety_penalty_cost=("safety_penalty_cost", "sum"),
        )

        warehouse_summary["fill_rate"] = 1 - (
            warehouse_summary["shortage_qty"] / warehouse_summary["optimization_demand"].replace(0, 1)
        )
        warehouse_summary["scenario"] = label
        frames.append(warehouse_summary)

    return pd.concat(frames, ignore_index=True)


def save_summary(summary_df: pd.DataFrame, warehouse_summary_df: pd.DataFrame, results_dir: str) -> None:
    summary_path = os.path.join(results_dir, "optimization_comparison_summary.csv")
    warehouse_path = os.path.join(results_dir, "warehouse_comparison_summary.csv")

    summary_df.to_csv(summary_path, index=False)
    warehouse_summary_df.to_csv(warehouse_path, index=False)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved warehouse summary to: {warehouse_path}")


def make_cost_shortage_plot(summary_df: pd.DataFrame, output_path: str) -> None:
    plot_df = summary_df[["scenario", "total_shortage_qty", "total_cost"]].copy()
    ax = plot_df.set_index("scenario").plot(kind="bar", rot=0)
    ax.set_ylabel("Value")
    ax.set_title("Optimization trade-off: shortage vs cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {output_path}")


def make_fill_rate_plot(warehouse_summary_df: pd.DataFrame, output_path: str) -> None:
    pivot_df = warehouse_summary_df.pivot(index="store_id", columns="scenario", values="fill_rate")
    ax = pivot_df.plot(kind="bar", rot=0)
    ax.set_ylabel("Fill rate")
    ax.set_title("Warehouse-level fill rate comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {output_path}")


def run_comparison(results_dir: str) -> None:
    no_sl_df, sl_df = load_results(results_dir)
    summary_df = build_overall_summary(no_sl_df, sl_df)
    warehouse_summary_df = build_warehouse_summary(no_sl_df, sl_df)

    save_summary(summary_df, warehouse_summary_df, results_dir)

    make_cost_shortage_plot(
        summary_df,
        os.path.join(results_dir, "optimization_cost_shortage_comparison.png"),
    )
    make_fill_rate_plot(
        warehouse_summary_df,
        os.path.join(results_dir, "warehouse_fill_rate_comparison.png"),
    )

    print("\n=== COMPARISON SUMMARY ===")
    print(summary_df[[
        "scenario",
        "total_demand",
        "total_replenishment_qty",
        "total_ending_inventory",
        "total_shortage_qty",
        "average_fill_rate",
        "total_cost",
        "cost_change_pct_vs_baseline",
        "inventory_change_pct_vs_baseline",
        "fill_rate_change_vs_baseline",
    ]])


def main():
    project_root = Path(__file__).resolve().parents[2]
    run_comparison(str(project_root / "results"))


if __name__ == "__main__":
    main()