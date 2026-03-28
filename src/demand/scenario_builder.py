from pathlib import Path
import os

import pandas as pd
import numpy as np


def load_adjusted_summary(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Loaded adjusted summary:", df.shape)
    return df


def build_demand_scenarios(
    df: pd.DataFrame,
    low_factor_base: float = 0.90,
    high_factor_base: float = 1.10,
    censoring_impact: float = 0.20,
) -> pd.DataFrame:
    df = df.copy()

    if "adjusted_demand" not in df.columns:
        raise ValueError("adjusted_demand column is missing.")

    if "avg_censoring_score" not in df.columns:
        df["avg_censoring_score"] = 0.0

    uncertainty_scale = 1 + censoring_impact * df["avg_censoring_score"]

    df["demand_base"] = df["adjusted_demand"]
    df["demand_low"] = df["adjusted_demand"] * (low_factor_base / uncertainty_scale)
    df["demand_high"] = df["adjusted_demand"] * (high_factor_base * uncertainty_scale)

    df["demand_low"] = df["demand_low"].clip(lower=0)
    df["demand_base"] = df["demand_base"].clip(lower=0)
    df["demand_high"] = df["demand_high"].clip(lower=0)

    return df


def build_optimization_demand(
    df: pd.DataFrame,
    low_weight: float = 0.2,
    base_weight: float = 0.5,
    high_weight: float = 0.3,
) -> pd.DataFrame:
    df = df.copy()

    total_weight = low_weight + base_weight + high_weight
    if abs(total_weight - 1.0) > 1e-8:
        raise ValueError("Scenario weights must sum to 1.0")

    df["scenario_demand"] = (
        low_weight * df["demand_low"]
        + base_weight * df["demand_base"]
        + high_weight * df["demand_high"]
    )

    return df


def build_dynamic_safety_stock_features(
    df: pd.DataFrame,
    base_safety_factor: float = 0.05,
    spread_weight: float = 0.3,
    censoring_weight: float = 0.2,
    min_factor: float = 0.03,
    max_factor: float = 0.20,
) -> pd.DataFrame:
    df = df.copy()

    df["demand_spread"] = df["demand_high"] - df["demand_low"]
    df["uncertainty_ratio"] = df["demand_spread"] / df["demand_base"].replace(0, 1)

    df["dynamic_safety_factor"] = (
        base_safety_factor
        + spread_weight * df["uncertainty_ratio"]
        + censoring_weight * df["avg_censoring_score"] * 0.1
    )

    df["dynamic_safety_factor"] = df["dynamic_safety_factor"].clip(
        lower=min_factor,
        upper=max_factor,
    )

    return df


def save_outputs(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved scenario demand summary to: {output_path}")


def print_summary(df: pd.DataFrame):
    print("\n=== DEMAND SCENARIO SUMMARY ===")
    print("Total demand_low:", df["demand_low"].sum())
    print("Total demand_base:", df["demand_base"].sum())
    print("Total demand_high:", df["demand_high"].sum())
    print("Total scenario_demand:", df["scenario_demand"].sum())
    print("Average uncertainty_ratio:", df["uncertainty_ratio"].mean())
    print("Average dynamic_safety_factor:", df["dynamic_safety_factor"].mean())


def run_scenario_builder(input_path: str, output_path: str):
    df = load_adjusted_summary(input_path)
    df = build_demand_scenarios(df)
    df = build_optimization_demand(df)
    df = build_dynamic_safety_stock_features(df)

    save_outputs(df, output_path)
    print_summary(df)


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "adjusted_forecast_summary.csv"
    output_path = project_root / "data" / "processed" / "scenario_forecast_summary.csv"

    run_scenario_builder(
        input_path=str(input_path),
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()