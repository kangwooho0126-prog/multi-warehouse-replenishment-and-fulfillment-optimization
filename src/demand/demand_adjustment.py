from pathlib import Path
import os

import pandas as pd
import numpy as np


def load_censoring_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Loaded censoring data:", df.shape)
    return df


def adjust_demand(
    df: pd.DataFrame,
    base_uplift: float = 0.10,
    max_uplift: float = 0.35,
) -> pd.DataFrame:
    df = df.copy()

    uplift_factor = base_uplift + (max_uplift - base_uplift) * df["censoring_score"]

    df["adjusted_demand"] = np.where(
        df["is_censored"] == 1,
        df["predicted_sales"] * (1 + uplift_factor),
        df["predicted_sales"],
    )

    df["adjusted_demand"] = df["adjusted_demand"].clip(lower=0)

    return df


def build_adjusted_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["store_id", "item_id"], as_index=False)
        .agg(
            forecast_demand=("predicted_sales", "sum"),
            adjusted_demand=("adjusted_demand", "sum"),
            censored_days=("is_censored", "sum"),
            avg_censoring_score=("censoring_score", "mean"),
        )
    )

    print("Adjusted demand summary shape:", summary.shape)
    return summary


def save_outputs(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    detail_path: str,
    summary_path: str,
):
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved adjusted detail to: {detail_path}")
    print(f"Saved adjusted summary to: {summary_path}")


def print_summary(summary_df: pd.DataFrame):
    print("\n=== ADJUSTED DEMAND SUMMARY ===")
    print("Total forecast demand:", summary_df["forecast_demand"].sum())
    print("Total adjusted demand:", summary_df["adjusted_demand"].sum())
    print("Total censored days:", summary_df["censored_days"].sum())


def run_adjustment(input_path: str, detail_path: str, summary_path: str):
    df = load_censoring_data(input_path)
    df = adjust_demand(df)
    summary_df = build_adjusted_summary(df)

    save_outputs(df, summary_df, detail_path, summary_path)
    print_summary(summary_df)


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "results" / "forecast_with_censoring.csv"
    detail_path = project_root / "results" / "forecast_with_adjusted_demand.csv"
    summary_path = project_root / "data" / "processed" / "adjusted_forecast_summary.csv"

    run_adjustment(
        input_path=str(input_path),
        detail_path=str(detail_path),
        summary_path=str(summary_path),
    )


if __name__ == "__main__":
    main()