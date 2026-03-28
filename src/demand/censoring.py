from pathlib import Path
import os

import pandas as pd


PREDICTION_COL = "predicted_sales"
ACTUAL_COL = "actual_sales"


def load_prediction_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print("Loaded prediction data:", df.shape)
    return df


def compute_censoring_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gap"] = df[PREDICTION_COL] - df[ACTUAL_COL]
    df["relative_gap"] = df["gap"] / df[PREDICTION_COL].replace(0, 1)
    df["low_sales_flag"] = (df[ACTUAL_COL] <= df[ACTUAL_COL].quantile(0.30)).astype(int)
    df["under_realized_flag"] = (df[PREDICTION_COL] > df[ACTUAL_COL]).astype(int)
    return df


def identify_censored_demand(df: pd.DataFrame, gap_quantile: float = 0.75, relative_gap_threshold: float = 0.30) -> pd.DataFrame:
    df = df.copy()
    gap_threshold = df["gap"].quantile(gap_quantile)

    df["is_censored"] = (
        (df["gap"] >= gap_threshold)
        & (df["relative_gap"] >= relative_gap_threshold)
        & (df["under_realized_flag"] == 1)
        & (df["low_sales_flag"] == 1)
    ).astype(int)

    df["censoring_score"] = (
        0.5 * df["relative_gap"].clip(lower=0)
        + 0.3 * (df["gap"] / (gap_threshold if gap_threshold != 0 else 1)).clip(lower=0)
        + 0.2 * df["low_sales_flag"]
    ).clip(0, 1)
    return df


def save_censoring_results(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved censoring results to: {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    censored = int(df["is_censored"].sum())
    ratio = censored / total if total > 0 else 0.0
    print("\n=== CENSORING SUMMARY ===")
    print("Total records:", total)
    print("Censored records:", censored)
    print(f"Censored ratio: {ratio:.4f}")


def run_censoring(input_path: str, output_path: str) -> None:
    df = load_prediction_data(input_path)
    df = compute_censoring_features(df)
    df = identify_censored_demand(df)
    save_censoring_results(df, output_path)
    print_summary(df)


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "results" / "forecast_test_predictions_enhanced.csv"
    output_path = project_root / "results" / "forecast_with_censoring.csv"
    run_censoring(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
