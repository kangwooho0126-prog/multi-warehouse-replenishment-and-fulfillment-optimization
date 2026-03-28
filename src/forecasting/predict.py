from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.model import build_lightgbm_model


GROUP_COLS = ["store_id", "item_id"]


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["date"])
    print("Loaded feature data:", df.shape)
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    df["lag_1"] = df.groupby(GROUP_COLS)["sales_qty"].shift(1)
    df["lag_3"] = df.groupby(GROUP_COLS)["sales_qty"].shift(3)
    df["rolling_std_7"] = (
        df.groupby(GROUP_COLS)["sales_qty"]
        .transform(lambda x: x.shift(1).rolling(window=7).std())
    )
    df["rolling_std_28"] = (
        df.groupby(GROUP_COLS)["sales_qty"]
        .transform(lambda x: x.shift(1).rolling(window=28).std())
    )
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if "sell_price" in df.columns:
        df["sell_price"] = (
            df.groupby(GROUP_COLS)["sell_price"]
            .transform(lambda x: x.ffill().bfill())
        )

    df = add_extra_features(df)
    required_cols = [
        "lag_1",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "price_lag_7",
    ]
    before_shape = df.shape
    df = df.dropna(subset=[col for col in required_cols if col in df.columns]).reset_index(drop=True)
    after_shape = df.shape
    print("Before dropna in enhanced model:", before_shape)
    print("After dropna in enhanced model:", after_shape)
    return df


def prepare_features(df: pd.DataFrame):
    candidate_features = [
        "sell_price",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "lag_1",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "price_lag_7",
        "price_change_7",
    ]
    feature_cols = [col for col in candidate_features if col in df.columns]
    target_col = "sales_qty"
    df = df.sort_values("date").reset_index(drop=True)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return df, X, y, feature_cols, target_col


def time_based_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.70)
    valid_end = int(n_dates * 0.90)

    train_dates = unique_dates[:train_end]
    valid_dates = unique_dates[train_end:valid_end]
    test_dates = unique_dates[valid_end:]

    train_mask = df["date"].isin(train_dates)
    valid_mask = df["date"].isin(valid_dates)
    test_mask = df["date"].isin(test_dates)

    return (
        X[train_mask],
        y[train_mask],
        X[valid_mask],
        y[valid_mask],
        X[test_mask],
        y[test_mask],
        df[train_mask].copy(),
        df[valid_mask].copy(),
        df[test_mask].copy(),
    )


def evaluate_model(y_true, y_pred, dataset_name="dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{dataset_name} MAE: {mae:.4f}")
    print(f"{dataset_name} RMSE: {rmse:.4f}")
    return mae, rmse


def evaluate_baselines(df_valid: pd.DataFrame, df_test: pd.DataFrame):
    baseline_results = []
    for name, pred_col, subset_name, subset_df in [
        ("lag_7_baseline", "lag_7", "Validation", df_valid),
        ("rolling_mean_7_baseline", "rolling_mean_7", "Validation", df_valid),
        ("lag_7_baseline", "lag_7", "Test", df_test),
        ("rolling_mean_7_baseline", "rolling_mean_7", "Test", df_test),
    ]:
        y_true = subset_df["sales_qty"]
        y_pred = subset_df[pred_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        baseline_results.append({"dataset": subset_name, "model": name, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(baseline_results)


def train_lightgbm(X_train, y_train, X_valid, y_valid):
    model = build_lightgbm_model(random_state=42)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
    )
    return model


def save_predictions(df_test: pd.DataFrame, y_test, y_pred, output_path: str):
    result_df = df_test.copy()
    result_df["actual_sales"] = y_test.values
    result_df["predicted_sales"] = y_pred
    result_df["abs_error"] = (result_df["actual_sales"] - result_df["predicted_sales"]).abs()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


def save_feature_importance(model, feature_cols, output_path: str):
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"Saved feature importance to: {output_path}")


def save_metrics(baseline_df: pd.DataFrame, lgb_valid_metrics: dict, lgb_test_metrics: dict, output_path: str):
    lgb_rows = pd.DataFrame([
        {"dataset": "Validation", "model": "LightGBM_enhanced", "MAE": lgb_valid_metrics["MAE"], "RMSE": lgb_valid_metrics["RMSE"]},
        {"dataset": "Test", "model": "LightGBM_enhanced", "MAE": lgb_test_metrics["MAE"], "RMSE": lgb_test_metrics["RMSE"]},
    ])
    metrics_df = pd.concat([baseline_df, lgb_rows], ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved metrics comparison to: {output_path}")


def run_prediction(input_path: str, pred_output_path: str, importance_output_path: str, metrics_output_path: str):
    df = load_data(input_path)
    df = preprocess_data(df)
    df, X, y, feature_cols, _ = prepare_features(df)
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        _,
        df_valid,
        df_test,
    ) = time_based_split(df, X, y)

    baseline_df = evaluate_baselines(df_valid, df_test)
    model = train_lightgbm(X_train, y_train, X_valid, y_valid)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_mae, valid_rmse = evaluate_model(y_valid, valid_pred, dataset_name="Validation LightGBM_enhanced")
    test_mae, test_rmse = evaluate_model(y_test, test_pred, dataset_name="Test LightGBM_enhanced")

    save_predictions(df_test, y_test, test_pred, pred_output_path)
    save_feature_importance(model, feature_cols, importance_output_path)
    save_metrics(
        baseline_df=baseline_df,
        lgb_valid_metrics={"MAE": valid_mae, "RMSE": valid_rmse},
        lgb_test_metrics={"MAE": test_mae, "RMSE": test_rmse},
        output_path=metrics_output_path,
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "forecasting_features.csv"
    pred_output_path = project_root / "results" / "forecast_test_predictions_enhanced.csv"
    importance_output_path = project_root / "results" / "feature_importance_enhanced.csv"
    metrics_output_path = project_root / "results" / "forecast_metrics_comparison.csv"

    run_prediction(
        input_path=str(input_path),
        pred_output_path=str(pred_output_path),
        importance_output_path=str(importance_output_path),
        metrics_output_path=str(metrics_output_path),
    )


if __name__ == "__main__":
    main()
