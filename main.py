from src.data.preprocessing import process_in_chunks
from src.features.feature_engineering import build_forecasting_features
from src.forecasting.train import main as train_main
from src.forecasting.predict import main as predict_main
from src.demand.censoring import main as censoring_main
from src.demand.demand_adjustment import main as adjustment_main
from src.demand.scenario_builder import main as scenario_main
from src.optimization.prepare_inputs import main as prepare_inputs_main
from src.optimization.replenishment import main as replenishment_main
from src.optimization.service_level import main as service_level_main
from src.optimization.compare import main as compare_main
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print("Step 1: Preprocess raw sales data...")
    process_in_chunks(
        raw_dir=str(raw_dir),
        output_path=str(processed_dir / "daily_sales_chunk_subset.csv"),
    )

    print("Step 2: Build forecasting features...")
    build_forecasting_features(
        input_path=str(processed_dir / "daily_sales_chunk_subset.csv"),
        output_path=str(processed_dir / "forecasting_features.csv"),
    )

    print("Step 3: Train baseline forecasting model...")
    train_main()

    print("Step 4: Generate enhanced forecasting predictions...")
    predict_main()

    print("Step 5: Identify potentially censored demand...")
    censoring_main()

    print("Step 6: Build adjusted demand...")
    adjustment_main()

    print("Step 7: Build demand scenarios...")
    scenario_main()

    print("Step 8: Prepare optimization inputs...")
    prepare_inputs_main()

    print("Step 9: Run replenishment optimization...")
    replenishment_main()

    print("Step 10: Run service-level optimization...")
    service_level_main()

    print("Step 11: Compare optimization results...")
    compare_main()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
