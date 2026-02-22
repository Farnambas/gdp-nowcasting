"""
GDP Nowcasting Pipeline
=======================

This script runs the complete GDP nowcasting pipeline:
1. Collect data from FRED API, Google Trends, and simulated sources
2. Engineer features with lags and rolling statistics
3. Train Random Forest and Gradient Boosting models
4. Evaluate models and visualize results

Usage:
    python main.py --fred-api-key YOUR_API_KEY

Or set the FRED_API_KEY environment variable.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import collect_all_data
from src.feature_engineering import engineer_features, prepare_model_data
from src.models import train_models
from src.evaluation import evaluate_models, plot_predictions, print_evaluation_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GDP Nowcasting Pipeline")

    parser.add_argument(
        "--fred-api-key",
        type=str,
        default=None,
        help="FRED API key (or set FRED_API_KEY env var)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date for data collection (default: 2010-01-01)"
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=8,
        help="Number of quarters to hold out for testing (default: 8)"
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)"
    )

    parser.add_argument(
        "--use-simulated-trends",
        action="store_true",
        help="Use simulated Google Trends data instead of fetching"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save outputs (default: output)"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )

    return parser.parse_args()


def main():
    """Run the GDP nowcasting pipeline."""
    args = parse_args()

    # Get API key
    fred_api_key = args.fred_api_key or os.getenv("FRED_API_KEY")
    if not fred_api_key:
        print("Error: FRED API key required.")
        print("Provide via --fred-api-key argument or FRED_API_KEY environment variable.")
        print("\nGet a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("GDP NOWCASTING PIPELINE")
    print("="*60)
    print(f"Start date: {args.start_date}")
    print(f"Test size: {args.test_size} quarters")
    print(f"CV splits: {args.cv_splits}")
    print(f"Output directory: {args.output_dir}")

    # Step 1: Collect data
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)

    try:
        data = collect_all_data(
            fred_api_key=fred_api_key,
            start_date=args.start_date,
            use_simulated_trends=args.use_simulated_trends
        )
    except Exception as e:
        print(f"Error collecting data: {e}")
        sys.exit(1)

    print("\nRaw data preview:")
    print(data.head())

    # Save raw data
    raw_data_path = os.path.join(args.output_dir, "raw_data.csv")
    data.to_csv(raw_data_path)
    print(f"\nRaw data saved to {raw_data_path}")

    # Step 2: Feature engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)

    data_features = engineer_features(data)

    print("\nFeature data preview:")
    print(data_features.head())

    # Save featured data
    features_path = os.path.join(args.output_dir, "featured_data.csv")
    data_features.to_csv(features_path)
    print(f"\nFeatured data saved to {features_path}")

    # Step 3: Prepare model data
    print("\n" + "="*60)
    print("STEP 3: PREPARE MODEL DATA")
    print("="*60)

    try:
        X_train, X_test, y_train, y_test, feature_names = prepare_model_data(
            data_features,
            target_col="gdp_growth",
            test_size=args.test_size
        )
    except ValueError as e:
        print(f"Error preparing data: {e}")
        print("Try using a later start date or smaller test size.")
        sys.exit(1)

    # Step 4: Train models
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)

    models, results = train_models(
        X_train, y_train,
        X_test, y_test,
        cv_splits=args.cv_splits
    )

    # Step 5: Evaluate models
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)

    metrics_df = evaluate_models(y_test, results)
    print_evaluation_report(y_test, results, metrics_df)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Save predictions
    predictions_df = y_test.to_frame(name="actual")
    for model_name, model_results in results.items():
        predictions_df[f"{model_name}_predicted"] = model_results["predictions"]
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path)
    print(f"Predictions saved to {predictions_path}")

    # Step 6: Visualization
    if not args.no_plot:
        print("\n" + "="*60)
        print("STEP 6: VISUALIZATION")
        print("="*60)

        plot_path = os.path.join(args.output_dir, "gdp_nowcast_results.png")
        fig = plot_predictions(y_train, y_test, results, save_path=plot_path)

        # Try to show the plot (may not work in all environments)
        try:
            import matplotlib
            if matplotlib.get_backend() != "agg":
                print("Displaying plot...")
                import matplotlib.pyplot as plt
                plt.show()
        except Exception:
            pass

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("Files created:")
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
