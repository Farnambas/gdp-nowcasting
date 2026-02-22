"""
Evaluation and visualization module for GDP nowcasting.
Computes metrics and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R2, and MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (handle zero values)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }


def evaluate_models(
    y_test: pd.Series,
    results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Evaluate multiple models and create comparison table.

    Args:
        y_test: Actual test values
        results: Results dictionary from train_models

    Returns:
        DataFrame with metrics for each model
    """
    metrics_list = []

    for model_name, model_results in results.items():
        predictions = model_results["predictions"]
        metrics = compute_metrics(y_test.values, predictions)
        metrics["Model"] = model_name.replace("_", " ").title()

        # Add CV scores
        cv_scores = model_results["cv_scores"]
        metrics["CV_RMSE"] = cv_scores["cv_rmse_mean"]
        metrics["CV_MAE"] = cv_scores["cv_mae_mean"]

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[["Model", "RMSE", "MAE", "R2", "MAPE", "CV_RMSE", "CV_MAE"]]

    return metrics_df


def plot_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted GDP growth.

    Args:
        y_train: Training target values
        y_test: Test target values
        results: Results dictionary with predictions
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time series of actual vs predictions
    ax1 = axes[0, 0]

    # Combine train and test indices
    all_dates = y_train.index.tolist() + y_test.index.tolist()
    all_actual = y_train.tolist() + y_test.tolist()

    ax1.plot(all_dates, all_actual, "b-", linewidth=2, label="Actual GDP Growth", marker="o", markersize=4)

    colors = {"random_forest": "green", "gradient_boosting": "red"}
    for model_name, model_results in results.items():
        predictions = model_results["predictions"]
        ax1.plot(
            y_test.index, predictions,
            color=colors.get(model_name, "gray"),
            linestyle="--", linewidth=2,
            label=f"{model_name.replace('_', ' ').title()} Predictions",
            marker="s", markersize=4
        )

    # Mark train/test split
    ax1.axvline(x=y_test.index[0], color="gray", linestyle=":", alpha=0.7)
    ax1.text(y_test.index[0], ax1.get_ylim()[1], "Test Period",
             ha="left", va="top", fontsize=9, color="gray")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("GDP Growth (%)")
    ax1.set_title("GDP Growth: Actual vs Predicted")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 2: Scatter plot of actual vs predicted (test set only)
    ax2 = axes[0, 1]

    for model_name, model_results in results.items():
        predictions = model_results["predictions"]
        ax2.scatter(
            y_test.values, predictions,
            label=model_name.replace("_", " ").title(),
            alpha=0.7, s=60
        )

    # Perfect prediction line
    min_val = min(y_test.min(), min(r["predictions"].min() for r in results.values()))
    max_val = max(y_test.max(), max(r["predictions"].max() for r in results.values()))
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect Prediction")

    ax2.set_xlabel("Actual GDP Growth (%)")
    ax2.set_ylabel("Predicted GDP Growth (%)")
    ax2.set_title("Actual vs Predicted (Test Set)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals over time
    ax3 = axes[1, 0]

    for model_name, model_results in results.items():
        predictions = model_results["predictions"]
        residuals = y_test.values - predictions
        ax3.bar(
            range(len(residuals)), residuals,
            alpha=0.6, label=model_name.replace("_", " ").title()
        )

    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Test Sample Index")
    ax3.set_ylabel("Residual (Actual - Predicted)")
    ax3.set_title("Prediction Residuals")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature importance comparison
    ax4 = axes[1, 1]

    importance_data = []
    for model_name, model_results in results.items():
        fi = model_results["feature_importance"].head(10).copy()
        fi["model"] = model_name.replace("_", " ").title()
        importance_data.append(fi)

    if importance_data:
        combined = pd.concat(importance_data)
        pivot = combined.pivot(index="feature", columns="model", values="importance")
        pivot = pivot.fillna(0)

        # Sort by average importance
        pivot["avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg", ascending=True).drop(columns="avg")

        pivot.plot(kind="barh", ax=ax4, width=0.8)
        ax4.set_xlabel("Importance")
        ax4.set_ylabel("Feature")
        ax4.set_title("Top 10 Feature Importance")
        ax4.legend(title="Model")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def print_evaluation_report(
    y_test: pd.Series,
    results: Dict[str, Dict],
    metrics_df: pd.DataFrame
) -> None:
    """
    Print a comprehensive evaluation report.

    Args:
        y_test: Test target values
        results: Results dictionary
        metrics_df: DataFrame with computed metrics
    """
    print("\n" + "="*60)
    print("GDP NOWCASTING MODEL EVALUATION REPORT")
    print("="*60)

    print(f"\nTest Period: {y_test.index.min()} to {y_test.index.max()}")
    print(f"Number of test samples: {len(y_test)}")

    print("\n" + "-"*60)
    print("MODEL PERFORMANCE METRICS")
    print("-"*60)
    print(metrics_df.to_string(index=False))

    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    best_rmse_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    best_mae_model = metrics_df.loc[metrics_df["MAE"].idxmin(), "Model"]
    best_r2_model = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]

    print(f"- Best RMSE: {best_rmse_model}")
    print(f"- Best MAE: {best_mae_model}")
    print(f"- Best R-squared: {best_r2_model}")

    print("\n" + "-"*60)
    print("TOP PREDICTIVE FEATURES")
    print("-"*60)

    for model_name, model_results in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        fi = model_results["feature_importance"].head(5)
        for _, row in fi.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    # Test evaluation
    import numpy as np

    # Create sample data
    dates = pd.date_range(start="2020-01-01", periods=8, freq="QE")
    y_test = pd.Series([2.1, -1.5, 3.2, 2.8, 1.9, 2.3, 2.7, 3.1], index=dates)

    results = {
        "random_forest": {
            "predictions": np.array([2.0, -1.2, 3.0, 2.5, 2.1, 2.5, 2.4, 2.9]),
            "cv_scores": {"cv_rmse_mean": 0.5, "cv_mae_mean": 0.4},
            "feature_importance": pd.DataFrame({
                "feature": ["lag1", "lag2", "trend", "seasonal"],
                "importance": [0.3, 0.25, 0.2, 0.15]
            })
        },
        "gradient_boosting": {
            "predictions": np.array([2.2, -1.3, 3.1, 2.6, 2.0, 2.4, 2.5, 3.0]),
            "cv_scores": {"cv_rmse_mean": 0.45, "cv_mae_mean": 0.35},
            "feature_importance": pd.DataFrame({
                "feature": ["lag1", "trend", "lag2", "seasonal"],
                "importance": [0.35, 0.22, 0.18, 0.12]
            })
        }
    }

    metrics_df = evaluate_models(y_test, results)
    print_evaluation_report(y_test, results, metrics_df)
