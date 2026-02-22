#!/usr/bin/env python3
"""
GDP Nowcasting Demo - Fully Simulated
======================================

Standalone demo that runs with simulated data (no API keys required).
Requires: pandas, numpy, scikit-learn, matplotlib

Install dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Check dependencies
def check_dependencies():
    missing = []
    for pkg in ["pandas", "numpy", "sklearn", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "sklearn" else "scikit-learn")

    if missing:
        print("Missing required packages:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def generate_simulated_data(start_date="2010-01-01", n_quarters=56, seed=42):
    """Generate fully simulated GDP and indicator data."""
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_quarters, freq="QE")
    n = len(dates)

    # GDP: trend + cycles + noise
    trend = np.linspace(15000, 22000, n)
    business_cycle = 500 * np.sin(np.arange(n) * 2 * np.pi / 20)
    noise = np.random.normal(0, 100, n)

    # COVID shock
    covid_idx = 40  # Around Q1 2020
    shock = np.zeros(n)
    if covid_idx < n:
        shock[covid_idx] = -1500
        shock[covid_idx+1] = -800 if covid_idx+1 < n else 0
        shock[covid_idx+2] = 200 if covid_idx+2 < n else 0

    gdp = trend + business_cycle + noise + shock

    # Compute GDP growth as numpy array first, then assign
    gdp_growth = np.zeros(n)
    gdp_growth[0] = np.nan
    gdp_growth[1:] = (gdp[1:] - gdp[:-1]) / gdp[:-1] * 100

    # Port traffic (correlated with GDP)
    port_traffic = 100 + 0.003 * (gdp - 15000) + np.random.normal(0, 5, n)

    # Electricity consumption (correlated with GDP + seasonal)
    quarters = np.array([d.quarter for d in dates])
    seasonal = np.where(np.isin(quarters, [1, 3]), 50, -30)
    electricity = 1000 + 0.02 * (gdp - 15000) + seasonal + np.random.normal(0, 20, n)

    # Google Trends proxies (inversely correlated with GDP for unemployment)
    job_search = 50 - 0.001 * (gdp - 18000) + np.random.normal(0, 8, n)
    unemployment = 40 - 0.002 * (gdp - 18000) + np.random.normal(0, 10, n)
    hiring = 55 + 0.001 * (gdp - 18000) + np.random.normal(0, 6, n)

    df = pd.DataFrame({
        "gdp": gdp,
        "gdp_growth": gdp_growth,
        "port_traffic": port_traffic,
        "electricity_consumption": electricity,
        "job_search": job_search,
        "unemployment": unemployment,
        "hiring": hiring
    }, index=dates)
    df.index.name = "date"

    return df


def create_features(df, target_col="gdp_growth"):
    """Create lag and rolling features."""
    df = df.copy()

    indicator_cols = ["port_traffic", "electricity_consumption", "job_search", "unemployment", "hiring"]

    # Lag features
    for col in indicator_cols + [target_col]:
        if col in df.columns:
            for lag in [1, 2, 3, 4]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling features
    for col in indicator_cols:
        if col in df.columns:
            for window in [2, 4]:
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window).std()

    # Percent changes
    for col in indicator_cols:
        if col in df.columns:
            df[f"{col}_qoq_change"] = df[col].pct_change() * 100
            df[f"{col}_yoy_change"] = df[col].pct_change(4) * 100

    # Seasonal dummies
    df["quarter"] = df.index.quarter
    for q in [1, 2, 3, 4]:
        df[f"q{q}"] = (df["quarter"] == q).astype(int)

    # Interaction features
    df["port_elec_ratio"] = df["port_traffic"] / df["electricity_consumption"] * 100
    df["job_search_hiring_ratio"] = df["job_search"] / (df["hiring"] + 1)

    return df


def prepare_data(df, target_col="gdp_growth", test_size=8):
    """Prepare train/test split."""
    # Drop rows with NaN but keep enough data
    df = df.copy()

    # First, identify which rows have NaN in target
    valid_target = df[target_col].notna()
    df = df[valid_target]

    # For features, fill NaN with column means or forward fill
    feature_cols = [c for c in df.columns if c not in [target_col, "gdp", "quarter"]]

    # Forward fill then backward fill remaining NaNs
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # If still have NaN, fill with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train models and return results."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=5)
    rf_cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=tscv, scoring="neg_mean_squared_error")
    rf_cv_rmse = np.sqrt(-rf_cv_scores).mean()
    print(f"  CV RMSE: {rf_cv_rmse:.4f}")

    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    results["Random Forest"] = {
        "predictions": rf_pred,
        "cv_rmse": rf_cv_rmse,
        "importance": pd.DataFrame({
            "feature": X_train.columns,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
    }

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    gb_cv_scores = cross_val_score(gb, X_train_scaled, y_train, cv=tscv, scoring="neg_mean_squared_error")
    gb_cv_rmse = np.sqrt(-gb_cv_scores).mean()
    print(f"  CV RMSE: {gb_cv_rmse:.4f}")

    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)

    results["Gradient Boosting"] = {
        "predictions": gb_pred,
        "cv_rmse": gb_cv_rmse,
        "importance": pd.DataFrame({
            "feature": X_train.columns,
            "importance": gb.feature_importances_
        }).sort_values("importance", ascending=False)
    }

    return results


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def create_visualization(y_train, y_test, results, save_path="output/gdp_nowcast_demo.png"):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time series
    ax1 = axes[0, 0]
    all_dates = list(y_train.index) + list(y_test.index)
    all_actual = list(y_train.values) + list(y_test.values)

    ax1.plot(all_dates, all_actual, "b-", linewidth=2, label="Actual GDP Growth", marker="o", markersize=3)

    colors = {"Random Forest": "green", "Gradient Boosting": "red"}
    for name, res in results.items():
        ax1.plot(y_test.index, res["predictions"], color=colors[name],
                linestyle="--", linewidth=2, label=f"{name}", marker="s", markersize=4)

    ax1.axvline(x=y_test.index[0], color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("GDP Growth (%)")
    ax1.set_title("GDP Growth: Actual vs Predicted")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 2: Scatter
    ax2 = axes[0, 1]
    for name, res in results.items():
        ax2.scatter(y_test.values, res["predictions"], label=name, alpha=0.7, s=80)

    min_val = min(y_test.min(), min(r["predictions"].min() for r in results.values()))
    max_val = max(y_test.max(), max(r["predictions"].max() for r in results.values()))
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect")
    ax2.set_xlabel("Actual GDP Growth (%)")
    ax2.set_ylabel("Predicted GDP Growth (%)")
    ax2.set_title("Actual vs Predicted (Test Set)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = axes[1, 0]
    width = 0.35
    x = np.arange(len(y_test))
    for i, (name, res) in enumerate(results.items()):
        residuals = y_test.values - res["predictions"]
        ax3.bar(x + i*width, residuals, width, label=name, alpha=0.7)

    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Test Sample")
    ax3.set_ylabel("Residual")
    ax3.set_title("Prediction Residuals")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature importance
    ax4 = axes[1, 1]
    top_features = []
    for name, res in results.items():
        fi = res["importance"].head(10).copy()
        fi["model"] = name
        top_features.append(fi)

    combined = pd.concat(top_features)
    pivot = combined.pivot(index="feature", columns="model", values="importance").fillna(0)
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=True).drop(columns="avg").tail(10)
    pivot.plot(kind="barh", ax=ax4)
    ax4.set_xlabel("Importance")
    ax4.set_title("Top 10 Feature Importance")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")

    return fig


def main():
    print("="*60)
    print("GDP NOWCASTING DEMO (Fully Simulated Data)")
    print("="*60)

    # Generate data
    print("\n[1/5] Generating simulated economic data...")
    data = generate_simulated_data(n_quarters=56)
    print(f"  Generated {len(data)} quarters of data")
    print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")

    # Feature engineering
    print("\n[2/5] Engineering features...")
    data_features = create_features(data)
    print(f"  Created {len(data_features.columns)} features")

    # Prepare data
    print("\n[3/5] Preparing train/test split...")
    X_train, X_test, y_train, y_test, features = prepare_data(data_features, test_size=8)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(features)}")

    # Train models
    print("\n[4/5] Training models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Evaluate
    print("\n[5/5] Evaluating models...")
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\n{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'CV RMSE':>10}")
    print("-"*60)

    for name, res in results.items():
        metrics = compute_metrics(y_test.values, res["predictions"])
        print(f"{name:<20} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f} {metrics['R2']:>10.4f} {res['cv_rmse']:>10.4f}")

    # Top features
    print("\n" + "="*60)
    print("TOP 5 PREDICTIVE FEATURES")
    print("="*60)
    for name, res in results.items():
        print(f"\n{name}:")
        for _, row in res["importance"].head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Create visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    create_visualization(y_train, y_test, results)

    # Save outputs
    os.makedirs("output", exist_ok=True)

    # Save predictions
    pred_df = pd.DataFrame({"actual": y_test})
    for name, res in results.items():
        pred_df[name.lower().replace(" ", "_")] = res["predictions"]
    pred_df.to_csv("output/predictions.csv")
    print(f"Predictions saved to: output/predictions.csv")

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
