"""
Feature engineering module for GDP nowcasting.
Creates lag features, rolling statistics, and transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 4]
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods (e.g., [1, 2, 3, 4] for Q-1 to Q-4)

    Returns:
        DataFrame with additional lag columns
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [2, 4]
) -> pd.DataFrame:
    """
    Create rolling mean and std features.

    Args:
        df: Input DataFrame
        columns: Columns to create rolling features for
        windows: Window sizes in periods

    Returns:
        DataFrame with rolling features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window).std()

    return df


def create_pct_change_features(
    df: pd.DataFrame,
    columns: List[str],
    periods: List[int] = [1, 4]
) -> pd.DataFrame:
    """
    Create percentage change features.

    Args:
        df: Input DataFrame
        columns: Columns to compute pct change for
        periods: Periods for pct change (1 = QoQ, 4 = YoY)

    Returns:
        DataFrame with pct change features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        for period in periods:
            suffix = "qoq" if period == 1 else f"yoy" if period == 4 else f"p{period}"
            df[f"{col}_{suffix}_change"] = df[col].pct_change(periods=period) * 100

    return df


def create_momentum_features(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Create momentum indicators (difference from moving average).

    Args:
        df: Input DataFrame
        columns: Columns to compute momentum for

    Returns:
        DataFrame with momentum features
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        ma4 = df[col].rolling(window=4).mean()
        df[f"{col}_momentum"] = df[col] - ma4

    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal dummy variables from the date index.

    Args:
        df: Input DataFrame with DatetimeIndex

    Returns:
        DataFrame with quarter dummy columns
    """
    df = df.copy()

    df["quarter"] = df.index.quarter
    df["q1"] = (df["quarter"] == 1).astype(int)
    df["q2"] = (df["quarter"] == 2).astype(int)
    df["q3"] = (df["quarter"] == 3).astype(int)
    df["q4"] = (df["quarter"] == 4).astype(int)

    return df


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "gdp_growth",
    indicator_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Args:
        df: Input DataFrame with GDP and indicator data
        target_col: Name of the target variable column
        indicator_cols: Columns to use as economic indicators

    Returns:
        DataFrame with engineered features
    """
    if indicator_cols is None:
        # Default indicator columns
        indicator_cols = [
            "port_traffic",
            "electricity_consumption",
            "job_search",
            "unemployment",
            "hiring"
        ]

    # Filter to columns that exist
    indicator_cols = [c for c in indicator_cols if c in df.columns]

    print(f"Engineering features for indicators: {indicator_cols}")

    # Create lag features for indicators (key for nowcasting)
    df = create_lag_features(df, indicator_cols, lags=[1, 2, 3, 4])

    # Create lag features for target (autoregressive component)
    df = create_lag_features(df, [target_col], lags=[1, 2, 4])

    # Create rolling statistics
    df = create_rolling_features(df, indicator_cols, windows=[2, 4])

    # Create percent change features
    df = create_pct_change_features(df, indicator_cols, periods=[1, 4])

    # Create momentum features
    df = create_momentum_features(df, indicator_cols)

    # Create seasonal dummies
    df = create_seasonal_features(df)

    # Create interaction features between key indicators
    if "port_traffic" in df.columns and "electricity_consumption" in df.columns:
        df["port_elec_ratio"] = df["port_traffic"] / df["electricity_consumption"] * 100

    if "job_search" in df.columns and "hiring" in df.columns:
        df["job_search_hiring_ratio"] = df["job_search"] / (df["hiring"] + 1)

    print(f"Total features created: {len(df.columns)} columns")

    return df


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = "gdp_growth",
    test_size: int = 8,
    drop_na: bool = True
) -> tuple:
    """
    Prepare data for model training.

    Args:
        df: DataFrame with features
        target_col: Target variable name
        test_size: Number of periods to hold out for testing
        drop_na: Whether to drop rows with NaN values

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    df = df.copy()

    # First, keep only rows where target is valid
    df = df[df[target_col].notna()]

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in [target_col, "gdp", "quarter"]]

    # Forward fill then backward fill NaN in features
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Fill any remaining NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    if len(df) == 0:
        raise ValueError("No data remaining after dropping NaN values")

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in [target_col, "gdp", "quarter"]]
    X = df[feature_cols]
    y = df[target_col]

    # Time series split (don't shuffle!)
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    print(f"Training set: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Test set: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")
    print(f"Number of features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    # Test feature engineering
    import numpy as np

    # Create sample data
    dates = pd.date_range(start="2015-01-01", periods=40, freq="QE")
    np.random.seed(42)

    df = pd.DataFrame({
        "gdp": np.linspace(18000, 22000, 40) + np.random.normal(0, 100, 40),
        "gdp_growth": np.random.normal(2, 1, 40),
        "port_traffic": 100 + np.random.normal(0, 10, 40),
        "electricity_consumption": 1000 + np.random.normal(0, 50, 40),
        "job_search": 50 + np.random.normal(0, 10, 40),
        "unemployment": 40 + np.random.normal(0, 8, 40),
        "hiring": 55 + np.random.normal(0, 7, 40),
    }, index=dates)

    print("Original data shape:", df.shape)
    print("\nEngineering features...")

    df_features = engineer_features(df)
    print("\nFeatured data shape:", df_features.shape)

    print("\nPreparing model data...")
    X_train, X_test, y_train, y_test, features = prepare_model_data(df_features)

    print("\nSample features:")
    print(X_train.columns.tolist()[:20])
