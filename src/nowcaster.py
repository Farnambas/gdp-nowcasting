"""
GDP Nowcaster - Production-Ready Model
=======================================

This module contains the complete GDP nowcasting system:
- Data collection from FRED
- Feature engineering
- Model training and saving
- Real-time nowcasting

Usage:
    from src.nowcaster import GDPNowcaster

    # Train and save model
    nowcaster = GDPNowcaster(api_key="your_fred_api_key")
    nowcaster.train()
    nowcaster.save("models/gdp_nowcaster.pkl")

    # Load and predict
    nowcaster = GDPNowcaster.load("models/gdp_nowcaster.pkl")
    prediction = nowcaster.nowcast()
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from fredapi import Fred
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class GDPNowcaster:
    """
    Production-ready GDP Nowcasting Model.

    Uses 30+ economic indicators from FRED to predict GDP growth
    before official releases.
    """

    # Comprehensive list of FRED indicators
    INDICATORS = {
        # === GDP (Target) ===
        'GDPC1': 'Real GDP',

        # === Trade & International (Port Traffic Proxies) ===
        'IMPGS': 'Imports of Goods & Services',
        'EXPGS': 'Exports of Goods & Services',
        'BOPGSTB': 'Trade Balance',
        'IMPGSC1': 'Real Imports of Goods & Services',
        'EXPGSC1': 'Real Exports of Goods & Services',

        # === Industrial Production & Manufacturing ===
        'INDPRO': 'Industrial Production Index',
        'TCU': 'Capacity Utilization',
        'DGORDER': 'Durable Goods Orders',
        'NEWORDER': 'Manufacturers New Orders',
        'AMTMNO': 'Total Manufacturing Orders',
        'IPMAN': 'Industrial Production: Manufacturing',
        'BUSINV': 'Total Business Inventories',

        # === Labor Market ===
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Total Nonfarm Payrolls',
        'ICSA': 'Initial Jobless Claims',
        'CCSA': 'Continued Jobless Claims',
        'JTSJOL': 'Job Openings (JOLTS)',
        'AWHMAN': 'Avg Weekly Hours: Manufacturing',
        'CES0500000003': 'Avg Hourly Earnings',

        # === Consumer Spending & Sentiment ===
        'PCE': 'Personal Consumption Expenditure',
        'PCEC96': 'Real PCE',
        'RSAFS': 'Retail Sales',
        'RRSFS': 'Real Retail Sales',
        'UMCSENT': 'Consumer Sentiment (U of Michigan)',
        'CSCICP03USM665S': 'Consumer Confidence',
        'DSPIC96': 'Real Disposable Personal Income',

        # === Housing Market ===
        'HOUST': 'Housing Starts',
        'PERMIT': 'Building Permits',
        'HSN1F': 'New Home Sales',
        'EXHOSLUSM495S': 'Existing Home Sales',
        'MSPUS': 'Median Home Price',

        # === Financial Conditions ===
        'FEDFUNDS': 'Federal Funds Rate',
        'T10Y2Y': 'Yield Curve (10Y-2Y)',
        'T10Y3M': 'Yield Curve (10Y-3M)',
        'BAA10Y': 'Corporate Bond Spread',
        'SP500': 'S&P 500 Index',
        'VIXCLS': 'VIX Volatility Index',
        'DTWEXBGS': 'Trade Weighted Dollar Index',

        # === Prices & Inflation ===
        'CPIAUCSL': 'CPI All Items',
        'CPILFESL': 'Core CPI',
        'PPIACO': 'Producer Price Index',
        'PCEPI': 'PCE Price Index',

        # === Credit & Money ===
        'TOTCI': 'Commercial & Industrial Loans',
        'CONSUMER': 'Consumer Loans',
        'M2SL': 'M2 Money Supply',

        # === Leading Indicators ===
        'USSLIND': 'Leading Index',
        'CFNAI': 'Chicago Fed National Activity Index',
        'UMCSENT': 'Consumer Sentiment',
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the nowcaster.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key required")

        self.fred = Fred(api_key=self.api_key)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.indicators_loaded = []
        self.last_training_date = None
        self.training_metrics = {}

    def fetch_data(self, start_date: str = '2000-01-01') -> pd.DataFrame:
        """
        Fetch all available indicators from FRED.

        Args:
            start_date: Start date for data collection

        Returns:
            DataFrame with all indicators
        """
        print("Fetching data from FRED...")
        df = pd.DataFrame()
        self.indicators_loaded = []

        for code, name in self.INDICATORS.items():
            try:
                series = self.fred.get_series(code, observation_start=start_date)
                series.index = pd.to_datetime(series.index)

                # Resample to quarterly
                if len(series) > 150:  # Monthly or higher frequency
                    quarterly = series.resample('QE').mean()
                else:
                    quarterly = series.resample('QE').last()

                df[code] = quarterly
                self.indicators_loaded.append(code)
                print(f"  ✓ {code}")
            except Exception as e:
                pass  # Skip unavailable series

        # Forward fill and backward fill
        df = df.ffill().bfill()

        print(f"\nLoaded {len(self.indicators_loaded)} indicators")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features from raw indicators.

        Args:
            df: DataFrame with raw indicators

        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        df = df.copy()

        # Identify indicator columns (exclude GDP and gdp_growth)
        indicators = [c for c in df.columns if c not in ['GDPC1', 'gdp_growth']]

        # 1. Lag features (1-4 quarters)
        for col in indicators:
            for lag in [1, 2, 3, 4]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # GDP lags (autoregressive)
        if 'gdp_growth' in df.columns:
            for lag in [1, 2, 3, 4]:
                df[f'gdp_growth_lag{lag}'] = df['gdp_growth'].shift(lag)

        # 2. Growth rates
        for col in indicators:
            df[f'{col}_qoq'] = df[col].pct_change() * 100  # Quarter-over-quarter
            df[f'{col}_yoy'] = df[col].pct_change(4) * 100  # Year-over-year

        # 3. Momentum (deviation from 4-quarter moving average)
        for col in indicators:
            ma4 = df[col].rolling(4).mean()
            df[f'{col}_momentum'] = df[col] - ma4

        # 4. Rolling statistics
        for col in indicators:
            df[f'{col}_roll_std4'] = df[col].rolling(4).std()

        # 5. Acceleration (change in growth rate)
        for col in indicators:
            qoq = f'{col}_qoq'
            if qoq in df.columns:
                df[f'{col}_accel'] = df[qoq].diff()

        # 6. Composite indicators
        if 'IMPGS' in df.columns and 'EXPGS' in df.columns:
            df['trade_balance'] = df['EXPGS'] - df['IMPGS']
            df['trade_total'] = df['EXPGS'] + df['IMPGS']
            df['export_share'] = df['EXPGS'] / (df['trade_total'] + 1)

        if 'UNRATE' in df.columns and 'PAYEMS' in df.columns:
            df['labor_market_strength'] = df['PAYEMS'] / (df['UNRATE'] + 0.1)

        if 'T10Y2Y' in df.columns:
            df['yield_curve_inverted'] = (df['T10Y2Y'] < 0).astype(int)

        if 'HOUST' in df.columns and 'PERMIT' in df.columns:
            df['housing_pipeline'] = df['PERMIT'] - df['HOUST']

        # 7. Seasonal dummies
        df['q1'] = (df.index.quarter == 1).astype(int)
        df['q2'] = (df.index.quarter == 2).astype(int)
        df['q3'] = (df.index.quarter == 3).astype(int)
        df['q4'] = (df.index.quarter == 4).astype(int)

        # 8. Time trend
        df['time_trend'] = np.arange(len(df))

        print(f"  Created {len(df.columns)} total features")
        return df

    def prepare_data(self, df: pd.DataFrame, test_size: int = 8) -> Tuple:
        """
        Prepare data for training.

        Args:
            df: DataFrame with features
            test_size: Number of quarters for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Create target
        df['gdp_growth'] = df['GDPC1'].pct_change() * 100
        df = df.dropna(subset=['gdp_growth'])

        # Define features
        exclude = ['GDPC1', 'gdp_growth']
        self.feature_cols = [c for c in df.columns if c not in exclude]

        # Handle NaN and inf
        df[self.feature_cols] = df[self.feature_cols].ffill().bfill().fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # Split
        X = df[self.feature_cols]
        y = df['gdp_growth']

        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        return X_train, X_test, y_train, y_test

    def train(self, start_date: str = '2000-01-01', test_size: int = 8) -> Dict:
        """
        Train the nowcasting model.

        Args:
            start_date: Start date for training data
            test_size: Number of quarters to hold out for testing

        Returns:
            Dictionary with training metrics
        """
        # Fetch and prepare data
        df = self.fetch_data(start_date)
        df = self.engineer_features(df)
        X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)

        print(f"\nTraining data: {len(X_train)} quarters")
        print(f"Test data: {len(X_test)} quarters")
        print(f"Features: {len(self.feature_cols)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter tuning
        print("\nTuning model hyperparameters...")
        tscv = TimeSeriesSplit(n_splits=5)

        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'min_samples_leaf': [2, 3]
        }

        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        print(f"  Best params: {grid_search.best_params_}")

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'test_predictions': list(zip(y_test.index, y_test.values, y_pred)),
            'n_features': len(self.feature_cols),
            'n_indicators': len(self.indicators_loaded),
            'train_start': X_train.index[0],
            'train_end': X_train.index[-1],
            'test_start': X_test.index[0],
            'test_end': X_test.index[-1],
        }

        self.last_training_date = datetime.now()

        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"  RMSE: {self.training_metrics['rmse']:.4f}%")
        print(f"  MAE:  {self.training_metrics['mae']:.4f}%")
        print(f"  R²:   {self.training_metrics['r2']:.4f}")

        return self.training_metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.head(top_n)

    def nowcast(self, as_of_date: Optional[str] = None) -> Dict:
        """
        Generate a GDP nowcast using latest available data.

        Args:
            as_of_date: Date to nowcast as of (default: today)

        Returns:
            Dictionary with nowcast results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")

        print("\n" + "="*50)
        print("GDP NOWCAST")
        print("="*50)

        # Fetch latest data
        print("\nFetching latest data from FRED...")
        df = self.fetch_data(start_date='2015-01-01')
        df = self.engineer_features(df)

        # Create target for historical data
        df['gdp_growth'] = df['GDPC1'].pct_change() * 100

        # Get the latest row for prediction
        latest = df[self.feature_cols].iloc[-1:].copy()
        latest = latest.ffill().bfill().fillna(0)
        latest = latest.replace([np.inf, -np.inf], 0)

        # Scale and predict
        latest_scaled = self.scaler.transform(latest)
        prediction = self.model.predict(latest_scaled)[0]

        # Determine which quarter we're nowcasting
        latest_date = df.index[-1]
        current_quarter = f"Q{latest_date.quarter} {latest_date.year}"

        # Get last known actual GDP
        last_actual_idx = df['gdp_growth'].last_valid_index()
        last_actual = df.loc[last_actual_idx, 'gdp_growth']
        last_actual_quarter = f"Q{last_actual_idx.quarter} {last_actual_idx.year}"

        result = {
            'nowcast_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'target_quarter': current_quarter,
            'predicted_gdp_growth': prediction,
            'last_actual_quarter': last_actual_quarter,
            'last_actual_gdp_growth': last_actual,
            'model_rmse': self.training_metrics.get('rmse', None),
            'confidence_band': f"±{self.training_metrics.get('rmse', 0.5):.2f}%"
        }

        print(f"\n  Nowcast for: {current_quarter}")
        print(f"  Predicted GDP Growth: {prediction:.2f}%")
        print(f"  Confidence Band: {result['confidence_band']}")
        print(f"\n  Last Known: {last_actual_quarter} = {last_actual:.2f}%")

        return result

    def save(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'indicators_loaded': self.indicators_loaded,
            'training_metrics': self.training_metrics,
            'last_training_date': self.last_training_date,
            'api_key': self.api_key,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GDPNowcaster':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded GDPNowcaster instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        nowcaster = cls(api_key=save_data['api_key'])
        nowcaster.model = save_data['model']
        nowcaster.scaler = save_data['scaler']
        nowcaster.feature_cols = save_data['feature_cols']
        nowcaster.indicators_loaded = save_data['indicators_loaded']
        nowcaster.training_metrics = save_data['training_metrics']
        nowcaster.last_training_date = save_data['last_training_date']

        print(f"Model loaded from {filepath}")
        print(f"  Trained on: {nowcaster.last_training_date}")
        print(f"  RMSE: {nowcaster.training_metrics['rmse']:.4f}%")

        return nowcaster


if __name__ == '__main__':
    # Example usage
    import sys

    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("Set FRED_API_KEY environment variable")
        sys.exit(1)

    # Train model
    nowcaster = GDPNowcaster(api_key=api_key)
    nowcaster.train()

    # Show feature importance
    print("\nTop 20 Features:")
    print(nowcaster.get_feature_importance(20))

    # Make nowcast
    result = nowcaster.nowcast()

    # Save model
    nowcaster.save('models/gdp_nowcaster.pkl')
