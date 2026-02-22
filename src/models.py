"""
Model training module for GDP nowcasting.
Implements Random Forest and Gradient Boosting regressors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class GDPNowcaster:
    """
    GDP Nowcasting model wrapper.
    Supports Random Forest and Gradient Boosting.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize the nowcaster.

        Args:
            model_type: "random_forest" or "gradient_boosting"
            random_state: Random seed for reproducibility
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        self._init_model(model_params)

    def _init_model(self, params: Dict[str, Any]):
        """Initialize the underlying model."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": -1
            }
            default_params.update(params)
            self.model = RandomForestRegressor(**default_params)

        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                "random_state": self.random_state
            }
            default_params.update(params)
            self.model = GradientBoostingRegressor(**default_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True
    ) -> "GDPNowcaster":
        """
        Fit the model to training data.

        Args:
            X: Feature DataFrame
            y: Target Series
            scale_features: Whether to standardize features

        Returns:
            self
        """
        self.feature_names = list(X.columns)

        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            scale_features: Whether to scale features

        Returns:
            Dictionary with CV scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # RMSE scores (negative MSE from sklearn)
        mse_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=tscv, scoring="neg_mean_squared_error"
        )
        rmse_scores = np.sqrt(-mse_scores)

        # MAE scores
        mae_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=tscv, scoring="neg_mean_absolute_error"
        )

        return {
            "cv_rmse_mean": rmse_scores.mean(),
            "cv_rmse_std": rmse_scores.std(),
            "cv_mae_mean": -mae_scores.mean(),
            "cv_mae_std": mae_scores.std()
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return importance_df.head(top_n)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5
) -> Tuple[Dict[str, GDPNowcaster], Dict[str, Dict]]:
    """
    Train both Random Forest and Gradient Boosting models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        cv_splits: Number of cross-validation splits

    Returns:
        Tuple of (models dict, results dict)
    """
    models = {}
    results = {}

    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)

    rf = GDPNowcaster(model_type="random_forest")
    rf_cv = rf.cross_validate(X_train, y_train, n_splits=cv_splits)
    print(f"CV RMSE: {rf_cv['cv_rmse_mean']:.4f} (+/- {rf_cv['cv_rmse_std']:.4f})")
    print(f"CV MAE: {rf_cv['cv_mae_mean']:.4f} (+/- {rf_cv['cv_mae_std']:.4f})")

    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)

    models["random_forest"] = rf
    results["random_forest"] = {
        "cv_scores": rf_cv,
        "predictions": rf_predictions,
        "feature_importance": rf.get_feature_importance()
    }

    # Train Gradient Boosting
    print("\n" + "="*50)
    print("Training Gradient Boosting...")
    print("="*50)

    gb = GDPNowcaster(model_type="gradient_boosting")
    gb_cv = gb.cross_validate(X_train, y_train, n_splits=cv_splits)
    print(f"CV RMSE: {gb_cv['cv_rmse_mean']:.4f} (+/- {gb_cv['cv_rmse_std']:.4f})")
    print(f"CV MAE: {gb_cv['cv_mae_mean']:.4f} (+/- {gb_cv['cv_mae_std']:.4f})")

    gb.fit(X_train, y_train)
    gb_predictions = gb.predict(X_test)

    models["gradient_boosting"] = gb
    results["gradient_boosting"] = {
        "cv_scores": gb_cv,
        "predictions": gb_predictions,
        "feature_importance": gb.get_feature_importance()
    }

    return models, results


if __name__ == "__main__":
    # Test model training
    from sklearn.datasets import make_regression

    # Create sample data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Split
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    # Train
    models, results = train_models(X_train, y_train, X_test, y_test)

    print("\n" + "="*50)
    print("Top features (Random Forest):")
    print(results["random_forest"]["feature_importance"].head(10))
