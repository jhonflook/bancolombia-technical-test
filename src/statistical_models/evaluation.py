"""Model evaluation utilities with cross-validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class CVResults:
    """Container for cross-validation results."""

    fold_results: pd.DataFrame
    mean_rmse: float
    std_rmse: float
    mean_mae: float
    mean_r2: float


def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    scale_features: bool = True,
) -> dict[str, Any]:
    """Evaluate model using time series cross-validation.

    Performs time series cross-validation where training data always
    precedes test data temporally. This prevents data leakage.

    Args:
        model: sklearn-compatible model with fit/predict methods.
        X: Feature DataFrame.
        y: Target Series.
        cv: TimeSeriesSplit cross-validator.
        scale_features: Whether to standardize features.

    Returns:
        Dictionary containing:
            - fold_results: DataFrame with per-fold metrics
            - mean_rmse: Mean RMSE across folds
            - std_rmse: Standard deviation of RMSE
            - mean_mae: Mean MAE across folds
            - mean_r2: Mean R2 across folds
    """
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        fold_results.append(
            {
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }
        )

    results_df = pd.DataFrame(fold_results)

    return {
        "fold_results": results_df,
        "mean_rmse": results_df["rmse"].mean(),
        "std_rmse": results_df["rmse"].std(),
        "mean_mae": results_df["mae"].mean(),
        "mean_r2": results_df["r2"].mean(),
    }


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with rmse, mae, and r2 metrics.
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
