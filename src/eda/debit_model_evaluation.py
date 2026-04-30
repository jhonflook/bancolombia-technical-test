"""Modeling utilities for cryptocurrency price prediction.

This module provides functions for preparing data and training
regression models for time series prediction.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default columns to exclude from features
DEFAULT_EXCLUDE_COLUMNS = [
    "date", "coin_id", "id", "json_response", "year_month",
    "target", "price_usd", "market_cap_usd", "total_volume_usd",
    "day_name", "month_name", "trend_7d", "risk_classification"
]


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = "target",
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Prepare data for modeling.

    This function:
    1. Drops rows with NaN target
    2. Encodes categorical variables
    3. Selects numeric features
    4. Drops rows with NaN features

    Args:
        df: DataFrame with all features.
        target_col: Name of the target column.
        exclude_columns: Columns to exclude from features.

    Returns:
        Tuple of (X, y, feature_cols, dates).
    """
    if exclude_columns is None:
        exclude_columns = DEFAULT_EXCLUDE_COLUMNS

    df_model = df.copy()

    # Drop rows where target is NaN
    df_model = df_model.dropna(subset=[target_col])

    # Encode categorical features
    trend_map = {"Upward": 1, "Stable": 0, "Downward": -1}
    if "trend_7d" in df_model.columns:
        df_model["trend_7d_encoded"] = df_model["trend_7d"].map(trend_map).fillna(0)

    risk_map = {"High Risk": 2, "Medium Risk": 1, "Low Risk": 0}
    if "risk_classification" in df_model.columns:
        df_model["risk_encoded"] = df_model["risk_classification"].map(risk_map).fillna(0)

    # Select feature columns
    feature_cols = []
    for col in df_model.columns:
        if col in exclude_columns:
            continue
        if df_model[col].dtype in ["float64", "int64", "int32"]:
            feature_cols.append(col)

    # Drop rows with NaN in features
    df_clean = df_model[feature_cols + [target_col]].dropna()

    X = df_clean[feature_cols]
    y = df_clean[target_col]
    dates = df_model.loc[df_clean.index, "date"]

    return X, y, feature_cols, dates


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with RMSE, MAE, R², and MAPE metrics.
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def get_default_models() -> dict[str, Any]:
    """Get dictionary of default regression models.

    Returns:
        Dictionary mapping model names to model instances.
    """
    return {
        "OLS (Linear Regression)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    coin_name: str,
    test_size: float = 0.2,
    models: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Train multiple regression models and compare performance.

    Uses time-series split to maintain temporal order.

    Args:
        X: Feature matrix.
        y: Target variable.
        dates: Date series for each sample.
        coin_name: Name of the cryptocurrency.
        test_size: Proportion of data to use for testing.
        models: Dictionary of models to train. If None, uses default models.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (results, X_train, X_test, y_train, y_test, dates_test).
        results is a dictionary with model results including metrics and predictions.
    """
    if models is None:
        models = get_default_models()

    # Time-series split (keep temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:]

    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL TRAINING FOR {coin_name.upper()}")
        print(f"{'='*70}")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results = {}

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        metrics = {
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_mape": train_metrics["mape"],
            "test_mape": test_metrics["mape"],
        }

        results[name] = {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred_test,
        }

        if verbose:
            print(f"  {name}: Test RMSE=${test_metrics['rmse']:,.2f}, R²={test_metrics['r2']:.4f}")

    return results, X_train, X_test, y_train, y_test, dates_test


def get_best_model(results: dict) -> tuple[str, dict]:
    """Get the best model based on test RMSE.

    Args:
        results: Dictionary of model results from train_and_evaluate_models.

    Returns:
        Tuple of (best_model_name, best_result).
    """
    best_name = min(results.keys(), key=lambda x: results[x]["metrics"]["test_rmse"])
    return best_name, results[best_name]


def create_comparison_dataframe(results: dict) -> pd.DataFrame:
    """Create a comparison DataFrame from model results.

    Args:
        results: Dictionary of model results from train_and_evaluate_models.

    Returns:
        DataFrame with model comparison metrics.
    """
    comparison_data = []
    for name, result in results.items():
        m = result["metrics"]
        comparison_data.append({
            "Model": name,
            "Train RMSE": f"${m['train_rmse']:,.2f}",
            "Test RMSE": f"${m['test_rmse']:,.2f}",
            "Train R²": f"{m['train_r2']:.4f}",
            "Test R²": f"{m['test_r2']:.4f}",
            "Test MAPE": f"{m['test_mape']:.2f}%",
        })
    return pd.DataFrame(comparison_data)


def get_feature_importance(
    model: Any,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Get feature importance from a trained model.

    Works with tree-based models that have feature_importances_ attribute.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_cols: List of feature column names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with feature names and importance scores.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return importance_df.head(top_n)
