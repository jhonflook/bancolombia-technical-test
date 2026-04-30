"""Lag-based feature engineering for time series prediction.

This module provides functions to create lagged features for
time series modeling.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def create_lag_features_and_target(
    df: pd.DataFrame,
    n_lags: int = 7,
) -> pd.DataFrame:
    """Create lagged price features and target variable for time series prediction.

    Features added:
    - price_t_1 to price_t_n: Price from 1 to n days ago
    - target: Next day's price (T+1) for prediction

    Args:
        df: DataFrame with 'date' and 'price_usd' columns.
        n_lags: Number of lag features to create (default 7).

    Returns:
        DataFrame with added lag features and target column.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Create lagged price features (T-1 to T-n)
    for lag in range(1, n_lags + 1):
        df[f"price_t_{lag}"] = df["price_usd"].shift(lag)

    # Create target variable: next day's price (T+1)
    df["target"] = df["price_usd"].shift(-1)

    return df


def add_rolling_statistical_features(
    df: pd.DataFrame,
    column: str = "price_usd",
    windows: list[int] = None,
) -> pd.DataFrame:
    """Add rolling statistical features including skewness and kurtosis.

    Features added for each window:
    - Rolling skewness: Measures asymmetry of the distribution
    - Rolling kurtosis: Measures tailedness of the distribution
    - Rolling standard deviation
    - Rolling coefficient of variation (CV = std/mean)

    Args:
        df: DataFrame with the target column.
        column: Column name to compute statistics for.
        windows: List of window sizes (default [7, 14, 30]).

    Returns:
        DataFrame with added rolling statistical features.
    """
    if windows is None:
        windows = [7, 14, 30]

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    for window in windows:
        # Rolling skewness
        df[f"skewness_{window}d"] = (
            df[column]
            .rolling(window=window, min_periods=window)
            .apply(lambda x: skew(x, nan_policy="omit"), raw=False)
        )

        # Rolling kurtosis
        df[f"kurtosis_{window}d"] = (
            df[column]
            .rolling(window=window, min_periods=window)
            .apply(lambda x: kurtosis(x, nan_policy="omit"), raw=False)
        )

        # Rolling standard deviation
        df[f"std_{window}d"] = df[column].rolling(window=window, min_periods=window).std()

        # Coefficient of variation (relative volatility)
        rolling_mean = df[column].rolling(window=window, min_periods=window).mean()
        rolling_std = df[column].rolling(window=window, min_periods=window).std()
        df[f"cv_{window}d"] = rolling_std / rolling_mean

    return df


def add_scaling_features(
    df: pd.DataFrame,
    columns_to_scale: list[str] = None,
) -> pd.DataFrame:
    """Add scaled versions of numeric columns using different scaling methods.

    Scaling methods:
    - MinMax: Scales to [0, 1] range
    - Standard (Z-score): Mean=0, Std=1
    - Robust: Uses median and IQR, robust to outliers

    Args:
        df: DataFrame with numeric columns to scale.
        columns_to_scale: List of column names to scale.

    Returns:
        DataFrame with added scaled features.
    """
    if columns_to_scale is None:
        columns_to_scale = ["price_usd"]

    df = df.copy()

    for col in columns_to_scale:
        if col not in df.columns:
            continue

        values = df[col].values.reshape(-1, 1)
        mask = ~np.isnan(values.flatten())

        # MinMax scaling
        df[f"{col}_minmax"] = np.nan
        if mask.sum() > 0:
            scaler = MinMaxScaler()
            df.loc[mask, f"{col}_minmax"] = scaler.fit_transform(values[mask].reshape(-1, 1)).flatten()

        # Standard scaling (Z-score)
        df[f"{col}_zscore"] = np.nan
        if mask.sum() > 0:
            scaler = StandardScaler()
            df.loc[mask, f"{col}_zscore"] = scaler.fit_transform(values[mask].reshape(-1, 1)).flatten()

        # Robust scaling
        df[f"{col}_robust"] = np.nan
        if mask.sum() > 0:
            scaler = RobustScaler()
            df.loc[mask, f"{col}_robust"] = scaler.fit_transform(values[mask].reshape(-1, 1)).flatten()

    return df


def prepare_regression_features(
    df: pd.DataFrame,
    exclude_columns: list[str] = None,
    categorical_columns: list[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare features for regression modeling.

    This function:
    1. Encodes categorical variables
    2. Selects numeric features
    3. Returns clean feature dataframe

    Args:
        df: DataFrame with all features.
        exclude_columns: Columns to exclude from features.
        categorical_columns: Categorical columns to encode.

    Returns:
        Tuple of (feature DataFrame, list of feature column names).
    """
    if exclude_columns is None:
        exclude_columns = [
            "date", "coin_id", "id", "json_response", "year_month",
            "target", "price_usd", "market_cap_usd", "total_volume_usd",
        ]

    if categorical_columns is None:
        categorical_columns = ["trend_7d", "risk_classification", "day_name", "month_name"]

    df = df.copy()

    # Encode categorical features
    if "trend_7d" in df.columns:
        trend_map = {"Upward": 1, "Stable": 0, "Downward": -1}
        df["trend_7d_encoded"] = df["trend_7d"].map(trend_map).fillna(0)

    if "risk_classification" in df.columns:
        risk_map = {"High Risk": 2, "Medium Risk": 1, "Low Risk": 0}
        df["risk_encoded"] = df["risk_classification"].map(risk_map).fillna(0)

    # Select numeric feature columns
    feature_cols = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        if col in categorical_columns:
            continue
        if df[col].dtype in ["float64", "int64", "int32"]:
            feature_cols.append(col)

    return df[feature_cols], feature_cols
