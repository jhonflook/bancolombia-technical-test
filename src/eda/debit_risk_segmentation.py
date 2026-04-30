"""Risk classification functions for cryptocurrency analysis.

This module provides functions to classify cryptocurrencies based on
their price volatility and risk levels.
"""

from typing import Literal

import numpy as np
import pandas as pd

RiskLevel = Literal["High Risk", "Medium Risk", "Low Risk"]


def calculate_daily_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage change between consecutive days.

    Args:
        df: DataFrame with 'date' and 'price_usd' columns.

    Returns:
        DataFrame with added 'pct_change' column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["price_usd"].pct_change() * 100
    return df


def classify_risk(max_drop: float) -> RiskLevel:
    """Classify risk based on maximum drop percentage.

    Args:
        max_drop: The maximum percentage drop (negative value).

    Returns:
        Risk classification: "High Risk", "Medium Risk", or "Low Risk".
    """
    if pd.isna(max_drop):
        return "Low Risk"
    if max_drop <= -50:
        return "High Risk"
    elif max_drop <= -20:
        return "Medium Risk"
    else:
        return "Low Risk"


def classify_coin_risk_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Classify coin risk for each calendar month based on max consecutive day drop.

    High Risk: 50%+ drop on any two consecutive days in a month
    Medium Risk: 20%+ drop on consecutive days
    Low Risk: Otherwise

    Args:
        df: DataFrame with 'date', 'price_usd', and 'pct_change' columns.

    Returns:
        DataFrame with added 'year_month' and 'risk_classification' columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    # Ensure pct_change exists
    if "pct_change" not in df.columns:
        df = calculate_daily_pct_change(df)

    # Group by month and find the minimum (most negative) pct_change
    monthly_max_drop = df.groupby("year_month")["pct_change"].min().reset_index()
    monthly_max_drop.columns = ["year_month", "max_drop"]

    # Classify risk based on max drop
    monthly_max_drop["risk_classification"] = monthly_max_drop["max_drop"].apply(classify_risk)

    # Merge back to original df
    df = df.merge(
        monthly_max_drop[["year_month", "risk_classification"]],
        on="year_month",
        how="left"
    )
    return df


def add_trend_and_variance(
    df: pd.DataFrame,
    threshold_pct: float = 2.0
) -> pd.DataFrame:
    """Add 7-day trend and variance columns to the dataframe.

    Features added:
    - trend_7d: 'Upward', 'Downward', or 'Stable' based on T0 vs avg of T-1 to T-7
    - variance_7d: Rolling variance over the past 7 days

    Args:
        df: DataFrame with 'date' and 'price_usd' columns.
        threshold_pct: Percentage threshold for trend classification (default 2%).

    Returns:
        DataFrame with added 'trend_7d' and 'variance_7d' columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate rolling variance for past 7 days (including current day)
    df["variance_7d"] = df["price_usd"].rolling(window=7, min_periods=7).var()

    # Calculate 7-day rolling mean (excluding current day, using shift)
    df["avg_prev_7d"] = df["price_usd"].shift(1).rolling(window=7, min_periods=7).mean()

    # Determine trend by comparing current price (T0) to average of previous 7 days
    def determine_trend(row) -> str | None:
        if pd.isna(row["avg_prev_7d"]):
            return None
        pct_diff = ((row["price_usd"] - row["avg_prev_7d"]) / row["avg_prev_7d"]) * 100
        if pct_diff > threshold_pct:
            return "Upward"
        elif pct_diff < -threshold_pct:
            return "Downward"
        else:
            return "Stable"

    df["trend_7d"] = df.apply(determine_trend, axis=1)

    # Drop helper column
    df = df.drop(columns=["avg_prev_7d"])

    return df


def get_risk_summary(df: pd.DataFrame, coin_name: str) -> dict:
    """Get summary statistics for risk analysis.

    Args:
        df: DataFrame with risk classification features.
        coin_name: Name of the cryptocurrency.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "coin": coin_name,
        "total_days": len(df),
        "risk_distribution": df["risk_classification"].value_counts().to_dict(),
        "trend_distribution": df["trend_7d"].value_counts().to_dict() if "trend_7d" in df.columns else {},
        "variance_mean": df["variance_7d"].mean() if "variance_7d" in df.columns else None,
        "variance_std": df["variance_7d"].std() if "variance_7d" in df.columns else None,
        "max_drop": df["pct_change"].min() if "pct_change" in df.columns else None,
        "max_gain": df["pct_change"].max() if "pct_change" in df.columns else None,
    }
    return summary
