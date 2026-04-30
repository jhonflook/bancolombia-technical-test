"""Volume-based feature engineering for cryptocurrency analysis.

This module provides functions to extract and engineer features from
trading volume data.
"""

import numpy as np
import pandas as pd


def extract_market_data(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract market cap and volume from JSON response.

    Args:
        df: DataFrame with 'json_response' column containing CoinGecko data.

    Returns:
        Tuple of (market_cap_series, total_volume_series).
    """
    market_cap = df["json_response"].apply(
        lambda x: x.get("market_data", {}).get("market_cap", {}).get("usd")
        if isinstance(x, dict) else None
    )
    total_volume = df["json_response"].apply(
        lambda x: x.get("market_data", {}).get("total_volume", {}).get("usd")
        if isinstance(x, dict) else None
    )
    return market_cap, total_volume


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create volume-based features for trading analysis.

    Features added:
    - volume_pct_change: Daily percentage change in volume
    - volume_ma_30d: 30-day moving average
    - volume_ratio_30d: Current volume / 30-day average (spike indicator)
    - volume_std_30d: Rolling standard deviation
    - volume_price_ratio: Volume / Price (liquidity indicator)
    - volume_market_cap_ratio: Volume / Market Cap (turnover ratio)
    - high_volume_day: 1 if volume > 1.5x 30-day average
    - low_volume_day: 1 if volume < 0.5x 30-day average

    Args:
        df: DataFrame with 'total_volume_usd' column.

    Returns:
        DataFrame with added volume features.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    vol_col = "total_volume_usd"

    if vol_col not in df.columns:
        return df

    # Volume percentage change
    df["volume_pct_change"] = df[vol_col].pct_change() * 100

    # Moving averages
    df["volume_ma_30d"] = df[vol_col].rolling(window=30, min_periods=1).mean()

    # Volume ratios (spike detection)
    df["volume_ratio_30d"] = df[vol_col] / df["volume_ma_30d"]

    # Rolling standard deviation
    df["volume_std_30d"] = df[vol_col].rolling(window=30, min_periods=30).std()

    # Volume relative to price (liquidity measure)
    if "price_usd" in df.columns:
        df["volume_price_ratio"] = df[vol_col] / df["price_usd"]

    # Volume relative to market cap (turnover ratio)
    if "market_cap_usd" in df.columns:
        df["volume_market_cap_ratio"] = df[vol_col] / df["market_cap_usd"]

    # High/Low volume day flags
    df["high_volume_day"] = (df["volume_ratio_30d"] > 1.5).astype(int)
    df["low_volume_day"] = (df["volume_ratio_30d"] < 0.5).astype(int)

    return df


def get_volume_summary(df: pd.DataFrame, coin_name: str) -> dict:
    """Get summary statistics for volume analysis.

    Args:
        df: DataFrame with volume features.
        coin_name: Name of the cryptocurrency.

    Returns:
        Dictionary with volume summary statistics.
    """
    summary = {
        "coin": coin_name,
        "total_days": len(df),
    }

    if "total_volume_usd" in df.columns:
        summary["avg_volume"] = df["total_volume_usd"].mean()
        summary["max_volume"] = df["total_volume_usd"].max()
        summary["min_volume"] = df["total_volume_usd"].min()

    if "high_volume_day" in df.columns:
        summary["high_volume_days"] = df["high_volume_day"].sum()
        summary["low_volume_days"] = df["low_volume_day"].sum()

    if "volume_ratio_30d" in df.columns:
        summary["max_volume_spike"] = df["volume_ratio_30d"].max()

    return summary
