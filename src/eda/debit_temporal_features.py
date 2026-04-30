"""Time-based feature engineering for cryptocurrency analysis.

This module provides functions to extract time-based features from dates.
"""

import holidays
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the date column.

    Features added:
    - day_of_week: 0=Monday, 6=Sunday
    - day_name: Monday, Tuesday, etc.
    - is_weekend: 1 if Saturday/Sunday, 0 otherwise
    - is_weekday: 1 if Monday-Friday, 0 otherwise
    - week_of_year: 1-52
    - month: 1-12
    - month_name: January, February, etc.
    - quarter: 1-4
    - day_of_month: 1-31
    - day_of_year: 1-365
    - is_month_start: 1 if first day of month
    - is_month_end: 1 if last day of month
    - is_quarter_start: 1 if first day of quarter
    - is_quarter_end: 1 if last day of quarter

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        DataFrame with added time features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Day of week features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_weekday"] = (df["day_of_week"] < 5).astype(int)

    # Week features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Month features
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()

    # Quarter features
    df["quarter"] = df["date"].dt.quarter

    # Day features
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    # Period boundary features
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday features for US and China markets.

    Features added:
    - is_us_holiday: 1 if US federal holiday, 0 otherwise
    - is_china_holiday: 1 if Chinese holiday, 0 otherwise
    - is_any_holiday: 1 if holiday in either country
    - days_to_us_holiday: Days until next US holiday
    - days_from_us_holiday: Days since last US holiday

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        DataFrame with added holiday features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Get date range for holidays
    years = df["date"].dt.year.unique().tolist()

    # Initialize holiday calendars
    us_holidays = holidays.US(years=years)
    china_holidays = holidays.China(years=years)

    # US holiday features
    df["is_us_holiday"] = df["date"].apply(lambda x: 1 if x in us_holidays else 0)

    # China holiday features
    df["is_china_holiday"] = df["date"].apply(lambda x: 1 if x in china_holidays else 0)

    # Combined holiday flag
    df["is_any_holiday"] = ((df["is_us_holiday"] == 1) | (df["is_china_holiday"] == 1)).astype(int)

    # Days to/from nearest US holiday
    us_holiday_dates = sorted([pd.Timestamp(d) for d in us_holidays.keys()])

    def days_to_next_holiday(date):
        future = [h for h in us_holiday_dates if h > date]
        return (future[0] - date).days if future else None

    def days_from_last_holiday(date):
        past = [h for h in us_holiday_dates if h <= date]
        return (date - past[-1]).days if past else None

    df["days_to_us_holiday"] = df["date"].apply(days_to_next_holiday)
    df["days_from_us_holiday"] = df["date"].apply(days_from_last_holiday)

    return df
