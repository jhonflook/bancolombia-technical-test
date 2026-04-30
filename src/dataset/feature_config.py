"""Feature configuration for cryptocurrency price prediction models."""

from dataclasses import dataclass, field
from enum import Enum

from src.models import TargetType


class FeatureAvailability(Enum):
    """Classification of feature availability for future predictions."""

    DETERMINISTIC = "deterministic"
    LAG_DEPENDENT = "lag_dependent"
    EXTERNAL = "external"
    TARGET = "target"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering with availability metadata."""

    target_type: TargetType = "pct_change"

    deterministic: list[str] = field(
        default_factory=lambda: [
            "day_of_week",
            "is_weekend",
            "is_weekday",
            "week_of_year",
            "month",
            "quarter",
            "day_of_month",
            "day_of_year",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_us_holiday",
            "is_china_holiday",
            "is_any_holiday",
            "days_to_us_holiday",
            "days_from_us_holiday",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ]
    )

    # Lag-dependent features will be dynamically generated based on target_type
    lag_dependent: list[str] = field(default_factory=list)

    external: list[str] = field(
        default_factory=lambda: [
            "volume_ma_30d",
            "volume_ratio_30d",
            "volume_price_ratio",
            "market_cap_usd",
        ]
    )

    exclude: list[str] = field(
        default_factory=lambda: [
            "date",
            "coin_id",
            "id",
            "json_response",
            "year_month",
            "price_usd",
            "pct_change",
            "target",
            "day_name",
            "month_name",
            "risk_classification",
            "trend_7d",
        ]
    )

    target: str = "target"

    # Base feature always included in model (strongest predictor)
    base_feature: str = "target_t_1"

    def __post_init__(self):
        """Initialize lag_dependent features based on target_type."""
        self._update_lag_dependent_features()

    def _update_lag_dependent_features(self):
        """Update lag-dependent features based on target type."""
        if self.target_type == "pct_change":
            # Use lags of pct_change for pct_change target
            self.lag_dependent = [
                "target_t_1",
                "target_t_2",
                "target_t_3",
                "target_t_4",
                "target_t_5",
                "target_t_6",
                "target_t_7",
                "rolling_mean_7d",
                "rolling_mean_14d",
                "rolling_mean_30d",
                "rolling_std_7d",
                "rolling_std_14d",
                "rolling_std_30d",
                "rolling_skew_7d",
                "rolling_kurt_7d",
                "momentum_7d",
                "momentum_14d",
                "rsi_14d",
                "trend_7d_encoded",
                # Also keep price-based features for context
                "price_t_1",
                "price_t_2",
                "price_t_3",
            ]
        else:
            # Use lags of price_usd for price target
            self.lag_dependent = [
                "target_t_1",
                "target_t_2",
                "target_t_3",
                "target_t_4",
                "target_t_5",
                "target_t_6",
                "target_t_7",
                "return_t_1",
                "return_t_2",
                "return_t_3",
                "rolling_mean_7d",
                "rolling_mean_14d",
                "rolling_mean_30d",
                "rolling_std_7d",
                "rolling_std_14d",
                "rolling_std_30d",
                "rolling_skew_7d",
                "rolling_kurt_7d",
                "momentum_7d",
                "momentum_14d",
                "rsi_14d",
                "trend_7d_encoded",
            ]

    def set_target_type(self, target_type: TargetType):
        """Set target type and update lag-dependent features."""
        self.target_type = target_type
        self._update_lag_dependent_features()

    def get_all_features(self) -> list[str]:
        """Get all feature names."""
        return self.deterministic + self.lag_dependent + self.external

    def get_candidate_features(self) -> list[str]:
        """Get features to consider for selection (excluding base feature)."""
        all_features = self.deterministic + self.lag_dependent + self.external
        return [f for f in all_features if f != self.base_feature]


def create_feature_config(target_type: TargetType) -> FeatureConfig:
    """Factory function to create a FeatureConfig with the specified target type."""
    return FeatureConfig(target_type=target_type)
