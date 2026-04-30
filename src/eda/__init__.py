"""Módulo EDA para análisis de débitos recurrentes — Bancolombia.

Funciones de exploración, ingeniería de features y evaluación de modelos
orientadas al dominio de mora temprana (1-30 días) y clasificación de débito.
"""

from src.eda.debit_lag_features import (
    add_rolling_statistical_features,
    add_scaling_features,
    create_lag_features_and_target,
    prepare_regression_features,
)
from src.eda.debit_model_evaluation import (
    calculate_metrics,
    create_comparison_dataframe,
    get_best_model,
    get_default_models,
    get_feature_importance,
    prepare_model_data,
    train_and_evaluate_models,
)
from src.eda.debit_risk_segmentation import (
    add_trend_and_variance,
    calculate_daily_pct_change,
    classify_coin_risk_by_month,
    classify_risk,
    get_risk_summary,
)
from src.eda.debit_temporal_features import add_holiday_features, add_time_features
from src.eda.visualization import (
    plot_excedentes_profile,
    plot_feature_importance,
    plot_gestiones_profile,
    plot_model_performance,
    plot_mora_by_class,
    plot_payment_channels,
    plot_risk_segments,
    plot_target_distribution,
    plot_temporal_evolution,
)
from src.eda.debit_payment_features import (
    add_volume_features,
    extract_market_data,
    get_volume_summary,
)

__all__ = [
    # Segmentación de riesgo
    "calculate_daily_pct_change",
    "classify_risk",
    "classify_coin_risk_by_month",
    "add_trend_and_variance",
    "get_risk_summary",
    # Features de volumen/canal
    "extract_market_data",
    "add_volume_features",
    "get_volume_summary",
    # Features temporales
    "add_time_features",
    "add_holiday_features",
    # Features de lag
    "create_lag_features_and_target",
    "add_rolling_statistical_features",
    "add_scaling_features",
    "prepare_regression_features",
    # Evaluación de modelos
    "prepare_model_data",
    "train_and_evaluate_models",
    "calculate_metrics",
    "get_default_models",
    "get_best_model",
    "create_comparison_dataframe",
    "get_feature_importance",
    # Visualizaciones
    "plot_target_distribution",
    "plot_temporal_evolution",
    "plot_payment_channels",
    "plot_mora_by_class",
    "plot_gestiones_profile",
    "plot_risk_segments",
    "plot_feature_importance",
    "plot_model_performance",
    "plot_excedentes_profile",
]
