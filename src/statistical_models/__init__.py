"""Statistical models module for cryptocurrency price prediction."""

from src.statistical_models.base import BaseStatisticalModel, ModelMetrics
from src.statistical_models.elasticnet import ElasticNetModel
from src.statistical_models.evaluation import CVResults, compute_metrics, evaluate_model_cv
from src.statistical_models.gradient_boosting import GradientBoostingModel
from src.statistical_models.prophet_model import ProphetModel
from src.statistical_models.random_forest import RandomForestModel
from src.statistical_models.ridge import RidgeModel
from src.statistical_models.sarimax import SARIMAXModel

# Registry mapping model names to their classes
MODEL_REGISTRY: dict[str, type[BaseStatisticalModel]] = {
    "ridge": RidgeModel,
    "elasticnet": ElasticNetModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "sarimax": SARIMAXModel,
    "prophet": ProphetModel,
}

__all__ = [
    # Base class and metrics
    "BaseStatisticalModel",
    "ModelMetrics",
    # Model implementations
    "RidgeModel",
    "ElasticNetModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "SARIMAXModel",
    "ProphetModel",
    # Evaluation utilities
    "CVResults",
    "evaluate_model_cv",
    "compute_metrics",
    # Registry
    "MODEL_REGISTRY",
]
