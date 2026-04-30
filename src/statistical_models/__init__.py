"""Módulo de clasificadores para débitos recurrentes.

Expone MODEL_REGISTRY y DEFAULT_MODELS para el pipeline de entrenamiento
multi-modelo (deploy/train_debit_classifier.py).
"""

from src.statistical_models.base import BaseDebitClassifier, ClassificationMetrics
from src.statistical_models.evaluation import (
    CVResults,
    compute_classification_metrics,
    compute_ks,
    evaluate_classifier_cv,
)
from src.statistical_models.gradient_boosting_classifier import GradientBoostingDebitClassifier
from src.statistical_models.logistic_regression import LogisticRegressionDebitClassifier
from src.statistical_models.random_forest_classifier import RandomForestDebitClassifier
from src.statistical_models.xgboost_classifier import XGBoostDebitClassifier

# Modelos entrenados por defecto (excluye logistic_regression — solo baseline)
DEFAULT_MODELS: list[str] = ["xgboost", "random_forest", "gradient_boosting"]

# Registro completo: nombre → clase del clasificador
MODEL_REGISTRY: dict[str, type[BaseDebitClassifier]] = {
    "xgboost":             XGBoostDebitClassifier,
    "random_forest":       RandomForestDebitClassifier,
    "gradient_boosting":   GradientBoostingDebitClassifier,
    "logistic_regression": LogisticRegressionDebitClassifier,
}

__all__ = [
    # Base y métricas
    "BaseDebitClassifier",
    "ClassificationMetrics",
    # Clasificadores
    "XGBoostDebitClassifier",
    "RandomForestDebitClassifier",
    "GradientBoostingDebitClassifier",
    "LogisticRegressionDebitClassifier",
    # Evaluación
    "CVResults",
    "evaluate_classifier_cv",
    "compute_classification_metrics",
    "compute_ks",
    # Registro
    "DEFAULT_MODELS",
    "MODEL_REGISTRY",
]
