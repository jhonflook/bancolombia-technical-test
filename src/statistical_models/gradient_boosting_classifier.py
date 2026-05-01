"""Gradient Boosting classifier para débitos recurrentes."""

from typing import Any

import optuna
from sklearn.ensemble import HistGradientBoostingClassifier

from src.statistical_models.base import BaseDebitClassifier


class GradientBoostingDebitClassifier(BaseDebitClassifier):
    """Histogram-based Gradient Boosting binario con corrección de desbalance nativa.

    Usa HistGradientBoostingClassifier en lugar de GradientBoostingClassifier porque:
    - 10-100x más rápido (algoritmo basado en histogramas, paralelo con OpenMP)
    - class_weight nativo: no requiere sample_weight manual
    - early_stopping desactivado para que Optuna controle la optimización

    El desbalance 3.7:1 se corrige con class_weight={0:1.0, 1:scale_pos_weight} (S8).
    """

    name = "gradient_boosting"
    requires_scaling = False

    def _create_model(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        scale_pos_weight: float = 1.0,
        **kwargs: Any,
    ) -> HistGradientBoostingClassifier:
        """Crear HistGradientBoostingClassifier con class_weight para desbalance.

        Parameters
        ----------
        max_iter : int
            Número máximo de etapas de boosting.
        learning_rate : float
            Tasa de aprendizaje (shrinkage).
        max_depth : int
            Profundidad máxima de cada árbol. None = sin límite.
        min_samples_leaf : int
            Mínimo de muestras en nodo hoja.
        l2_regularization : float
            Regularización L2 sobre las hojas.
        scale_pos_weight : float
            Peso de la clase positiva (S8). Clase negativa siempre = 1.0.
        **kwargs : Any
            Argumentos adicionales ignorados.

        Returns
        -------
        HistGradientBoostingClassifier
            Clasificador configurado.
        """
        class_weight = {0: 1.0, 1: scale_pos_weight}
        return HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            class_weight=class_weight,
            early_stopping=False,
            random_state=42,
        )

    def get_feature_importances(self) -> dict[str, float] | None:
        """Devuelve None: HistGradientBoostingClassifier no expone feature_importances_.

        Para importancias usar sklearn.inspection.permutation_importance externamente.
        """
        return None

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Espacio de búsqueda Optuna para Histogram Gradient Boosting.

        Parameters
        ----------
        trial : optuna.Trial
            Trial de Optuna.

        Returns
        -------
        dict[str, Any]
            Hiperparámetros sugeridos.
        """
        return {
            "max_iter":          trial.suggest_int("max_iter", 100, 400),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        }
