"""Logistic Regression classifier para débitos recurrentes."""

from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression

from src.statistical_models.base import BaseDebitClassifier


class LogisticRegressionDebitClassifier(BaseDebitClassifier):
    """Regresión logística binaria — baseline lineal interpretable.

    Baseline de referencia para comparar contra modelos de árbol. Útil
    para auditorías de modelo y verificación de que features lineales aportan.

    Requiere escalado de features (requires_scaling = True).
    Usa class_weight para corrección de desbalance (S8).
    Regularización L2 por defecto (solver lbfgs).
    """

    name = "logistic_regression"
    requires_scaling = True

    def _create_model(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 2000,
        scale_pos_weight: float = 1.0,
        **kwargs: Any,
    ) -> LogisticRegression:
        """Crear LogisticRegression con class_weight para desbalance.

        Parameters
        ----------
        C : float
            Inverso de la fuerza de regularización (C alto = menos regularización).
        solver : str
            Algoritmo de optimización: 'lbfgs' (L2) o 'saga' (L1/L2/ElasticNet).
        max_iter : int
            Máximo de iteraciones para convergencia.
        scale_pos_weight : float
            Peso relativo de la clase positiva (S8).
        **kwargs : Any
            Argumentos adicionales ignorados.

        Returns
        -------
        LogisticRegression
            Clasificador configurado.
        """
        class_weight = {0: 1.0, 1: scale_pos_weight}
        return LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Espacio de búsqueda Optuna para Logistic Regression.

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
            "C":        trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver":   trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 2000,
        }
