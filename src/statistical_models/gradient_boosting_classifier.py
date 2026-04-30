"""Gradient Boosting classifier para débitos recurrentes."""

from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier

from src.statistical_models.base import BaseDebitClassifier


class GradientBoostingDebitClassifier(BaseDebitClassifier):
    """Gradient Boosting binario (sklearn) con corrección de desbalance vía sample_weight.

    GradientBoostingClassifier no soporta class_weight nativo; el desbalance
    3.7:1 se corrige construyendo sample_weight proporcional a scale_pos_weight:
      - Positivos (clase 1): peso = scale_pos_weight
      - Negativos (clase 0): peso = 1.0
    Esto equivale funcionalmente a class_weight={0:1, 1:scale_pos_weight} (S8).
    """

    name = "gradient_boosting"
    requires_scaling = False

    def _create_model(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        min_samples_leaf: int = 20,
        scale_pos_weight: float = 1.0,
        **kwargs: Any,
    ) -> GradientBoostingClassifier:
        """Crear GradientBoostingClassifier.

        Parameters
        ----------
        n_estimators : int
            Número de etapas de boosting.
        learning_rate : float
            Tasa de aprendizaje (shrinkage).
        max_depth : int
            Profundidad máxima de cada árbol base.
        subsample : float
            Fracción de muestras por etapa.
        min_samples_leaf : int
            Mínimo de muestras en nodo hoja (regularización).
        scale_pos_weight : float
            Almacenado internamente para construir sample_weight en fit().
        **kwargs : Any
            Argumentos adicionales ignorados.

        Returns
        -------
        GradientBoostingClassifier
            Clasificador configurado.
        """
        self._scale_pos_weight = scale_pos_weight
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

    def fit(
        self,
        X,
        y,
        selected_features: list[str],
        scale_pos_weight: float = 1.0,
        **params: Any,
    ):
        """Ajustar con sample_weight para corrección de desbalance.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento.
        y : pd.Series
            Target binario.
        selected_features : list[str]
            Columnas a utilizar.
        scale_pos_weight : float
            Peso de la clase positiva (S8). Clase negativa siempre = 1.0.
        **params : Any
            Hiperparámetros del modelo.
        """
        self.selected_features = selected_features
        X_proc = X[selected_features].values

        self.model = self._create_model(scale_pos_weight=scale_pos_weight, **params)

        sample_weight = np.where(np.asarray(y) == 1, scale_pos_weight, 1.0)
        self.model.fit(X_proc, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Espacio de búsqueda Optuna para Gradient Boosting.

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
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        }
