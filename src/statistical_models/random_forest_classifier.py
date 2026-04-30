"""Random Forest classifier para débitos recurrentes."""

from typing import Any

import optuna
from sklearn.ensemble import RandomForestClassifier

from src.statistical_models.base import BaseDebitClassifier


class RandomForestDebitClassifier(BaseDebitClassifier):
    """Random Forest binario con corrección de desbalance vía class_weight.

    Convierte scale_pos_weight (n_neg/n_pos) en class_weight explícito
    {0: 1.0, 1: scale_pos_weight} para compatibilidad con el interface
    unificado de BaseDebitClassifier (S8).

    No requiere escalado de features; robusto a escala.
    """

    name = "random_forest"
    requires_scaling = False

    def _create_model(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        scale_pos_weight: float = 1.0,
        **kwargs: Any,
    ) -> RandomForestClassifier:
        """Crear RandomForestClassifier con class_weight para desbalance.

        Parameters
        ----------
        n_estimators : int
            Número de árboles en el bosque.
        max_depth : int or None
            Profundidad máxima. None = nodos expandidos hasta hojas puras.
        min_samples_split : int
            Mínimo de muestras para dividir un nodo interno.
        min_samples_leaf : int
            Mínimo de muestras requeridas en nodo hoja.
        max_features : str
            Número de features a considerar en cada split ('sqrt', 'log2').
        scale_pos_weight : float
            Peso relativo de la clase positiva (S8).
        **kwargs : Any
            Argumentos adicionales ignorados.

        Returns
        -------
        RandomForestClassifier
            Clasificador configurado.
        """
        class_weight = {0: 1.0, 1: scale_pos_weight}
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Espacio de búsqueda Optuna para Random Forest.

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
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
