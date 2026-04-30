"""XGBoost classifier para débitos recurrentes."""

from typing import Any

import optuna
from xgboost import XGBClassifier

from src.statistical_models.base import BaseDebitClassifier


class XGBoostDebitClassifier(BaseDebitClassifier):
    """Clasificador XGBoost binario con corrección de desbalance.

    Maneja el desbalance 3.7:1 vía scale_pos_weight (S8).
    Usa early stopping con el 15% final del conjunto de entrenamiento
    como validación interna para evitar overfitting sin tocar Test (S8).

    El espacio de hiperparámetros incluye parámetros de regularización
    (gamma, reg_alpha, reg_lambda) especialmente relevantes para la alta
    dimensionalidad del bloque canales (~4,252 features, S10).
    """

    name = "xgboost"
    requires_scaling = False

    def _create_model(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        **kwargs: Any,
    ) -> XGBClassifier:
        """Crear XGBClassifier binario.

        Parameters
        ----------
        n_estimators : int
            Número máximo de árboles (early stopping puede reducirlo).
        max_depth : int
            Profundidad máxima de cada árbol.
        learning_rate : float
            Tasa de aprendizaje (shrinkage).
        subsample : float
            Fracción de observaciones por árbol.
        colsample_bytree : float
            Fracción de features por árbol.
        min_child_weight : int
            Suma mínima de pesos en nodo hoja.
        gamma : float
            Reducción mínima de pérdida para particionar.
        reg_alpha : float
            Regularización L1.
        reg_lambda : float
            Regularización L2.
        scale_pos_weight : float
            Cociente n_neg/n_pos para corrección de desbalance (S8).
        **kwargs : Any
            Argumentos adicionales ignorados.

        Returns
        -------
        XGBClassifier
            Clasificador configurado.
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(
        self,
        X,
        y,
        selected_features: list[str],
        scale_pos_weight: float = 1.0,
        **params: Any,
    ):
        """Ajustar XGBoost con early stopping temporal.

        El 15% final del conjunto recibido se usa como validación interna
        para early stopping, preservando el orden temporal de los datos (S7).

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento.
        y : pd.Series
            Target binario.
        selected_features : list[str]
            Columnas a utilizar.
        scale_pos_weight : float
            Cociente n_neg/n_pos (S8).
        **params : Any
            Hiperparámetros del modelo.
        """
        self.selected_features = selected_features
        X_sel = X[selected_features].fillna(0.0)

        split_idx = int(len(X_sel) * 0.85)
        X_tr, X_val = X_sel.iloc[:split_idx], X_sel.iloc[split_idx:]
        y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = self._create_model(scale_pos_weight=scale_pos_weight, **params)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> "np.ndarray":
        """Predecir probabilidades con imputación de nulos (S4, S5)."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ajustarse antes de predecir.")
        return self.model.predict_proba(X[self.selected_features].fillna(0.0))[:, 1]

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Espacio de búsqueda Optuna para XGBoost.

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
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
        }
