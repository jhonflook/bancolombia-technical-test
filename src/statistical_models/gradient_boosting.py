"""Gradient Boosting regression model implementation."""

from typing import Any

import optuna
from sklearn.ensemble import GradientBoostingRegressor

from src.statistical_models.base import BaseStatisticalModel


class GradientBoostingModel(BaseStatisticalModel):
    """Gradient Boosting regression model.

    Gradient Boosting builds an ensemble of weak learners (decision trees)
    sequentially, where each new tree corrects errors made by the previous
    ensemble.
    """

    name = "gradient_boosting"
    requires_scaling = False

    def _create_model(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        **kwargs: Any,
    ) -> GradientBoostingRegressor:
        """Create Gradient Boosting regression model.

        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Shrinks contribution of each tree.
            max_depth: Maximum depth of each tree.
            subsample: Fraction of samples used for fitting each tree.
            **kwargs: Additional arguments (ignored).

        Returns:
            Configured GradientBoostingRegressor model.
        """
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42,
        )

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for Gradient Boosting.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with boosting-related hyperparameters.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
