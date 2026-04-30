"""ElasticNet regression model implementation."""

from typing import Any

import optuna
from sklearn.linear_model import ElasticNet

from src.statistical_models.base import BaseStatisticalModel


class ElasticNetModel(BaseStatisticalModel):
    """ElasticNet regression model with combined L1/L2 regularization.

    ElasticNet combines Lasso (L1) and Ridge (L2) regularization,
    providing both feature selection and coefficient shrinkage.
    """

    name = "elasticnet"
    requires_scaling = True

    def _create_model(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        **kwargs: Any,
    ) -> ElasticNet:
        """Create ElasticNet regression model.

        Args:
            alpha: Constant that multiplies the penalty terms.
            l1_ratio: Mix between L1 and L2 (0 = Ridge, 1 = Lasso).
            **kwargs: Additional arguments (ignored).

        Returns:
            Configured ElasticNet model.
        """
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for ElasticNet.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with 'alpha' and 'l1_ratio' hyperparameters.
        """
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }
