"""SARIMAX model implementation for time series forecasting."""

from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.statistical_models.base import BaseStatisticalModel


class SARIMAXWrapper:
    """Sklearn-compatible wrapper for SARIMAX model.

    This wrapper provides fit/predict interface compatible with sklearn
    for use in cross-validation and hyperparameter optimization.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """Initialize wrapper with SARIMAX parameters.

        Args:
            order: (p, d, q) order of the model.
            seasonal_order: (P, D, Q, s) seasonal order.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.results = None
        self._endog_train = None

    def fit(self, X: np.ndarray, y: np.ndarray | pd.Series) -> "SARIMAXWrapper":
        """Fit SARIMAX model.

        Args:
            X: Exogenous variables (features).
            y: Endogenous variable (target).

        Returns:
            Self for sklearn compatibility.
        """
        # Store training data length for prediction offset
        self._endog_train = np.asarray(y)

        # Handle exogenous variables
        exog = X if X.shape[1] > 0 else None

        model = SARIMAX(
            endog=self._endog_train,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.results = model.fit(disp=False, maxiter=500, warn_convergence=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted SARIMAX model.

        For cross-validation, this performs out-of-sample forecasting.

        Args:
            X: Exogenous variables for prediction period.

        Returns:
            Array of predictions.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")

        n_predictions = X.shape[0]
        exog = X if X.shape[1] > 0 else None

        # Forecast future values
        forecast = self.results.get_forecast(steps=n_predictions, exog=exog)
        predicted_mean = forecast.predicted_mean
        # Handle both pandas Series and numpy array
        if hasattr(predicted_mean, "values"):
            return predicted_mean.values
        return np.asarray(predicted_mean)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        }


class SARIMAXModel(BaseStatisticalModel):
    """SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors).

    SARIMAX is a time series model that combines:
    - AR (AutoRegressive): Uses past values of the target
    - I (Integrated): Differencing to make series stationary
    - MA (Moving Average): Uses past forecast errors
    - S (Seasonal): Handles seasonal patterns
    - X (eXogenous): Incorporates external features

    This implementation adapts SARIMAX to work with the feature-based
    interface used by other models in this module.
    """

    name = "sarimax"
    requires_scaling = False  # SARIMAX handles its own normalization

    def __init__(self):
        """Initialize the SARIMAX model."""
        super().__init__()
        self.order: tuple[int, int, int] = (1, 0, 1)
        self.seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.results: Any = None

    def _create_model(self, **params: Any) -> SARIMAXWrapper:
        """Create SARIMAX wrapper with given parameters.

        Args:
            **params: Model parameters (p, d, q, P, D, Q, s).

        Returns:
            SARIMAXWrapper instance ready for fitting.
        """
        # Extract order parameters
        p = params.get("p", 1)
        d = params.get("d", 0)
        q = params.get("q", 1)
        self.order = (p, d, q)

        # Extract seasonal order parameters
        P = params.get("P", 0)
        D = params.get("D", 0)
        Q = params.get("Q", 0)
        s = params.get("s", 7)  # Default weekly seasonality
        self.seasonal_order = (P, D, Q, s)

        return SARIMAXWrapper(order=self.order, seasonal_order=self.seasonal_order)

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for SARIMAX.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with SARIMAX hyperparameters.
        """
        return {
            # Non-seasonal order (p, d, q)
            "p": trial.suggest_int("p", 0, 3),
            "d": trial.suggest_int("d", 0, 2),
            "q": trial.suggest_int("q", 0, 3),
            # Seasonal order (P, D, Q, s)
            # Reduced complexity to avoid "too few observations" errors
            "P": trial.suggest_int("P", 0, 1),
            "D": trial.suggest_int("D", 0, 1),
            "Q": trial.suggest_int("Q", 0, 1),
            "s": trial.suggest_categorical("s", [7]),  # Weekly only for stability
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: list[str],
        **params: Any,
    ) -> Self:
        """Fit the SARIMAX model on training data.

        Args:
            X: Feature DataFrame (used as exogenous variables).
            y: Target Series (endogenous variable).
            selected_features: List of feature names to use as exogenous.
            **params: Model hyperparameters (p, d, q, P, D, Q, s).

        Returns:
            Self for method chaining.
        """
        self.selected_features = selected_features
        self.model = self._create_model(**params)

        # Prepare exogenous variables
        X_selected = X[selected_features].values if selected_features else np.empty((len(y), 0))

        # Fit the wrapper
        self.model.fit(X_selected, y.values)
        self.results = self.model.results
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted SARIMAX model.

        Args:
            X: Feature DataFrame for prediction period.

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Prepare exogenous variables
        if self.selected_features:
            X_selected = X[self.selected_features].values
        else:
            X_selected = np.empty((len(X), 0))

        return self.model.predict(X_selected)

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of SARIMAX parameters.
        """
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        }

    def get_feature_importances(self) -> dict[str, float] | None:
        """Get feature importances based on exogenous variable coefficients.

        Returns:
            Dictionary mapping feature names to absolute coefficient values,
            or None if not available.
        """
        if not self.is_fitted or self.results is None:
            return None

        if not self.selected_features:
            return None

        # Get exogenous variable coefficients from params
        params = self.results.params
        importances = {}

        for i, feature in enumerate(self.selected_features):
            param_name = f"x{i+1}"  # SARIMAX names exog vars as x1, x2, etc.
            if param_name in params.index:
                importances[feature] = float(np.abs(params[param_name]))
            else:
                # Try alternative naming
                for name in params.index:
                    if feature.lower() in name.lower():
                        importances[feature] = float(np.abs(params[name]))
                        break

        return importances if importances else None
