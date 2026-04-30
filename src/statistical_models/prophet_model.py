"""Prophet model implementation for time series forecasting."""

import logging
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet

from src.statistical_models.base import BaseStatisticalModel

# Suppress Prophet's verbose logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetWrapper:
    """Sklearn-compatible wrapper for Prophet model.

    This wrapper provides fit/predict interface compatible with sklearn
    for use in cross-validation and hyperparameter optimization.

    Note:
        Prophet requires datetime information. This wrapper generates
        sequential dates for CV compatibility when working with arrays.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
    ):
        """Initialize wrapper with Prophet parameters.

        Args:
            changepoint_prior_scale: Flexibility of trend changepoints.
            seasonality_prior_scale: Strength of seasonality model.
            seasonality_mode: 'additive' or 'multiplicative'.
            yearly_seasonality: Include yearly seasonality.
            weekly_seasonality: Include weekly seasonality.
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model: Prophet | None = None
        self._n_features = 0
        self._train_end_date: pd.Timestamp | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | pd.Series) -> "ProphetWrapper":
        """Fit Prophet model.

        Args:
            X: Feature array (used as additional regressors).
            y: Target array.

        Returns:
            Self for sklearn compatibility.
        """
        self._n_features = X.shape[1] if len(X.shape) > 1 else 0

        # Generate sequential dates for training
        dates = pd.date_range(start="2020-01-01", periods=len(y), freq="D")
        self._train_end_date = dates[-1]

        # Create Prophet DataFrame
        df = pd.DataFrame({"ds": dates, "y": np.asarray(y)})

        # Create and configure Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
        )

        # Add features as regressors
        for i in range(self._n_features):
            regressor_name = f"regressor_{i}"
            self.model.add_regressor(regressor_name, standardize=True)
            df[regressor_name] = X[:, i]

        # Fit the model
        self.model.fit(df)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted Prophet model.

        Args:
            X: Feature array for prediction period.

        Returns:
            Array of predictions.
        """
        if self.model is None or self._train_end_date is None:
            raise ValueError("Model must be fitted before prediction")

        n_predictions = X.shape[0]

        # Generate dates following training period
        future_dates = pd.date_range(
            start=self._train_end_date + pd.Timedelta(days=1),
            periods=n_predictions,
            freq="D",
        )

        # Create future DataFrame
        future = pd.DataFrame({"ds": future_dates})

        # Add regressor values
        for i in range(self._n_features):
            future[f"regressor_{i}"] = X[:, i]

        # Make predictions
        forecast = self.model.predict(future)
        return forecast["yhat"].values

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
        }


class ProphetModel(BaseStatisticalModel):
    """Facebook Prophet model for time series forecasting.

    Prophet is a forecasting model developed by Meta/Facebook that handles:
    - Trend: Linear or logistic growth with changepoints
    - Seasonality: Daily, weekly, and yearly patterns
    - Holidays: Special events and their effects
    - Regressors: External features as additional regressors

    This implementation adapts Prophet to work with the feature-based
    interface used by other models in this module.

    Note:
        Prophet expects a DataFrame with 'ds' (datetime) column.
        If the input DataFrame has a DatetimeIndex or 'date' column,
        it will be used. Otherwise, a sequential date range is created.
    """

    name = "prophet"
    requires_scaling = False  # Prophet handles normalization internally

    def __init__(self):
        """Initialize the Prophet model."""
        super().__init__()
        self._prophet_params: dict[str, Any] = {}
        self._training_dates: pd.Series | None = None

    def _create_model(self, **params: Any) -> ProphetWrapper:
        """Create Prophet wrapper with given parameters.

        Args:
            **params: Prophet-specific hyperparameters.

        Returns:
            ProphetWrapper instance ready for fitting.
        """
        self._prophet_params = {
            "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.05),
            "seasonality_prior_scale": params.get("seasonality_prior_scale", 10.0),
            "seasonality_mode": params.get("seasonality_mode", "additive"),
            "yearly_seasonality": params.get("yearly_seasonality", True),
            "weekly_seasonality": params.get("weekly_seasonality", True),
        }

        return ProphetWrapper(**self._prophet_params)

    def get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for Prophet.

        Args:
            trial: Optuna trial for suggesting values.

        Returns:
            Dictionary with Prophet hyperparameters.
        """
        return {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.1, 100.0, log=True
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "yearly_seasonality": trial.suggest_categorical(
                "yearly_seasonality", [True, False]
            ),
            "weekly_seasonality": trial.suggest_categorical(
                "weekly_seasonality", [True, False]
            ),
        }

    def _extract_dates(self, X: pd.DataFrame) -> pd.Series:
        """Extract or generate dates from the DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            Series of datetime values.
        """
        # Check for DatetimeIndex
        if isinstance(X.index, pd.DatetimeIndex):
            return pd.Series(X.index, index=X.index)

        # Check for 'date' or 'ds' column
        for col in ["date", "ds", "Date", "DS", "datetime", "Datetime"]:
            if col in X.columns:
                return pd.to_datetime(X[col])

        # Generate sequential dates as fallback
        return pd.Series(pd.date_range(start="2020-01-01", periods=len(X), freq="D"))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: list[str],
        **params: Any,
    ) -> Self:
        """Fit the Prophet model on training data.

        Args:
            X: Feature DataFrame (selected features used as regressors).
            y: Target Series.
            selected_features: List of feature names to use as regressors.
            **params: Prophet hyperparameters.

        Returns:
            Self for method chaining.
        """
        self.selected_features = selected_features
        self.model = self._create_model(**params)

        # Extract dates
        dates = self._extract_dates(X)
        self._training_dates = dates

        # Prepare feature array
        X_selected = X[selected_features].values if selected_features else np.empty((len(y), 0))

        # Fit using wrapper
        self.model.fit(X_selected, y.values)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted Prophet model.

        Args:
            X: Feature DataFrame for prediction period.

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Prepare feature array
        if self.selected_features:
            X_selected = X[self.selected_features].values
        else:
            X_selected = np.empty((len(X), 0))

        return self.model.predict(X_selected)

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dictionary of Prophet parameters.
        """
        return self._prophet_params.copy()

    def get_feature_importances(self) -> dict[str, float] | None:
        """Get feature importances based on regressor coefficients.

        Prophet provides regressor coefficients that can be used
        as a measure of feature importance.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available.
        """
        if not self.is_fitted or self.model is None:
            return None

        if not self.selected_features:
            return None

        # Get regressor coefficients from Prophet model
        prophet_model = self.model.model
        if prophet_model is None:
            return None

        importances = {}
        try:
            regressor_coeffs = prophet_model.params.get("beta", None)
            if regressor_coeffs is not None and len(regressor_coeffs) > 0:
                # Prophet stores coefficients as a 2D array
                coeffs = np.array(regressor_coeffs).flatten()
                # Map coefficients to feature names
                extra_regressors = list(prophet_model.extra_regressors.keys())
                for i, reg_name in enumerate(extra_regressors):
                    if i < len(coeffs) and i < len(self.selected_features):
                        feature = self.selected_features[i]
                        importances[feature] = float(np.abs(coeffs[i]))
        except (AttributeError, KeyError, IndexError):
            # If coefficient extraction fails, return None
            return None

        return importances if importances else None
