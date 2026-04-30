"""Forecasting service for generating predictions and saving to database."""

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.database import get_session
from src.database.crud import insert_forecast_metadata, insert_many_forecast_coins
from src.dataset import generate_deterministic_features_for_date
from src.models import ForecastCoin, ForecastMetadata, TargetType
from src.statistical_models import BaseStatisticalModel

logger = logging.getLogger(__name__)


class ForecastingService:
    """Handles recursive forecasting and database persistence."""

    def __init__(self, target_type: TargetType):
        """Initialize the forecasting service.

        Args:
            target_type: Target variable type ('pct_change' or 'price_usd').
        """
        self.target_type = target_type

    def predict_future_recursive(
        self,
        model: BaseStatisticalModel,
        df_historical: pd.DataFrame,
        n_days: int,
    ) -> pd.DataFrame:
        """Make recursive multi-step predictions.

        Always returns predicted prices (price_usd), converting from
        pct_change if needed.

        Args:
            model: Trained model with selected_features set.
            df_historical: Historical data with price_usd column.
            n_days: Number of days to forecast.

        Returns:
            DataFrame with predictions including date, predicted_price,
            predicted_pct_change, horizon_days, and confidence.
        """
        df_hist = df_historical.copy().sort_values("date").reset_index(drop=True)
        last_date = pd.to_datetime(df_hist["date"].max())

        # Keep track of both price and pct_change series
        price_series = df_hist["price_usd"].tolist()

        # Compute pct_change series from prices
        pct_change_series = [0.0]  # First element has no change
        for i in range(1, len(price_series)):
            if price_series[i - 1] != 0:
                pct_change_series.append(
                    ((price_series[i] / price_series[i - 1]) - 1) * 100
                )
            else:
                pct_change_series.append(0.0)

        predictions = []
        feature_names = model.selected_features

        for day in range(1, n_days + 1):
            target_date = last_date + pd.Timedelta(days=day)
            features = generate_deterministic_features_for_date(target_date)

            # Get the appropriate base series for lag features
            if self.target_type == "pct_change":
                base_series = pd.Series(pct_change_series)
            else:
                base_series = pd.Series(price_series)

            # Create lag features for target
            for lag in range(1, 8):
                features[f"target_t_{lag}"] = (
                    base_series.iloc[-lag] if len(base_series) >= lag else 0
                )

            # Add price context features if using pct_change
            if self.target_type == "pct_change":
                extended_prices = pd.Series(price_series)
                for lag in range(1, 4):
                    features[f"price_t_{lag}"] = (
                        extended_prices.iloc[-lag]
                        if len(extended_prices) >= lag
                        else 0
                    )
            else:
                # Add return features for price target
                extended_pct = pd.Series(pct_change_series)
                features["return_t_0"] = (
                    extended_pct.iloc[-1] if len(extended_pct) >= 1 else 0
                )

                for lag in range(1, 4):
                    if len(extended_pct) >= lag + 1:
                        features[f"return_t_{lag}"] = extended_pct.iloc[-lag - 1]
                    else:
                        features[f"return_t_{lag}"] = 0

            # Rolling statistics based on target type
            for window in [7, 14, 30]:
                if len(base_series) >= window:
                    window_data = base_series.tail(window)
                    mean_val = window_data.mean()
                    std_val = window_data.std()
                else:
                    mean_val = base_series.mean()
                    std_val = base_series.std()

                features[f"rolling_mean_{window}d"] = (
                    mean_val if not np.isnan(mean_val) else 0
                )
                features[f"rolling_std_{window}d"] = (
                    std_val if not np.isnan(std_val) else 0
                )

            # Skew and kurtosis
            if len(base_series) >= 7:
                skew_val = skew(base_series.tail(7), nan_policy="omit")
                kurt_val = kurtosis(base_series.tail(7), nan_policy="omit")
                features["rolling_skew_7d"] = (
                    skew_val if not np.isnan(skew_val) else 0
                )
                features["rolling_kurt_7d"] = (
                    kurt_val if not np.isnan(kurt_val) else 0
                )
            else:
                features["rolling_skew_7d"] = 0
                features["rolling_kurt_7d"] = 0

            # Momentum (always price-based)
            extended_prices = pd.Series(price_series)
            for window in [7, 14]:
                if (
                    len(extended_prices) >= window
                    and extended_prices.iloc[-window] != 0
                ):
                    momentum_val = (
                        extended_prices.iloc[-1] / extended_prices.iloc[-window] - 1
                    )
                    features[f"momentum_{window}d"] = (
                        momentum_val if not np.isnan(momentum_val) else 0
                    )
                else:
                    features[f"momentum_{window}d"] = 0

            # RSI (always price-based)
            if len(extended_prices) >= 14:
                delta = extended_prices.diff().tail(14)
                gain = delta.where(delta > 0, 0).mean()
                loss = (-delta.where(delta < 0, 0)).mean()
                if loss != 0 and not np.isnan(loss):
                    rs = gain / loss
                    features["rsi_14d"] = 100 - (100 / (1 + rs))
                elif gain > 0:
                    features["rsi_14d"] = 100  # All gains, no losses
                else:
                    features["rsi_14d"] = 50  # No movement
            else:
                features["rsi_14d"] = 50

            features["trend_7d_encoded"] = 0

            # Create feature DataFrame
            X_future = pd.DataFrame([features])

            # Add missing features with default value
            for col in feature_names:
                if col not in X_future.columns:
                    X_future[col] = 0

            X_future = X_future[feature_names]

            # Handle NaN and inf values
            X_future = X_future.fillna(0)
            X_future = X_future.replace([np.inf, -np.inf], 0)

            # Make prediction
            prediction = model.predict(X_future)[0]

            # Convert prediction to price if using pct_change target
            if self.target_type == "pct_change":
                pred_pct_change = prediction
                last_price = price_series[-1]
                pred_price = last_price * (1 + pred_pct_change / 100)

                # Update both series for next iteration
                pct_change_series.append(pred_pct_change)
                price_series.append(pred_price)
            else:
                pred_price = prediction
                last_price = price_series[-1]
                if last_price != 0:
                    pred_pct_change = ((pred_price / last_price) - 1) * 100
                else:
                    pred_pct_change = 0

                # Update both series for next iteration
                price_series.append(pred_price)
                pct_change_series.append(pred_pct_change)

            confidence = max(0.3, 1.0 - (day * 0.05))

            predictions.append(
                {
                    "date": target_date,
                    "predicted_price": pred_price,
                    "predicted_pct_change": pred_pct_change,
                    "horizon_days": day,
                    "confidence": confidence,
                }
            )

        return pd.DataFrame(predictions)

    def save_predictions_to_db(
        self,
        predictions: pd.DataFrame,
        coin_id: str,
        run_id: str,
    ) -> None:
        """Save predictions to the database (always as price_usd).

        Args:
            predictions: DataFrame with predictions.
            coin_id: Identifier for the coin.
            run_id: MLflow run ID for tracking (used as forecast_id).
        """
        forecast_records = []
        run_date = datetime.now()

        for _, row in predictions.iterrows():
            confidence = row["confidence"]
            uncertainty = 1 - confidence

            forecast_records.append(
                ForecastCoin(
                    coin_id=coin_id,
                    date=row["date"].date(),
                    forecast_id=run_id,
                    predicted_price=row["predicted_price"],
                    upper_predicted_price=row["predicted_price"]
                    * (1 + uncertainty * 0.1),
                    lower_predicted_price=row["predicted_price"]
                    * (1 - uncertainty * 0.1),
                    forecast_run_date=run_date,
                )
            )

        with get_session() as session:
            insert_many_forecast_coins(session=session, forecasts=forecast_records)

        logger.info(
            f"  Saved {len(forecast_records)} predictions for {coin_id} to database"
        )

    def save_metadata_to_db(
        self,
        forecast_id: str,
        coin_id: str,
        model_type: str,
        mape: float,
        rmse: float,
        r2: float,
        mae: float,
        n_train_samples: int,
        n_test_samples: int,
        n_trials: int,
        n_features: int,
        selected_features: list[str],
        forecast_days: int,
        train_start_date: date,
        train_end_date: date,
        mlflow_run_id: str | None = None,
        tag: str = "unofficial",
    ) -> ForecastMetadata:
        """Save forecast metadata to the database.

        Args:
            forecast_id: Unique identifier for this forecast run.
            coin_id: Identifier for the coin.
            model_type: Name of the model used.
            mape: Mean Absolute Percentage Error.
            rmse: Root Mean Square Error.
            r2: R-squared coefficient.
            mae: Mean Absolute Error.
            n_train_samples: Number of training samples.
            n_test_samples: Number of test samples.
            n_trials: Number of Optuna optimization trials.
            n_features: Number of selected features.
            selected_features: List of selected feature names.
            forecast_days: Number of days forecasted.
            train_start_date: Training data start date.
            train_end_date: Training data end date.
            mlflow_run_id: Optional MLflow run ID.
            tag: Tag for the forecast (official or unofficial).

        Returns:
            The created ForecastMetadata object.
        """
        def to_python_float(val):
            """Convert numpy float to native Python float."""
            if hasattr(val, "item"):
                return val.item()
            return float(val) if val is not None else None

        metadata = ForecastMetadata(
            forecast_id=forecast_id,
            coin_id=coin_id,
            model_type=model_type,
            target_type=self.target_type,
            mape=to_python_float(mape),
            rmse=to_python_float(rmse),
            r2=to_python_float(r2),
            mae=to_python_float(mae),
            n_train_samples=int(n_train_samples),
            n_test_samples=int(n_test_samples),
            n_trials=int(n_trials),
            n_features=int(n_features),
            selected_features=",".join(selected_features),
            forecast_days=int(forecast_days),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            forecast_run_date=datetime.now(),
            tag=tag,
            mlflow_run_id=mlflow_run_id,
        )

        with get_session() as session:
            insert_forecast_metadata(session=session, metadata=metadata)

        logger.info(
            f"  Saved forecast metadata for {coin_id} "
            f"(model: {model_type}, tag: {tag})"
        )

        return metadata
