import os
import uuid
import datetime
import pandas as pd
from src.observability.logger import get_logger

logger = get_logger("monitoring.prediction_logger")


class PredictionLogger:
    """Manages appending prediction requests to the production parquet dataset."""

    def __init__(
        self,
        production_path: str = "artifacts/monitoring/production/production_predictions.parquet",
    ):
        self.production_path = production_path
        os.makedirs(os.path.dirname(self.production_path), exist_ok=True)

    def log(
        self,
        features: dict,
        prediction_probability: float,
        prediction_class: str,
        model_version: str = "1.0",
    ):
        """Append a prediction to the production dataset.

        Args:
            features: Dictionary of the raw API input features
            prediction_probability: The churn probability (float)
            prediction_class: The final risk class or binary prediction string
            model_version: The version string of the model
        """
        try:
            # Add monitoring metadata
            data = features.copy()
            data["prediction_probability"] = prediction_probability
            data["prediction"] = prediction_class
            data["model_version"] = model_version
            data["prediction_timestamp"] = datetime.datetime.utcnow().isoformat()
            data["request_id"] = str(uuid.uuid4())

            df = pd.DataFrame([data])

            # Append or create parquet file
            if os.path.exists(self.production_path):
                # fastparquet supports append
                df.to_parquet(self.production_path, engine="fastparquet", append=True)
            else:
                df.to_parquet(self.production_path, engine="fastparquet")

        except Exception as e:
            # We don't want to crash the API if logging fails
            logger.error(f"Failed to log prediction to {self.production_path}: {e}")
