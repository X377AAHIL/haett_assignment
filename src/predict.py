"""
Prediction Module
=================
Handles model loading and inference for the churn prediction system.
Used by both the FastAPI endpoint and for batch predictions.
"""

import os
import joblib
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS
from src.recommendations import get_recommendation
from src.observability.logger import get_logger
from src.observability.exceptions import ModelNotLoadedError, PredictionError

logger = get_logger("prediction")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Risk level thresholds
RISK_THRESHOLDS = {
    "Low": 0.3,
    "Medium": 0.6,
}


def classify_risk(probability: float) -> str:
    """Classify churn probability into risk levels.

    Low:    probability < 0.3
    Medium: 0.3 <= probability < 0.6
    High:   probability >= 0.6
    """
    if probability < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif probability < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    else:
        return "High"


class ChurnPredictor:
    """Loads the trained model and transformer for making predictions."""

    def __init__(self, model_path: str = None, transformer_path: str = None):
        model_path = model_path or os.path.join(MODELS_DIR, "best_model.joblib")
        transformer_path = transformer_path or os.path.join(
            MODELS_DIR, "feature_transformer.joblib"
        )

        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            raise ModelNotLoadedError(
                f"Model not found at {model_path}. Run model training first."
            )

        if not os.path.exists(transformer_path):
            logger.error(f"Transformer not found at {transformer_path}")
            raise ModelNotLoadedError(
                f"Transformer not found at {transformer_path}. Run model training first."
            )

        self.model = joblib.load(model_path)
        self.transformer = joblib.load(transformer_path)
        self.feature_columns = FEATURE_COLUMNS

    def predict(self, features: dict) -> dict:
        """Make a churn prediction for a single user.

        Args:
            features: Dictionary with feature values matching FEATURE_COLUMNS

        Returns:
            Dictionary with churn_probability, risk_level, and recommendation
        """
        try:
            # Create DataFrame from input
            df = pd.DataFrame([features])

            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Select and order columns
            df = df[self.feature_columns]

            # Scale features
            X = self.transformer.scaler.transform(df)

            # Predict
            probability = float(self.model.predict_proba(X)[0, 1])
            risk_level = classify_risk(probability)

            # Get business recommendation
            recommendation = get_recommendation(features, risk_level)

            result = {
                "churn_probability": round(probability, 4),
                "risk_level": risk_level,
            }

            if recommendation:
                result["recommendation"] = recommendation

            logger.info(
                "Prediction successful",
                extra={
                    "churn_probability": result["churn_probability"],
                    "risk_level": risk_level,
                },
            )
            return result
        except Exception as e:
            logger.error(f"Prediction failed inside predictor: {e}", exc_info=True)
            raise PredictionError(f"Internal prediction logic failed: {e}")

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """Make predictions for multiple users."""
        return [self.predict(f) for f in features_list]
