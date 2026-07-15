"""
Tests for Model Loading and Prediction
=======================================
Validates model loading, prediction output format, and risk classification.
"""

import os
import pytest

from src.predict import ChurnPredictor, classify_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_EXISTS = os.path.exists(MODEL_PATH)


class TestRiskClassification:
    """Tests for risk level classification logic."""

    def test_low_risk(self):
        assert classify_risk(0.1) == "Low"
        assert classify_risk(0.0) == "Low"
        assert classify_risk(0.29) == "Low"

    def test_medium_risk(self):
        assert classify_risk(0.3) == "Medium"
        assert classify_risk(0.45) == "Medium"
        assert classify_risk(0.59) == "Medium"

    def test_high_risk(self):
        assert classify_risk(0.6) == "High"
        assert classify_risk(0.87) == "High"
        assert classify_risk(1.0) == "High"

    def test_boundary_low_medium(self):
        assert classify_risk(0.3) == "Medium"

    def test_boundary_medium_high(self):
        assert classify_risk(0.6) == "High"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestChurnPredictor:
    """Tests for the ChurnPredictor class."""

    @pytest.fixture
    def predictor(self):
        return ChurnPredictor()

    @pytest.fixture
    def sample_features(self):
        return {
            "days_since_last_order": 15,
            "orders_last_30_days": 2,
            "avg_order_value": 450.0,
            "subscription_duration_days": 120,
            "coupon_usage_rate": 0.3,
            "meal_swap_frequency": 0.1,
            "order_consistency": 5.2,
            "order_trend_slope": -0.5,
            "avg_rating": 4.2,
            "rating_trend": -0.3,
            "engagement_score": 25.0,
            "engagement_decline": 0.2,
            "support_ticket_count": 1,
            "total_lifetime_orders": 35,
            "avg_items_per_order": 2.5,
            "is_premium": 1,
            "days_to_subscription_expiry": 10,
        }

    def test_prediction_returns_dict(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert isinstance(result, dict)

    def test_prediction_has_required_keys(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert "churn_probability" in result
        assert "risk_level" in result

    def test_probability_in_range(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert 0 <= result["churn_probability"] <= 1

    def test_risk_level_valid(self, predictor, sample_features):
        result = predictor.predict(sample_features)
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_batch_prediction(self, predictor, sample_features):
        results = predictor.predict_batch([sample_features, sample_features])
        assert len(results) == 2
        for result in results:
            assert "churn_probability" in result

    def test_model_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError):
            ChurnPredictor(model_path="/nonexistent/path/model.joblib")
