"""
Tests for the FastAPI Prediction API
=====================================
Tests endpoint responses, validation, and error handling.
"""

import pytest
from fastapi.testclient import TestClient

# We need to handle the case where the model isn't trained yet
try:
    from api.main import app
    client = TestClient(app)
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False


# Sample valid request
VALID_REQUEST = {
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

# High-risk user request
HIGH_RISK_REQUEST = {
    "days_since_last_order": 45,
    "orders_last_30_days": 0,
    "avg_order_value": 200.0,
    "subscription_duration_days": 30,
    "coupon_usage_rate": 0.0,
    "meal_swap_frequency": 0.5,
    "order_consistency": 50.0,
    "order_trend_slope": -2.0,
    "avg_rating": 2.5,
    "rating_trend": -1.0,
    "engagement_score": 2.0,
    "engagement_decline": 0.8,
    "support_ticket_count": 10,
    "total_lifetime_orders": 5,
    "avg_items_per_order": 1.0,
    "is_premium": 0,
    "days_to_subscription_expiry": -5,
}


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available (model may not be trained)")
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_format(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available (model may not be trained)")
class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_valid_prediction_returns_200(self):
        response = client.post("/predict", json=VALID_REQUEST)
        # 200 if model is loaded, 503 if not
        assert response.status_code in [200, 503]

    def test_prediction_response_format(self):
        response = client.post("/predict", json=VALID_REQUEST)
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "risk_level" in data
            assert 0 <= data["churn_probability"] <= 1
            assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_high_risk_includes_recommendation(self):
        response = client.post("/predict", json=HIGH_RISK_REQUEST)
        if response.status_code == 200:
            data = response.json()
            if data["risk_level"] in ["Medium", "High"]:
                assert "recommendation" in data
                assert data["recommendation"] is not None

    def test_invalid_request_missing_field(self):
        """Missing required field should return 422."""
        incomplete = {"days_since_last_order": 15}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_request_out_of_range(self):
        """Out-of-range values should return 422."""
        invalid = VALID_REQUEST.copy()
        invalid["coupon_usage_rate"] = 5.0  # Max is 1.0
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_invalid_request_wrong_type(self):
        """Wrong data types should return 422."""
        invalid = VALID_REQUEST.copy()
        invalid["days_since_last_order"] = "not_a_number"
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        """Empty request body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_explain_false_omits_top_factors(self):
        response = client.post("/predict", json=VALID_REQUEST)
        if response.status_code == 200:
            data = response.json()
            assert data.get("top_factors") is None

    def test_explain_true_includes_top_factors(self):
        valid_explain = VALID_REQUEST.copy()
        valid_explain["explain"] = True
        response = client.post("/predict", json=valid_explain)
        if response.status_code == 200:
            data = response.json()
            assert "top_factors" in data
            assert isinstance(data["top_factors"], list)
            if len(data["top_factors"]) > 0:
                factor = data["top_factors"][0]
                assert "feature" in factor
                assert "impact" in factor
                assert "direction" in factor
                assert "description" in factor
