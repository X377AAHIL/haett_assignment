"""
Tests for Feature Engineering
=============================
Validates feature computation logic and transformer behavior.
"""

import pytest
import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FeatureTransformer,
    compute_order_features,
    compute_engagement_features,
)


@pytest.fixture
def sample_orders():
    """Create sample order data for testing."""
    return pd.DataFrame({
        "order_id": [f"ORD{i}" for i in range(10)],
        "user_id": ["U001"] * 5 + ["U002"] * 5,
        "order_date": pd.date_range("2025-04-01", periods=5).tolist() * 2,
        "order_value": [100, 150, 120, 200, 180, 90, 80, 70, 60, 50],
        "items_count": [2, 3, 2, 4, 3, 1, 2, 1, 1, 1],
        "meal_swapped": [0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        "coupon_used": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "rating": [4.5, 4.0, 4.2, 4.8, 4.5, 3.0, 2.5, 2.0, 2.5, 2.0],
    })


@pytest.fixture
def sample_engagement():
    """Create sample engagement data for testing."""
    return pd.DataFrame({
        "user_id": ["U001"] * 5 + ["U002"] * 5,
        "date": pd.date_range("2025-06-01", periods=5).tolist() * 2,
        "app_opens": [5, 4, 3, 4, 5, 1, 1, 0, 0, 0],
        "recipes_viewed": [3, 2, 3, 2, 3, 0, 1, 0, 0, 0],
        "support_tickets": [0, 0, 0, 0, 0, 1, 1, 2, 1, 1],
        "notification_clicks": [2, 1, 2, 1, 2, 0, 0, 0, 0, 0],
    })


class TestFeatureColumns:
    """Tests for feature column definitions."""

    def test_feature_columns_count(self):
        """Should have exactly 17 features."""
        assert len(FEATURE_COLUMNS) == 17

    def test_feature_columns_no_duplicates(self):
        """Feature column names should be unique."""
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_required_features_present(self):
        """Assignment-required features should be present."""
        required = [
            "days_since_last_order",
            "orders_last_30_days",
            "avg_order_value",
            "subscription_duration_days",
            "coupon_usage_rate",
            "meal_swap_frequency",
            "order_consistency",
        ]
        for feature in required:
            assert feature in FEATURE_COLUMNS, f"Missing required feature: {feature}"


class TestOrderFeatures:
    """Tests for order-based feature computation."""

    def test_computes_avg_order_value(self, sample_orders):
        user_ids = pd.Series(["U001", "U002"])
        features = compute_order_features(user_ids, sample_orders)
        assert "avg_order_value" in features.columns

        u001_avg = features[features["user_id"] == "U001"]["avg_order_value"].values[0]
        expected = np.mean([100, 150, 120, 200, 180])
        assert abs(u001_avg - expected) < 0.01

    def test_computes_total_orders(self, sample_orders):
        user_ids = pd.Series(["U001", "U002"])
        features = compute_order_features(user_ids, sample_orders)
        assert features[features["user_id"] == "U001"]["total_lifetime_orders"].values[0] == 5

    def test_computes_meal_swap_frequency(self, sample_orders):
        user_ids = pd.Series(["U001", "U002"])
        features = compute_order_features(user_ids, sample_orders)
        u002_swap = features[features["user_id"] == "U002"]["meal_swap_frequency"].values[0]
        assert u002_swap == 0.8  # 4 out of 5 orders


class TestEngagementFeatures:
    """Tests for engagement-based feature computation."""

    def test_computes_support_tickets(self, sample_engagement):
        user_ids = pd.Series(["U001", "U002"])
        features = compute_engagement_features(user_ids, sample_engagement)
        u002_tickets = features[features["user_id"] == "U002"]["support_ticket_count"].values[0]
        assert u002_tickets == 6  # Sum of [1, 1, 2, 1, 1]

    def test_computes_engagement_score(self, sample_engagement):
        user_ids = pd.Series(["U001", "U002"])
        features = compute_engagement_features(user_ids, sample_engagement)
        assert "engagement_score" in features.columns
        # U001 should have higher engagement than U002
        u001_score = features[features["user_id"] == "U001"]["engagement_score"].values[0]
        u002_score = features[features["user_id"] == "U002"]["engagement_score"].values[0]
        assert u001_score > u002_score


class TestFeatureTransformer:
    """Tests for the sklearn-compatible transformer."""

    def test_fit_transform_output_shape(self):
        """Transformer should produce correct output shape."""
        transformer = FeatureTransformer()
        df = pd.DataFrame({col: np.random.randn(10) for col in FEATURE_COLUMNS})
        result = transformer.fit_transform(df)
        assert result.shape == (10, 17)

    def test_transform_no_nan(self):
        """Transformed output should not contain NaN values."""
        transformer = FeatureTransformer()
        df = pd.DataFrame({col: np.random.randn(10) for col in FEATURE_COLUMNS})
        result = transformer.fit_transform(df)
        assert not np.any(np.isnan(result))

    def test_get_feature_names(self):
        """Should return correct feature names."""
        transformer = FeatureTransformer()
        names = transformer.get_feature_names_out()
        assert names == FEATURE_COLUMNS
