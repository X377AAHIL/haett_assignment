"""
Feature Engineering Module
==========================
Creates meaningful features for churn prediction from raw data.
Implements a scikit-learn compatible transformer for reproducible pipelines.

Features created:
- days_since_last_order: Days from last order to observation date
- orders_last_30_days: Order count in trailing 30-day window
- avg_order_value: Mean order value across all orders
- subscription_duration_days: Days since signup
- coupon_usage_rate: Fraction of orders using coupons
- meal_swap_frequency: Fraction of orders with meal swaps
- order_consistency: Std deviation of inter-order gaps (lower = more consistent)
- order_trend_slope: Linear trend in weekly order counts
- avg_rating: Mean rating given
- rating_trend: Change in avg rating (recent vs earlier)
- engagement_score: Composite engagement metric
- engagement_decline: % drop in engagement (recent vs prior)
- support_ticket_count: Total support tickets
- total_lifetime_orders: Total orders placed
- avg_items_per_order: Average items per order
- is_premium: Binary flag for Premium/Family plan
- days_to_subscription_expiry: Days until subscription expires
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Features are computed as of the prediction date (June 1, 2025)
# All data used for features is BEFORE this date
OBSERVATION_DATE = pd.Timestamp("2025-06-01")
CHURN_WINDOW = 30

# Feature names used for model input
FEATURE_COLUMNS = [
    "days_since_last_order",
    "orders_last_30_days",
    "avg_order_value",
    "subscription_duration_days",
    "coupon_usage_rate",
    "meal_swap_frequency",
    "order_consistency",
    "order_trend_slope",
    "avg_rating",
    "rating_trend",
    "engagement_score",
    "engagement_decline",
    "support_ticket_count",
    "total_lifetime_orders",
    "avg_items_per_order",
    "is_premium",
    "days_to_subscription_expiry",
]


def compute_order_features(user_ids: pd.Series, orders: pd.DataFrame) -> pd.DataFrame:
    """Compute order-related features per user."""
    # Filter orders for our user set
    user_orders = orders[orders["user_id"].isin(user_ids)].copy()
    user_orders = user_orders.sort_values(["user_id", "order_date"])

    # Group by user
    grouped = user_orders.groupby("user_id")

    # Basic aggregations
    order_features = grouped.agg(
        total_lifetime_orders=("order_id", "count"),
        avg_order_value=("order_value", "mean"),
        avg_items_per_order=("items_count", "mean"),
        avg_rating=("rating", "mean"),
        coupon_usage_rate=("coupon_used", "mean"),
        meal_swap_frequency=("meal_swapped", "mean"),
        last_order_date=("order_date", "max"),
    ).reset_index()

    # Days since last order
    order_features["days_since_last_order"] = (
        OBSERVATION_DATE - order_features["last_order_date"]
    ).dt.days

    # Orders in last 30 days
    churn_start = OBSERVATION_DATE - pd.Timedelta(days=CHURN_WINDOW)
    orders_last_30 = (
        user_orders[user_orders["order_date"] >= churn_start]
        .groupby("user_id")
        .size()
        .reset_index(name="orders_last_30_days")
    )
    order_features = order_features.merge(orders_last_30, on="user_id", how="left")
    order_features["orders_last_30_days"] = order_features[
        "orders_last_30_days"
    ].fillna(0)

    # Order consistency (std of inter-order gaps in days)
    def calc_consistency(group):
        if len(group) < 2:
            return 999.0  # High value = inconsistent (only 1 order)
        dates = group["order_date"].sort_values()
        gaps = dates.diff().dt.days.dropna()
        return gaps.std() if len(gaps) > 0 else 999.0

    consistency = user_orders.groupby("user_id").apply(calc_consistency).reset_index()
    consistency.columns = ["user_id", "order_consistency"]
    order_features = order_features.merge(consistency, on="user_id", how="left")

    # Order trend slope (weekly order count linear trend)
    def calc_order_trend(group):
        if len(group) < 4:
            return 0.0
        group = group.copy()
        group["week"] = (group["order_date"] - group["order_date"].min()).dt.days // 7
        weekly_counts = group.groupby("week").size()
        if len(weekly_counts) < 2:
            return 0.0
        x = np.arange(len(weekly_counts))
        y = weekly_counts.values
        slope = np.polyfit(x, y, 1)[0]
        return slope

    trends = user_orders.groupby("user_id").apply(calc_order_trend).reset_index()
    trends.columns = ["user_id", "order_trend_slope"]
    order_features = order_features.merge(trends, on="user_id", how="left")

    # Rating trend (recent half vs first half)
    def calc_rating_trend(group):
        if len(group) < 4:
            return 0.0
        sorted_group = group.sort_values("order_date")
        mid = len(sorted_group) // 2
        early_avg = sorted_group.iloc[:mid]["rating"].mean()
        recent_avg = sorted_group.iloc[mid:]["rating"].mean()
        return recent_avg - early_avg

    rating_trends = (
        user_orders.groupby("user_id").apply(calc_rating_trend).reset_index()
    )
    rating_trends.columns = ["user_id", "rating_trend"]
    order_features = order_features.merge(rating_trends, on="user_id", how="left")

    # Drop intermediate column
    order_features = order_features.drop(columns=["last_order_date"])

    return order_features


def compute_engagement_features(
    user_ids: pd.Series, engagement: pd.DataFrame
) -> pd.DataFrame:
    """Compute engagement-related features per user."""
    user_eng = engagement[engagement["user_id"].isin(user_ids)].copy()

    # Total support tickets
    support = user_eng.groupby("user_id")["support_tickets"].sum().reset_index()
    support.columns = ["user_id", "support_ticket_count"]

    # Engagement score: composite
    user_eng["daily_engagement"] = (
        user_eng["app_opens"]
        + user_eng["recipes_viewed"]
        + user_eng["notification_clicks"]
    )

    eng_score = user_eng.groupby("user_id")["daily_engagement"].mean().reset_index()
    eng_score.columns = ["user_id", "engagement_score"]

    # Engagement decline: compare recent 2 weeks vs prior
    cutoff = OBSERVATION_DATE - pd.Timedelta(days=14)

    recent_eng = (
        user_eng[user_eng["date"] >= cutoff]
        .groupby("user_id")["daily_engagement"]
        .mean()
        .reset_index()
    )
    recent_eng.columns = ["user_id", "recent_engagement"]

    prior_eng = (
        user_eng[user_eng["date"] < cutoff]
        .groupby("user_id")["daily_engagement"]
        .mean()
        .reset_index()
    )
    prior_eng.columns = ["user_id", "prior_engagement"]

    decline = recent_eng.merge(prior_eng, on="user_id", how="left")
    decline["engagement_decline"] = np.where(
        decline["prior_engagement"] > 0,
        (decline["prior_engagement"] - decline["recent_engagement"])
        / decline["prior_engagement"],
        0,
    )
    decline = decline[["user_id", "engagement_decline"]]

    # Merge all engagement features
    eng_features = support.merge(eng_score, on="user_id", how="outer")
    eng_features = eng_features.merge(decline, on="user_id", how="left")

    return eng_features


def compute_subscription_features(
    user_ids: pd.Series, users: pd.DataFrame, subscriptions: pd.DataFrame
) -> pd.DataFrame:
    """Compute subscription-related features."""
    # Subscription duration
    sub_features = users[users["user_id"].isin(user_ids)][
        ["user_id", "signup_date", "subscription_plan"]
    ].copy()

    sub_features["subscription_duration_days"] = (
        OBSERVATION_DATE - sub_features["signup_date"]
    ).dt.days

    # Is premium
    sub_features["is_premium"] = (
        sub_features["subscription_plan"].isin(["Premium", "Family"]).astype(int)
    )

    # Days to subscription expiry
    subs_filtered = subscriptions[subscriptions["user_id"].isin(user_ids)][
        ["user_id", "plan_end"]
    ].copy()
    sub_features = sub_features.merge(subs_filtered, on="user_id", how="left")
    sub_features["days_to_subscription_expiry"] = (
        sub_features["plan_end"] - OBSERVATION_DATE
    ).dt.days
    # Negative means already expired
    sub_features["days_to_subscription_expiry"] = sub_features[
        "days_to_subscription_expiry"
    ].fillna(-30)

    sub_features = sub_features[
        [
            "user_id",
            "subscription_duration_days",
            "is_premium",
            "days_to_subscription_expiry",
        ]
    ]

    return sub_features


def build_features(
    users_df: pd.DataFrame,
    orders: pd.DataFrame,
    subscriptions: pd.DataFrame,
    engagement: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete feature matrix for a set of users.

    Args:
        users_df: DataFrame with user_id and churned columns
        orders: Full orders DataFrame
        subscriptions: Full subscriptions DataFrame
        engagement: Full engagement DataFrame

    Returns:
        DataFrame with user_id, all features, and churned label
    """
    print("Building features...")
    user_ids = users_df["user_id"]

    # Compute feature groups
    order_feats = compute_order_features(user_ids, orders)
    eng_feats = compute_engagement_features(user_ids, engagement)
    sub_feats = compute_subscription_features(user_ids, users_df, subscriptions)

    # Merge all features
    features = users_df[["user_id", "churned"]].copy()
    features = features.merge(order_feats, on="user_id", how="left")
    features = features.merge(eng_feats, on="user_id", how="left")
    features = features.merge(sub_feats, on="user_id", how="left")

    # Fill NaN for users with no orders
    features["days_since_last_order"] = features["days_since_last_order"].fillna(999)
    features["orders_last_30_days"] = features["orders_last_30_days"].fillna(0)
    features["avg_order_value"] = features["avg_order_value"].fillna(0)
    features["total_lifetime_orders"] = features["total_lifetime_orders"].fillna(0)
    features["avg_items_per_order"] = features["avg_items_per_order"].fillna(0)
    features["avg_rating"] = features["avg_rating"].fillna(0)
    features["coupon_usage_rate"] = features["coupon_usage_rate"].fillna(0)
    features["meal_swap_frequency"] = features["meal_swap_frequency"].fillna(0)
    features["order_consistency"] = features["order_consistency"].fillna(999)
    features["order_trend_slope"] = features["order_trend_slope"].fillna(0)
    features["rating_trend"] = features["rating_trend"].fillna(0)
    features["engagement_score"] = features["engagement_score"].fillna(0)
    features["engagement_decline"] = features["engagement_decline"].fillna(0)
    features["support_ticket_count"] = features["support_ticket_count"].fillna(0)

    print(
        f"  Feature matrix: {features.shape[0]} users × {len(FEATURE_COLUMNS)} features"
    )
    return features


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature transformer.

    Scales numeric features using StandardScaler.
    Can be used in sklearn Pipelines for reproducible feature transformation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the scaler on training data."""
        self.scaler.fit(X[self.feature_columns])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        return self.scaler.transform(X[self.feature_columns])

    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        return self.feature_columns
