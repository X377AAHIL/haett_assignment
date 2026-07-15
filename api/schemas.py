"""
Pydantic Schemas for the Haett Churn Prediction API
====================================================
Defines request and response models with validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint.

    Represents a user's historical activity features.
    All fields are required with sensible defaults and validation.
    """
    
    explain: bool = Field(
        False, 
        description="Whether to include SHAP explanations in the response"
    )

    days_since_last_order: float = Field(
        ...,
        ge=0,
        le=999,
        description="Days since the user's last order (0 = ordered today)",
        examples=[15],
    )
    orders_last_30_days: float = Field(
        ...,
        ge=0,
        le=200,
        description="Number of orders placed in the last 30 days",
        examples=[2],
    )
    avg_order_value: float = Field(
        ...,
        ge=0,
        le=10000,
        description="Average order value in INR",
        examples=[450.0],
    )
    subscription_duration_days: float = Field(
        ...,
        ge=0,
        le=3650,
        description="Days since the user first subscribed",
        examples=[120],
    )
    coupon_usage_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of orders where a coupon was used (0.0 to 1.0)",
        examples=[0.3],
    )
    meal_swap_frequency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of orders where a meal was swapped (0.0 to 1.0)",
        examples=[0.1],
    )
    order_consistency: float = Field(
        ...,
        ge=0,
        le=999,
        description="Std deviation of inter-order gaps in days (lower = more consistent)",
        examples=[5.2],
    )
    order_trend_slope: float = Field(
        default=0.0,
        ge=-50,
        le=50,
        description="Linear trend in weekly order counts (negative = declining)",
        examples=[-0.5],
    )
    avg_rating: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="Average rating given by the user (1.0 to 5.0, 0 if no ratings)",
        examples=[4.2],
    )
    rating_trend: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="Change in average rating (recent vs earlier orders)",
        examples=[-0.3],
    )
    engagement_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite engagement score (app opens + recipes viewed + notification clicks)",
        examples=[25.0],
    )
    engagement_decline: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Recent drop in engagement compared to historical average",
        examples=[0.2],
    )
    support_ticket_count: int = Field(
        ...,
        ge=0,
        description="Number of support tickets raised by the user",
        examples=[1],
    )
    total_lifetime_orders: int = Field(
        ...,
        ge=0,
        le=5000,
        description="Total number of orders ever placed",
        examples=[35],
    )
    avg_items_per_order: float = Field(
        ...,
        ge=0,
        le=50,
        description="Average number of items per order",
        examples=[2.5],
    )
    is_premium: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the user is on Premium or Family plan, 0 otherwise",
        examples=[1],
    )
    days_to_subscription_expiry: float = Field(
        ...,
        ge=-365,
        le=365,
        description="Days until subscription expires (negative = already expired)",
        examples=[10],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "explain": True,
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
            ]
        }
    }


class RecommendationResponse(BaseModel):
    """Business recommendation for a user."""
    action: str = Field(..., description="Recommended retention action")
    reason: str = Field(..., description="Why this recommendation is appropriate")


class TopFactor(BaseModel):
    """Explains a single feature's impact on the prediction."""
    feature: str = Field(..., description="The name of the feature")
    impact: float = Field(..., description="The magnitude of the SHAP impact")
    direction: str = Field(..., description="Whether the impact increases or decreases risk (increase_risk, decrease_risk)")
    description: str = Field(..., description="Human-readable description of the feature")


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the user will churn in the next 30 days",
        examples=[0.87],
    )
    risk_level: str = Field(
        ...,
        description="Risk classification: Low, Medium, or High",
        examples=["High"],
    )
    recommendation: Optional[RecommendationResponse] = Field(
        None,
        description="Business recommendation for Medium/High risk users",
    )
    top_factors: Optional[List[TopFactor]] = Field(
        None,
        description="Top factors influencing the prediction (if explain=True)",
    )


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response format."""
    detail: str = Field(..., description="Error message")
