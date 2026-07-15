"""
Business Recommendation Engine
===============================
For every user classified as High Risk, recommends one actionable retention
strategy based on their behavioral signals.

The engine analyzes feature values to identify the primary driver of churn
risk and maps it to the most appropriate business action.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Recommendation:
    """A business recommendation for a high-risk user."""
    action: str
    reason: str


# Recommendation rules ordered by priority
# Each rule checks a condition on features and returns a recommendation
RECOMMENDATION_RULES = [
    {
        "name": "high_support_tickets",
        "condition": lambda f: f.get("support_ticket_count", 0) > 5,
        "action": "Prioritize customer support outreach",
        "reason": (
            "This user has raised multiple support tickets, suggesting unresolved issues. "
            "Proactive outreach from a support lead can address frustrations before they leave."
        ),
    },
    {
        "name": "high_value_declining_engagement",
        "condition": lambda f: f.get("avg_order_value", 0) > 100 and f.get("engagement_decline", 0) > 0.3,
        "action": "Offer a 20% discount on the next subscription renewal",
        "reason": (
            "This user has high order value but declining engagement, suggesting dissatisfaction "
            "rather than budget concerns. A targeted discount can re-engage them."
        ),
    },
    {
        "name": "frequent_meal_swaps",
        "condition": lambda f: f.get("meal_swap_frequency", 0) > 0.25,
        "action": "Recommend different meals that better match their dietary preferences",
        "reason": (
            "This user frequently swaps meals, indicating the default selections don't match "
            "their taste. Personalized meal recommendations can improve satisfaction."
        ),
    },
    {
        "name": "declining_ratings",
        "condition": lambda f: f.get("rating_trend", 0) < -0.5,
        "action": "Send a personalized feedback request and offer meal customization",
        "reason": (
            "This user's ratings have been declining over time, suggesting growing dissatisfaction "
            "with meal quality. A direct feedback loop can help address their concerns."
        ),
    },
    {
        "name": "new_user_at_risk",
        "condition": lambda f: f.get("subscription_duration_days", 0) < 60,
        "action": "Suggest a better subscription plan with a trial upgrade",
        "reason": (
            "This is a relatively new user showing early signs of disengagement. "
            "A trial upgrade to a premium plan can demonstrate additional value."
        ),
    },
    {
        "name": "low_engagement",
        "condition": lambda f: f.get("engagement_score", 0) < 3,
        "action": "Send a personalized re-engagement notification with curated content",
        "reason": (
            "This user has very low app engagement. A personalized push notification "
            "highlighting new recipes and features can bring them back to the platform."
        ),
    },
    {
        "name": "subscription_expiring_soon",
        "condition": lambda f: f.get("days_to_subscription_expiry", 999) < 7,
        "action": "Offer an exclusive renewal discount before subscription expires",
        "reason": (
            "This user's subscription is about to expire. An exclusive limited-time "
            "renewal offer can prevent churn at this critical decision point."
        ),
    },
]

# Default recommendation for high-risk users that don't match specific rules
DEFAULT_RECOMMENDATION = Recommendation(
    action="Send a personalized notification highlighting new features and meal options",
    reason=(
        "This user shows general signs of disengagement. A personalized outreach "
        "campaign can re-establish connection with the platform."
    ),
)


def get_recommendation(features: dict, risk_level: str) -> Optional[dict]:
    """Generate a business recommendation based on user features and risk level.

    Args:
        features: Dictionary of user feature values
        risk_level: "Low", "Medium", or "High"

    Returns:
        Dictionary with 'action' and 'reason', or None for low-risk users
    """
    if risk_level == "Low":
        return None

    if risk_level == "Medium":
        return {
            "action": "Monitor closely and send a satisfaction survey",
            "reason": (
                "This user shows moderate churn risk. A satisfaction survey can "
                "identify potential issues early before they escalate."
            ),
        }

    # High risk: apply rules in priority order
    for rule in RECOMMENDATION_RULES:
        if rule["condition"](features):
            return {
                "action": rule["action"],
                "reason": rule["reason"],
            }

    # Fallback
    return {
        "action": DEFAULT_RECOMMENDATION.action,
        "reason": DEFAULT_RECOMMENDATION.reason,
    }
