"""
Synthetic Data Generator for Haett Churn Prediction
====================================================
Generates realistic synthetic data simulating a healthy meal delivery platform.

TEMPORAL DESIGN (prevents data leakage):
- History period: Jan 1 – May 31, 2025 (5 months of features)
- Prediction point: June 1, 2025
- Churn window: June 1 – June 30, 2025 (the "next 30 days" we predict)
- Features are computed ONLY from data before June 1
- The churn label reflects behavior AFTER June 1

KEY DESIGN: Realistic noise and overlap
- Churn is PROBABILISTIC, not deterministic from behavior
- Many churned users look similar to loyal users in features
- Some loyal users show declining patterns but stay
- This ensures models achieve realistic (non-perfect) accuracy
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility
SEED = 42
np.random.seed(SEED)

# --- Configuration ---
NUM_USERS = 5000
PREDICTION_DATE = datetime(2025, 6, 1)
OBSERVATION_END = datetime(2025, 6, 30)
HISTORY_START = datetime(2025, 1, 1)
TARGET_CHURN_RATE = 0.27

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

SUBSCRIPTION_PLANS = ["Basic", "Premium", "Family"]
PLAN_PRICES = {"Basic": 499, "Premium": 899, "Family": 1299}
DIETARY_PREFERENCES = [
    "Vegan",
    "Vegetarian",
    "Keto",
    "Paleo",
    "Balanced",
    "High-Protein",
]
FITNESS_GOALS = ["Weight Loss", "Muscle Gain", "Maintenance", "General Health"]
CITIES = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
]


def generate_users(num_users: int) -> pd.DataFrame:
    """Generate user profiles.

    Churn propensity is assigned as a latent score based on multiple weak factors.
    This ensures significant overlap between churned/loyal users in observed features.
    """
    print(f"  Generating {num_users} user profiles...")

    signup_dates = [
        HISTORY_START + timedelta(days=int(np.random.exponential(scale=45)))
        for _ in range(num_users)
    ]
    signup_dates = [min(d, PREDICTION_DATE - timedelta(days=14)) for d in signup_dates]

    plans = np.random.choice(SUBSCRIPTION_PLANS, size=num_users, p=[0.45, 0.35, 0.20])

    users = pd.DataFrame(
        {
            "user_id": [f"U{str(i).zfill(5)}" for i in range(1, num_users + 1)],
            "signup_date": signup_dates,
            "subscription_plan": plans,
            "dietary_preference": np.random.choice(DIETARY_PREFERENCES, size=num_users),
            "fitness_goal": np.random.choice(FITNESS_GOALS, size=num_users),
            "age": np.random.randint(22, 55, size=num_users),
            "city": np.random.choice(CITIES, size=num_users),
        }
    )

    # Compute latent churn propensity from WEAK signals
    # Each factor contributes a little, but none is deterministic
    churn_score = np.zeros(num_users)

    # Factor 1: Basic plan users slightly more likely to churn
    churn_score += np.where(plans == "Basic", 0.08, -0.04)

    # Factor 2: Very new users slightly more likely to churn
    days_active = [(PREDICTION_DATE - d).days for d in signup_dates]
    churn_score += np.where(np.array(days_active) < 45, 0.06, -0.02)

    # Factor 3: Random individual propensity (dominates! makes it realistic)
    churn_score += np.random.normal(0, 0.3, size=num_users)

    # Convert to probability and sample churn
    churn_prob = 1 / (1 + np.exp(-churn_score * 2))
    # Adjust to hit target churn rate
    threshold = np.percentile(churn_prob, (1 - TARGET_CHURN_RATE) * 100)
    is_churned = (churn_prob >= threshold).astype(int)

    # Store the propensity for behavior generation (higher = more likely to churn)
    users["is_churned"] = is_churned
    users["_churn_propensity"] = np.clip(churn_prob, 0.05, 0.95)

    return users


def generate_orders(users: pd.DataFrame) -> pd.DataFrame:
    """Generate order history with OVERLAPPING behavioral patterns.

    Key design: churned users tend to have slightly different patterns,
    but with LARGE variance so there's significant overlap with loyal users.
    """
    print("  Generating order history...")
    all_orders = []
    order_id = 1

    for _, user in users.iterrows():
        uid = user["user_id"]
        signup = user["signup_date"]
        is_churned = user["is_churned"]
        plan = user["subscription_plan"]

        # Base frequency varies by plan AND by individual (high variance)
        plan_base = {"Basic": 3.0, "Premium": 4.0, "Family": 5.0}[plan]
        # Individual variation is LARGE (dominates plan effect)
        individual_freq = plan_base * np.random.uniform(0.5, 1.5)

        # Base order value with individual variation
        base_value = PLAN_PRICES[plan] * np.random.uniform(0.08, 0.2)

        # Churned users: SLIGHT decline factor (0.7-1.0), not dramatic
        # Loyal users: stable or slight increase (0.9-1.1)
        # Some churned users show NO decline (they churn suddenly)
        if is_churned:
            # 40% of churners show decline, 60% look similar to loyal users
            shows_decline = np.random.random() < 0.4
            decline_strength = np.random.uniform(0.15, 0.4) if shows_decline else 0.0
        else:
            # 10% of loyal users show some decline (but stay)
            shows_decline = np.random.random() < 0.10
            decline_strength = np.random.uniform(0.05, 0.15) if shows_decline else 0.0

        # Individual rating baseline (large variance between users)
        if is_churned:
            base_rating = np.random.uniform(3.0, 4.5)  # Wide range, overlaps with loyal
        else:
            base_rating = np.random.uniform(
                3.3, 4.8
            )  # Wide range, overlaps with churned

        # Individual swap/coupon rates (large variance)
        base_swap_rate = np.random.uniform(0.05, 0.35)  # Same range for both
        if is_churned:
            base_swap_rate += np.random.uniform(0, 0.1)  # Tiny nudge
        base_coupon_rate = np.random.uniform(0.15, 0.45)
        if is_churned:
            base_coupon_rate -= np.random.uniform(0, 0.08)

        # --- Generate Feature Period Orders (Jan - May) ---
        current_date = signup
        total_feature_days = (PREDICTION_DATE - signup).days
        week = 0
        total_weeks = max(total_feature_days // 7, 1)

        while current_date < PREDICTION_DATE:
            week += 1
            week_progress = week / total_weeks

            # Weekly order count with decline
            freq_multiplier = max(0.3, 1 - week_progress * decline_strength)
            weekly_orders = int(
                np.random.poisson(max(0.5, individual_freq * freq_multiplier))
            )

            for _ in range(weekly_orders):
                order_date = current_date + timedelta(
                    days=np.random.randint(0, 7),
                    hours=np.random.randint(7, 22),
                    minutes=np.random.randint(0, 60),
                )
                if order_date >= PREDICTION_DATE:
                    break

                # Order value with natural day-to-day variation
                value = base_value * np.random.uniform(0.7, 1.5)

                # Swap and coupon with per-order randomness
                meal_swapped = int(np.random.random() < base_swap_rate)
                coupon_used = int(np.random.random() < base_coupon_rate)

                # Rating with per-order noise
                rating_noise = np.random.normal(0, 0.6)
                rating = round(np.clip(base_rating + rating_noise, 1.0, 5.0), 1)

                items_count = np.random.randint(1, 5)

                all_orders.append(
                    {
                        "order_id": f"ORD{str(order_id).zfill(7)}",
                        "user_id": uid,
                        "order_date": order_date,
                        "order_value": round(value, 2),
                        "items_count": items_count,
                        "meal_swapped": meal_swapped,
                        "coupon_used": coupon_used,
                        "rating": rating,
                    }
                )
                order_id += 1

            current_date += timedelta(days=7)

        # --- Churn Window Orders (June 1-30) ---
        if is_churned:
            # Churned users: most have 0 orders, ~20% have 1-3 early orders
            if np.random.random() < 0.20:
                num_june = np.random.randint(1, 4)
                for _ in range(num_june):
                    od = PREDICTION_DATE + timedelta(days=np.random.randint(0, 10))
                    all_orders.append(
                        {
                            "order_id": f"ORD{str(order_id).zfill(7)}",
                            "user_id": uid,
                            "order_date": od,
                            "order_value": round(
                                base_value * np.random.uniform(0.5, 1.0), 2
                            ),
                            "items_count": np.random.randint(1, 3),
                            "meal_swapped": int(np.random.random() < base_swap_rate),
                            "coupon_used": 0,
                            "rating": round(
                                np.clip(np.random.normal(3.0, 0.8), 1.0, 5.0), 1
                            ),
                        }
                    )
                    order_id += 1
        else:
            # Loyal users continue ordering in June
            for w in range(4):
                wo = int(np.random.poisson(individual_freq))
                for _ in range(wo):
                    od = PREDICTION_DATE + timedelta(
                        days=w * 7 + np.random.randint(0, 7),
                        hours=np.random.randint(7, 22),
                    )
                    if od > OBSERVATION_END:
                        break
                    all_orders.append(
                        {
                            "order_id": f"ORD{str(order_id).zfill(7)}",
                            "user_id": uid,
                            "order_date": od,
                            "order_value": round(
                                base_value * np.random.uniform(0.8, 1.4), 2
                            ),
                            "items_count": np.random.randint(1, 5),
                            "meal_swapped": int(
                                np.random.random() < base_swap_rate * 0.8
                            ),
                            "coupon_used": int(np.random.random() < base_coupon_rate),
                            "rating": round(
                                np.clip(
                                    base_rating + np.random.normal(0, 0.5), 1.0, 5.0
                                ),
                                1,
                            ),
                        }
                    )
                    order_id += 1

    orders = pd.DataFrame(all_orders)
    print(f"  Generated {len(orders):,} total orders")
    fp = len(orders[orders["order_date"] < PREDICTION_DATE])
    cw = len(orders[orders["order_date"] >= PREDICTION_DATE])
    print(f"    Feature period: {fp:,} | Churn window: {cw:,}")
    return orders


def generate_subscriptions(users: pd.DataFrame) -> pd.DataFrame:
    """Generate subscription records.

    CRITICAL: Subscription end dates are generated INDEPENDENTLY of churn status.
    Both churned and loyal users have realistic subscription cycles.
    The only difference is whether is_renewed = 1 (decided after June).
    """
    print("  Generating subscription records...")
    subs = []

    for _, user in users.iterrows():
        uid = user["user_id"]
        signup = user["signup_date"]
        plan = user["subscription_plan"]
        is_churned = user["is_churned"]

        duration_months = int(np.random.choice([1, 3], p=[0.6, 0.4]))
        plan_start = signup
        plan_end = plan_start + timedelta(days=duration_months * 30)

        # Simulate renewal cycles up to around the prediction date
        # This is IDENTICAL for both churned and loyal users
        while plan_end < PREDICTION_DATE - timedelta(days=5):
            plan_start = plan_end
            plan_end = plan_start + timedelta(days=duration_months * 30)

        # Now plan_end is somewhere around or after the prediction date
        # for BOTH groups. The only difference is is_renewed (post-hoc).
        is_renewed = 0 if is_churned else 1

        subs.append(
            {
                "user_id": uid,
                "plan_start": plan_start,
                "plan_end": plan_end,
                "is_renewed": is_renewed,
                "plan_type": plan,
                "monthly_price": PLAN_PRICES[plan],
            }
        )

    return pd.DataFrame(subs)


def generate_engagement(users: pd.DataFrame) -> pd.DataFrame:
    """Generate engagement data with OVERLAPPING patterns.

    Both churned and loyal users have similar engagement distributions,
    with only subtle differences on average.
    """
    print("  Generating engagement data...")
    records = []

    for _, user in users.iterrows():
        uid = user["user_id"]
        signup = user["signup_date"]
        is_churned = user["is_churned"]

        # Individual baseline engagement (large variance, overlapping ranges)
        if is_churned:
            base_app = np.random.uniform(1.5, 4.5)
            base_recipe = np.random.uniform(1.0, 3.0)
            base_notif = np.random.uniform(0.5, 2.0)
            base_support = np.random.uniform(0.02, 0.08)
        else:
            base_app = np.random.uniform(2.0, 5.0)
            base_recipe = np.random.uniform(1.0, 3.5)
            base_notif = np.random.uniform(0.8, 2.5)
            base_support = np.random.uniform(0.01, 0.05)

        # Engagement decline (subtle for churned, none for most loyal)
        if is_churned and np.random.random() < 0.35:
            eng_decline = np.random.uniform(0.1, 0.3)
        elif not is_churned and np.random.random() < 0.05:
            eng_decline = np.random.uniform(0.05, 0.15)
        else:
            eng_decline = 0.0

        total_days = (PREDICTION_DATE - signup).days
        current_date = signup
        day_idx = 0

        while current_date < PREDICTION_DATE:
            day_idx += 1
            progress = day_idx / max(total_days, 1)

            decay = max(0.3, 1 - progress * eng_decline)

            app_opens = max(0, int(np.random.poisson(max(0.3, base_app * decay))))
            recipes = max(0, int(np.random.poisson(max(0.2, base_recipe * decay))))
            notifs = max(0, int(np.random.poisson(max(0.1, base_notif * decay))))
            tickets = int(
                np.random.poisson(base_support * (1 + progress * eng_decline * 2))
            )

            records.append(
                {
                    "user_id": uid,
                    "date": current_date,
                    "app_opens": app_opens,
                    "recipes_viewed": recipes,
                    "support_tickets": tickets,
                    "notification_clicks": notifs,
                }
            )

            current_date += timedelta(days=3)

    engagement = pd.DataFrame(records)
    print(f"  Generated {len(engagement):,} engagement records")
    return engagement


def main():
    """Generate all synthetic datasets and save to CSV."""
    print("=" * 60)
    print("Haett Synthetic Data Generator")
    print("=" * 60)
    print(f"  Feature period:  {HISTORY_START.date()} to {PREDICTION_DATE.date()}")
    print(f"  Churn window:    {PREDICTION_DATE.date()} to {OBSERVATION_END.date()}")

    os.makedirs(RAW_DIR, exist_ok=True)

    users = generate_users(NUM_USERS)
    orders = generate_orders(users)
    subscriptions = generate_subscriptions(users)
    engagement = generate_engagement(users)

    # Drop internal column before saving
    users_save = users.drop(columns=["_churn_propensity"])
    users_save.to_csv(os.path.join(RAW_DIR, "users.csv"), index=False)
    orders.to_csv(os.path.join(RAW_DIR, "orders.csv"), index=False)
    subscriptions.to_csv(os.path.join(RAW_DIR, "subscriptions.csv"), index=False)
    engagement.to_csv(os.path.join(RAW_DIR, "engagement.csv"), index=False)

    print("\n" + "=" * 60)
    print("Data Generation Summary")
    print("=" * 60)
    print(
        f"Users:         {len(users):,} ({users['is_churned'].mean():.1%} will churn)"
    )
    print(f"Orders:        {len(orders):,}")
    print(f"Subscriptions: {len(subscriptions):,}")
    print(f"Engagement:    {len(engagement):,}")
    print(f"\nFiles saved to: {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
