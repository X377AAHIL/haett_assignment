"""
SHAP Explainability Module
==========================
Provides model-agnostic SHAP explainability for churn prediction models.
Includes functionality to compute feature importances, plot visualizations,
and extract top factors for API consumption.
"""

import os
import json
import logging
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Feature descriptions for API
FEATURE_DESCRIPTIONS = {
    "days_since_last_order": "Recency of the user's last order.",
    "orders_last_30_days": "Number of orders placed recently.",
    "total_lifetime_orders": "Total number of historical orders.",
    "avg_order_value": "Average monetary value per order.",
    "avg_items_per_order": "Average number of items in each order.",
    "meal_swap_frequency": "How often the user changes default meal selections.",
    "coupon_usage_rate": "Frequency of using discounts or coupons.",
    "order_consistency": "Regularity of ordering patterns over time.",
    "order_trend_slope": "Trend of ordering frequency (increasing or decreasing).",
    "rating_trend": "Trend of user satisfaction ratings.",
    "avg_rating": "Average satisfaction rating given by the user.",
    "subscription_duration_days": "Total days since the user subscribed.",
    "days_to_subscription_expiry": "Days remaining until the subscription ends.",
    "is_premium": "Whether the user is on a premium plan.",
    "engagement_score": "Overall app engagement activity.",
    "engagement_decline": "Drop in app engagement compared to historical average.",
    "support_ticket_count": "Number of customer support tickets raised."
}


class ShapExplainer:
    """Wrapper for SHAP explainability."""

    def __init__(self, model: Any, background_data: pd.DataFrame = None):
        """Initializes the appropriate SHAP explainer based on model type.
        
        Args:
            model: The trained ML model.
            background_data: Optional background dataset (required for Linear/Kernel explainers).
        """
        self.model = model
        self.explainer = self._create_explainer(model, background_data)
        logger.info(f"Initialized {self.explainer.__class__.__name__} successfully.")

    def _create_explainer(self, model: Any, background_data: pd.DataFrame) -> shap.Explainer:
        """Detects model type and creates the most efficient explainer."""
        model_name = type(model).__name__.lower()
        
        try:
            if any(tree_type in model_name for tree_type in ["xgb", "lgbm", "randomforest", "tree"]):
                return shap.TreeExplainer(model)
            elif "logistic" in model_name or "linear" in model_name:
                if background_data is None:
                    raise ValueError("Background data is required for Linear models.")
                return shap.LinearExplainer(model, background_data)
            else:
                logger.warning(f"Unsupported model {model_name}, falling back to KernelExplainer.")
                if background_data is None:
                    raise ValueError("Background data is required for Kernel models.")
                # Use a small sample for KernelExplainer to avoid massive slowdowns
                sample = shap.sample(background_data, 50)
                return shap.KernelExplainer(model.predict_proba, sample)
        except Exception as e:
            logger.error(f"Failed to create explainer for model {model_name}: {e}")
            raise

    def get_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """Computes SHAP values safely handling different explainer output shapes."""
        logger.info(f"Generating SHAP values for {len(X)} samples...")
        shap_values = self.explainer(X)
        
        # Handle multi-class / probability output formats from TreeExplainer
        if len(shap_values.shape) > 2:
            # Usually index 1 is the positive class (Churn)
            return shap_values[:, :, 1]
        return shap_values

    def explain_prediction(self, X: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """Calculates SHAP values for a single prediction and returns top impacting factors.
        
        Args:
            X: A single-row DataFrame.
            top_k: Number of top factors to return.
            
        Returns:
            List of dictionaries with feature, impact, direction, and description.
        """
        try:
            shap_values = self.get_shap_values(X)
            
            # For a single prediction
            sv = shap_values.values[0]
            feature_names = X.columns.tolist()
            
            # Sort by absolute SHAP value
            top_indices = np.argsort(np.abs(sv))[::-1][:top_k]
            
            top_factors = []
            for idx in top_indices:
                impact = float(sv[idx])
                direction = "increase_risk" if impact > 0 else "decrease_risk"
                feature = feature_names[idx]
                
                top_factors.append({
                    "feature": feature,
                    "impact": round(impact, 4),
                    "direction": direction,
                    "description": FEATURE_DESCRIPTIONS.get(feature, f"Impact of {feature}.")
                })
            return top_factors
        except Exception as e:
            logger.exception(f"Error computing explain_prediction: {e}")
            # Ensure prediction API doesn't crash by returning empty explanation
            return []

    def save_summary_plot(self, shap_values: shap.Explanation, path: str):
        """Generates and saves a SHAP summary plot."""
        logger.info(f"Saving summary plot to {path}...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_bar_plot(self, shap_values: shap.Explanation, path: str):
        """Generates and saves a SHAP feature importance bar plot."""
        logger.info(f"Saving feature importance bar plot to {path}...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_waterfall_plot(self, shap_values: shap.Explanation, sample_index: int, path: str):
        """Generates and saves a waterfall plot for a specific sample index."""
        logger.info(f"Saving waterfall plot for sample {sample_index} to {path}...")
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[sample_index], show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_feature_importance_json(self, shap_values: shap.Explanation, path: str):
        """Calculates global mean absolute SHAP values and saves as JSON."""
        logger.info(f"Saving feature importance JSON to {path}...")
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_names = shap_values.feature_names
        
        # Sort descending
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        importance_list = []
        for idx in sorted_indices:
            importance_list.append({
                "feature": feature_names[idx],
                "importance": float(round(mean_abs_shap[idx], 4))
            })
            
        with open(path, "w") as f:
            json.dump(importance_list, f, indent=2)
