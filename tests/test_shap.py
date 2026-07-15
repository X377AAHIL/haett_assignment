"""
Tests for the SHAP explainability module.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explainer import ShapExplainer


@pytest.fixture
def mock_data():
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
    })
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def mock_model(mock_data):
    X, y = mock_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def explainer(mock_model):
    return ShapExplainer(model=mock_model)


def test_explainer_initialization(explainer):
    """Test that TreeExplainer is correctly initialized for RandomForest."""
    assert explainer.explainer is not None
    assert type(explainer.explainer).__name__ == "TreeExplainer"


def test_explain_prediction(explainer, mock_data):
    """Test that explain_prediction returns correctly formatted top factors."""
    X, _ = mock_data
    sample = X.iloc[[0]]
    
    top_factors = explainer.explain_prediction(sample, top_k=2)
    
    assert isinstance(top_factors, list)
    assert len(top_factors) == 2
    
    for factor in top_factors:
        assert "feature" in factor
        assert "impact" in factor
        assert "direction" in factor
        assert "description" in factor
        assert factor["direction"] in ["increase_risk", "decrease_risk"]


def test_save_plots(explainer, mock_data, tmp_path):
    """Test that plot generation functions do not crash and produce files."""
    X, _ = mock_data
    shap_values = explainer.get_shap_values(X)
    
    summary_path = tmp_path / "summary.png"
    bar_path = tmp_path / "bar.png"
    waterfall_path = tmp_path / "waterfall.png"
    json_path = tmp_path / "feature_importance.json"
    
    explainer.save_summary_plot(shap_values, str(summary_path))
    explainer.save_bar_plot(shap_values, str(bar_path))
    explainer.save_waterfall_plot(shap_values, 0, str(waterfall_path))
    explainer.save_feature_importance_json(shap_values, str(json_path))
    
    assert os.path.exists(summary_path)
    assert os.path.exists(bar_path)
    assert os.path.exists(waterfall_path)
    assert os.path.exists(json_path)
