import os
import pytest
from fastapi.testclient import TestClient

from src.monitoring.prediction_logger import PredictionLogger
from src.monitoring.drift_monitor import DriftMonitor


@pytest.fixture
def temp_logger(tmp_path):
    path = str(tmp_path / "prod.parquet")
    return PredictionLogger(production_path=path)


def test_prediction_logger_appends_data(temp_logger):
    """Test that the prediction logger correctly creates and appends to parquet."""
    features = {"avg_order_value": 45.0, "days_since_last_order": 2}

    # First log creates the file
    temp_logger.log(features, 0.85, "High")
    assert os.path.exists(temp_logger.production_path)

    # Second log appends
    temp_logger.log(features, 0.15, "Low")

    import pandas as pd

    df = pd.read_parquet(temp_logger.production_path)
    assert len(df) == 2
    assert "prediction_probability" in df.columns
    assert "model_version" in df.columns
    assert "request_id" in df.columns


def test_drift_monitor_history(tmp_path):
    """Test that the drift monitor appends metrics correctly."""
    history_path = str(tmp_path / "history.json")
    monitor = DriftMonitor(history_path=history_path)

    mock_result = {
        "status": "success",
        "timestamp": "2026-07-16_000000",
        "drift_share": 0.25,
        "drift_detected": False,
        "drifted_features": 2,
    }

    # First append
    monitor.log_metrics(mock_result)
    assert os.path.exists(history_path)

    # Second append
    monitor.log_metrics(mock_result)

    import json

    with open(history_path, "r") as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["drift_share"] == 0.25


# For API tests, we conditionally test if the API is available
try:
    from api.main import app

    client = TestClient(app)
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
def test_monitor_status_endpoint():
    """Test the GET /monitor/status endpoint."""
    response = client.get("/monitor/status")
    assert response.status_code == 200
    data = response.json()
    assert "reference_samples" in data
    assert "production_samples" in data
    assert "drift_detected" in data
