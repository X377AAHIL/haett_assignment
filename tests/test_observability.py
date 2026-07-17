from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint():
    # It might return 200 or 503 depending on if model is actually loaded in test env.
    # We just want to ensure it doesn't crash 500.
    response = client.get("/ready")
    data = response.json()
    if response.status_code == 503:
        data = data["detail"]
    assert "status" in data
    assert "checks" in data


def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "application_version" in data
    assert "model_version" in data
    assert "python_version" in data
    assert "git_commit" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_count" in data
    assert "average_latency" in data


def test_system_info_endpoint():
    response = client.get("/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "application_version" in data
    assert data["mlflow_tracking"] is True
    assert data["monitoring_enabled"] is True


def test_middleware_injects_headers():
    # Make a request to health which is lightweight
    response = client.get("/health", headers={"X-Correlation-ID": "test-corr-id"})
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == "test-corr-id"


def test_exception_handling():
    # Send a bad prediction request to trigger the global application error handler
    # We need to send an empty dict to trigger validation error or prediction error
    response = client.post("/predict", json={})
    # Might be 422 (FastAPI validation) or 500 (Prediction error)
    # Let's ensure it handles it gracefully
    assert response.status_code in [422, 503, 500]
