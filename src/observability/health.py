import os
import yaml
from fastapi import APIRouter, HTTPException
from src.observability.metrics import get_metrics
from src.observability.logger import get_observability_config
import datetime

health_router = APIRouter(tags=["Observability"])

@health_router.get("/health")
async def health_check():
    """Lightweight liveness probe."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@health_router.get("/ready")
async def ready_check():
    """Deep readiness probe verifying critical subsystems."""
    checks = {
        "model_loaded": False,
        "shap_initialized": False,
        "reference_dataset": False,
        "configuration": False,
        "logging": True # Assumed true if we reach here
    }
    
    # Check model and SHAP via api instances if possible, but since we are loosely coupled,
    # we can check global state in api.main directly.
    try:
        from api.main import predictor, explainer
        if predictor is not None:
            checks["model_loaded"] = True
        if explainer is not None:
            checks["shap_initialized"] = True
    except ImportError:
        pass
        
    # Check Reference Dataset
    mon_config_path = "config/monitoring.yaml"
    if os.path.exists(mon_config_path):
        with open(mon_config_path, "r") as f:
            mon_config = yaml.safe_load(f)
            ref_path = mon_config.get("reference_dataset_path")
            if ref_path and os.path.exists(ref_path):
                checks["reference_dataset"] = True

    # Check Configuration
    obs_config = get_observability_config()
    if obs_config:
        checks["configuration"] = True
        
    all_ready = all(checks.values())
    
    response = {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }
    
    if not all_ready:
        # Some orchestrators expect 503 if not ready
        raise HTTPException(status_code=503, detail=response)
        
    return response

@health_router.get("/version")
async def version_info():
    """Return application version metadata."""
    obs_config = get_observability_config()
    
    # Try to get git commit
    git_commit = "unknown"
    if os.path.exists(".git/refs/heads/main"):
        try:
            with open(".git/refs/heads/main", "r") as f:
                git_commit = f.read().strip()[:7]
        except Exception:
            pass

    return {
        "application_version": obs_config.get("application_version", "unknown"),
        "model_version": obs_config.get("model_version", "unknown"),
        "python_version": "3.11",
        "git_commit": git_commit
    }

@health_router.get("/metrics")
async def metrics_info():
    """Return lightweight in-memory metrics."""
    return get_metrics()

@health_router.get("/system/info")
async def system_info():
    """Return a comprehensive summary of the MLOps ecosystem."""
    obs_config = get_observability_config()
    
    # Check for monitoring run
    last_monitoring = "never"
    if os.path.exists("artifacts/monitoring/drift_history.json"):
        import json
        with open("artifacts/monitoring/drift_history.json", "r") as f:
            history = json.load(f)
            if history:
                last_monitoring = history[-1].get("timestamp", "unknown")
                
    return {
        "application_version": obs_config.get("application_version", "unknown"),
        "model_version": obs_config.get("model_version", "unknown"),
        "mlflow_tracking": True,
        "shap_enabled": True,
        "monitoring_enabled": True,
        "ci_pipeline": "GitHub Actions",
        "last_monitoring_run": last_monitoring
    }
