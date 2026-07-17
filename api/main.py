"""
Haett Churn Prediction API
==========================
FastAPI application that serves churn predictions.

Endpoints:
- POST /predict: Predict churn probability and risk level for a user
- GET  /health:  Health check endpoint
- GET  /docs:    Auto-generated API documentation (Swagger UI)
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.observability.logger import get_logger
from src.observability.middleware import ObservabilityMiddleware
from src.observability.health import health_router
from src.observability.exceptions import (
    ApplicationError,
    ModelNotLoadedError,
    PredictionError,
    global_exception_handler,
    application_error_handler,
)
from src.observability.metrics import track_duration, record_prediction_latency

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
    VersionResponse,
)
from src.predict import ChurnPredictor
from src.explainability import ShapExplainer
from src.monitoring.prediction_logger import PredictionLogger
import pandas as pd
import os
import json
import time

# Logger setup
logger = get_logger("api")

# Global predictor and explainer instances
predictor: Optional[ChurnPredictor] = None
explainer: Optional[ShapExplainer] = None
prediction_logger: Optional[PredictionLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global predictor, explainer, prediction_logger

    logger.info(
        "application_started",
        extra={
            "event": "application_started",
            "python": "3.11",
            "mlflow_tracking": "enabled",
        },
    )

    try:
        predictor = ChurnPredictor()
        prediction_logger = PredictionLogger()
        logger.info("✅ Model loaded successfully")

        # Initialize explainer for the API (no background data for Tree models)
        try:
            explainer = ShapExplainer(model=predictor.model)
            logger.info("✅ ShapExplainer initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ ShapExplainer initialization failed: {e}")
            explainer = None

    except (FileNotFoundError, ModelNotLoadedError) as e:
        logger.warning(f"Model not found: {e}. Train the model first.")
        predictor = None
        explainer = None
    finally:
        logger.info("application_shutdown", extra={"event": "application_shutdown"})
        logger.info("Shutting down server", extra={"event": "server_stopping"})
    yield


app = FastAPI(
    title="Haett Churn Prediction API",
    description=(
        "Predicts whether an active user on the Haett meal delivery platform "
        "is likely to churn within the next 30 days. Returns churn probability, "
        "risk level (Low/Medium/High), and actionable business recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
    },
)

# Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(ApplicationError, application_error_handler)

# Middleware
app.add_middleware(ObservabilityMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is running and the model is loaded."""
    from src.version import application_version

    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version=application_version,
    )


@app.get("/version", response_model=VersionResponse, tags=["System"])
async def version_check():
    """Get system version information."""
    from src.version import application_version, model_version, build_date, git_commit

    return VersionResponse(
        application_version=application_version,
        model_version=model_version,
        build_date=build_date,
        git_commit=git_commit,
    )


app.include_router(health_router)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict churn probability",
    response_description="Churn prediction with risk level and recommendation",
)
async def predict_churn(request: PredictionRequest):
    """Predict whether a user will churn within the next 30 days.

    Takes a user's historical activity features as input and returns:
    - **churn_probability**: Float between 0 and 1
    - **risk_level**: "Low" (< 0.3), "Medium" (0.3-0.6), or "High" (> 0.6)
    - **recommendation**: Actionable business recommendation (for Medium/High risk)
    """
    if predictor is None:
        raise ModelNotLoadedError()

    start_predict = time.perf_counter()
    try:
        # Convert request to feature dictionary
        features = request.model_dump()

        # Get prediction
        result = predictor.predict(features)

        # Build response
        recommendation = None
        if result.get("recommendation"):
            recommendation = RecommendationResponse(
                action=result["recommendation"]["action"],
                reason=result["recommendation"]["reason"],
            )

        top_factors = None
        if request.explain:
            if explainer is None:
                logger.warning(
                    "Explanation requested but explainer is not initialized."
                )
            else:
                try:
                    # Prepare and scale features identically to predict()
                    df = pd.DataFrame([features])
                    for col in predictor.feature_columns:
                        if col not in df.columns:
                            df[col] = 0
                    df = df[predictor.feature_columns]
                    X_scaled = predictor.transformer.scaler.transform(df)
                    X_scaled_df = pd.DataFrame(
                        X_scaled, columns=predictor.feature_columns
                    )

                    with track_duration("shap_explainability", logger):
                        top_factors = explainer.explain_prediction(X_scaled_df, top_k=5)
                except Exception as e:
                    logger.error(f"Error computing explain_prediction in API: {e}")

        # Record metrics
        latency_ms = (time.perf_counter() - start_predict) * 1000
        record_prediction_latency(latency_ms)

        # Log prediction to production dataset
        if prediction_logger:
            # Drop 'explain' flag since we only want features
            log_features = {k: v for k, v in features.items() if k != "explain"}
            prediction_logger.log(
                features=log_features,
                prediction_probability=result["churn_probability"],
                prediction_class=result["risk_level"],
            )

        return PredictionResponse(
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            recommendation=recommendation,
            top_factors=top_factors,
        )

    except ApplicationError:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise PredictionError(f"Prediction failed: {str(e)}")


@app.get("/monitor/status", tags=["Monitoring"])
async def monitor_status():
    """Return a summary of the current monitoring status."""
    try:
        from src.monitoring.report_generator import ReportGenerator

        generator = ReportGenerator()
        ref_samples = 0
        prod_samples = 0

        if os.path.exists(generator.reference_path):
            ref_samples = len(pd.read_parquet(generator.reference_path))
        if os.path.exists(generator.production_path):
            prod_samples = len(pd.read_parquet(generator.production_path))

        history = []
        if os.path.exists("artifacts/monitoring/drift_history.json"):
            with open("artifacts/monitoring/drift_history.json", "r") as f:
                history = json.load(f)

        latest_run = history[-1] if history else None

        return {
            "reference_samples": ref_samples,
            "production_samples": prod_samples,
            "last_monitoring_run": latest_run["timestamp"] if latest_run else None,
            "drift_detected": latest_run["drift_detected"] if latest_run else False,
            "total_monitoring_runs": len(history),
        }
    except Exception as e:
        logger.error(f"Monitoring status error: {e}")
        from src.observability.exceptions import MonitoringError

        raise MonitoringError(str(e))


@app.post("/monitor/drift", tags=["Monitoring"])
async def trigger_drift_report():
    """Manually trigger drift monitoring and report generation."""
    try:
        from src.monitoring.report_generator import ReportGenerator
        from src.monitoring.drift_monitor import DriftMonitor

        generator = ReportGenerator()
        monitor = DriftMonitor()

        with track_duration("drift_report_generation", logger):
            result = generator.generate_reports()
        monitor.log_metrics(result)

        return result
    except Exception as e:
        logger.error(f"Drift generation error: {e}")
        from src.observability.exceptions import MonitoringError

        raise MonitoringError(str(e))


if __name__ == "__main__":
    import uvicorn

    reload_mode = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=reload_mode)
