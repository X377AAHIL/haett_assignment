"""
Haett Churn Prediction API
==========================
FastAPI application that serves churn predictions.

Endpoints:
- POST /predict: Predict churn probability and risk level for a user
- GET  /health:  Health check endpoint
- GET  /docs:    Auto-generated API documentation (Swagger UI)
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from src.predict import ChurnPredictor
from src.explainability import ShapExplainer
from src.monitoring.prediction_logger import PredictionLogger
from src.monitoring.report_generator import ReportGenerator
from src.monitoring.drift_monitor import DriftMonitor
import pandas as pd
import os
import json

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor and explainer instances
predictor: Optional[ChurnPredictor] = None
explainer: Optional[ShapExplainer] = None
prediction_logger: Optional[PredictionLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global predictor, explainer, prediction_logger
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
            
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Model not found: {e}. Train the model first.")
        predictor = None
        explainer = None
    yield
    logger.info("Shutting down...")


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is running and the model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0",
    )


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
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: python -m src.model_training",
        )

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
                logger.warning("Explanation requested but explainer is not initialized.")
            else:
                try:
                    # Prepare and scale features identically to predict()
                    df = pd.DataFrame([features])
                    for col in predictor.feature_columns:
                        if col not in df.columns:
                            df[col] = 0
                    df = df[predictor.feature_columns]
                    X_scaled = predictor.transformer.scaler.transform(df)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=predictor.feature_columns)
                    
                    top_factors = explainer.explain_prediction(X_scaled_df, top_k=5)
                except Exception as e:
                    logger.error(f"Error computing explain_prediction in API: {e}")

        # Log prediction to production dataset
        if prediction_logger:
            # Drop 'explain' flag since we only want features
            log_features = {k: v for k, v in features.items() if k != "explain"}
            prediction_logger.log(
                features=log_features,
                prediction_probability=result["churn_probability"],
                prediction_class=result["risk_level"]
            )

        return PredictionResponse(
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            recommendation=recommendation,
            top_factors=top_factors,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/monitor/status", tags=["Monitoring"])
async def monitor_status():
    """Return a summary of the current monitoring status."""
    try:
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
            "total_monitoring_runs": len(history)
        }
    except Exception as e:
        logger.error(f"Monitoring status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/drift", tags=["Monitoring"])
async def trigger_drift_report():
    """Manually trigger drift monitoring and report generation."""
    try:
        generator = ReportGenerator()
        monitor = DriftMonitor()
        
        result = generator.generate_reports()
        monitor.log_metrics(result)
        
        return result
    except Exception as e:
        logger.error(f"Drift generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
