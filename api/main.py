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

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global predictor
    try:
        predictor = ChurnPredictor()
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Model not found: {e}. Train the model first.")
        predictor = None
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

        return PredictionResponse(
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
