from fastapi import Request
from fastapi.responses import JSONResponse


class ApplicationError(Exception):
    """Base exception for all custom application errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotLoadedError(ApplicationError):
    def __init__(self, message: str = "Machine learning model not loaded."):
        super().__init__(message, status_code=503)


class PredictionError(ApplicationError):
    def __init__(self, message: str = "Prediction failed."):
        super().__init__(message, status_code=500)


class FeatureValidationError(ApplicationError):
    def __init__(self, message: str = "Invalid feature input."):
        super().__init__(message, status_code=400)


class MonitoringError(ApplicationError):
    def __init__(self, message: str = "Monitoring subsystem error."):
        super().__init__(message, status_code=500)


class ExplainabilityError(ApplicationError):
    def __init__(self, message: str = "Failed to generate explanation."):
        super().__init__(message, status_code=500)


class ConfigurationError(ApplicationError):
    def __init__(self, message: str = "Configuration error."):
        super().__init__(message, status_code=500)


async def global_exception_handler(request: Request, exc: Exception):
    """Handles all generic unhandled exceptions gracefully."""
    # Note: we import the logger locally to avoid circular imports during setup
    from src.observability.logger import get_logger

    logger = get_logger("exception_handler")
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


async def application_error_handler(request: Request, exc: ApplicationError):
    """Handles our domain-specific ApplicationError hierarchy."""
    from src.observability.logger import get_logger

    logger = get_logger("exception_handler")
    logger.error(f"Application error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )
