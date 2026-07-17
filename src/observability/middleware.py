import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from src.observability.logger import request_id_var, correlation_id_var, get_logger

logger = get_logger("middleware")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate tracing IDs
        req_id = str(uuid.uuid4())
        corr_id_header = request.headers.get("X-Correlation-ID")

        # Validate correlation ID: max 100 chars, printable characters only
        if (
            corr_id_header
            and len(corr_id_header) <= 100
            and corr_id_header.isprintable()
        ):
            corr_id = corr_id_header
        else:
            corr_id = req_id

        # Set context variables
        request_id_var.set(req_id)
        correlation_id_var.set(corr_id)

        # Store in request state for convenience if needed
        request.state.request_id = req_id
        request.state.correlation_id = corr_id

        start_time = time.perf_counter()

        # Log incoming request
        logger.info(
            "request_received",
            extra={
                "event": "request_received",
                "method": request.method,
                "endpoint": request.url.path,
            },
        )

        # Process request
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Correlation-ID"] = corr_id
            status_code = response.status_code
        except Exception as exc:
            # If an unhandled exception bubbles up to here
            status_code = 500
            raise exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log completed request
            logger.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                },
            )

        return response
