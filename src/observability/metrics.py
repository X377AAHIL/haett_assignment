import time
from contextlib import contextmanager
from typing import Dict, Any

# In-memory metrics state
_metrics_state: Dict[str, Any] = {
    "prediction_count": 0,
    "average_latency": 0.0,
    "last_prediction_latency": 0.0,
    "errors": 0,
    "total_monitoring_runs": 0,
}


def get_metrics() -> Dict[str, Any]:
    """Return a snapshot of current in-memory metrics."""
    return _metrics_state.copy()


def increment_metric(key: str, value: int = 1):
    """Safely increment a counter metric."""
    if key in _metrics_state:
        _metrics_state[key] += value
    else:
        _metrics_state[key] = value


def record_prediction_latency(latency_ms: float):
    """Update prediction count and rolling average latency."""
    _metrics_state["prediction_count"] += 1
    count = _metrics_state["prediction_count"]
    current_avg = _metrics_state["average_latency"]

    # Cumulative moving average
    new_avg = current_avg + (latency_ms - current_avg) / count
    _metrics_state["average_latency"] = new_avg
    _metrics_state["last_prediction_latency"] = latency_ms


@contextmanager
def track_duration(operation_name: str, logger=None):
    """Context manager to track latency of a specific block of code."""
    start_time = time.perf_counter()
    try:
        yield
    except Exception:
        increment_metric("errors")
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        if logger:
            logger.debug(
                f"{operation_name} completed.",
                extra={"latency_ms": round(duration_ms, 2)},
            )
