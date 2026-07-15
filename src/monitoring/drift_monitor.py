import os
import json
from src.observability.logger import get_logger

logger = get_logger("monitoring.drift_monitor")

class DriftMonitor:
    """Handles extracting and storing historical drift metrics."""

    def __init__(self, history_path: str = "artifacts/monitoring/drift_history.json"):
        self.history_path = history_path
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

    def log_metrics(self, report_result: dict):
        """Append the latest drift metrics to the history file.
        
        Args:
            report_result: Dictionary returned by ReportGenerator.generate_reports()
        """
        if report_result.get("status") == "skipped":
            return
            
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to read {self.history_path}, starting fresh.")

        metric_entry = {
            "timestamp": report_result["timestamp"],
            "drift_share": report_result["drift_share"],
            "drift_detected": report_result["drift_detected"],
            "drifted_features": report_result["drifted_features"],
        }
        
        history.append(metric_entry)

        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)
            
        logger.info(f"Appended drift metrics to history. Total runs: {len(history)}")
