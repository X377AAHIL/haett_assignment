import os
import yaml
import datetime
import pandas as pd
import mlflow

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.observability.logger import get_logger

logger = get_logger("monitoring.report_generator")

class ReportGenerator:
    """Generates drift and data quality reports using Evidently AI."""
    
    def __init__(self, config_path: str = "config/monitoring.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.reference_path = self.config.get("reference_dataset_path")
        self.production_path = self.config.get("production_dataset_path")
        self.output_dir = self.config.get("output_directory", "artifacts/monitoring")
        self.min_samples = self.config.get("minimum_samples_before_monitoring", 10)
        
    def generate_reports(self) -> dict:
        """Loads data and generates reports. Returns basic drift summary."""
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference data not found at {self.reference_path}")
            
        if not os.path.exists(self.production_path):
            raise FileNotFoundError(f"Production data not found at {self.production_path}")
            
        # Load datasets
        ref_data = pd.read_parquet(self.reference_path)
        prod_data = pd.read_parquet(self.production_path)
        
        # Check minimum samples
        if len(prod_data) < self.min_samples:
            return {
                "status": "skipped",
                "reason": f"Insufficient production samples: {len(prod_data)} < {self.min_samples}"
            }
            
        logger.info(f"Generating reports for {len(ref_data)} reference vs {len(prod_data)} production samples.")
        
        # Build report (Data Quality + Data Drift)
        # Note: We monitor prediction probability drift as part of data drift
        report = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])
        
        report.run(reference_data=ref_data, current_data=prod_data)
        
        # Define output paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        html_dir = os.path.join(self.output_dir, "reports", "html")
        json_dir = os.path.join(self.output_dir, "reports", "json")
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        html_path = os.path.join(html_dir, f"drift_report_{timestamp}.html")
        json_path = os.path.join(json_dir, f"drift_report_{timestamp}.json")
        
        report.save_html(html_path)
        report.save_json(json_path)
        
        # MLflow Integration
        self._log_to_mlflow(html_path, json_path, self.reference_path, self.production_path)
        
        # Extract basic metrics from JSON representation
        report_dict = report.as_dict()
        drift_share = report_dict["metrics"][0]["result"]["dataset_drift"]
        drifted_features = report_dict["metrics"][0]["result"]["number_of_drifted_columns"]
        
        return {
            "status": "success",
            "drift_detected": drift_share > 0.5, # Default threshold in evidently
            "drift_share": drift_share,
            "drifted_features": drifted_features,
            "timestamp": timestamp,
            "html_report": html_path,
            "json_report": json_path,
        }

    def _log_to_mlflow(self, html_path: str, json_path: str, ref_path: str, prod_path: str):
        """Log reports and datasets to MLflow."""
        try:
            # We don't start a run if one is already active, but usually this runs outside training
            if mlflow.active_run() is None:
                mlflow.set_experiment("Monitoring")
                with mlflow.start_run(run_name=f"drift_check_{datetime.datetime.now().strftime('%Y%m%d')}"):
                    mlflow.log_artifact(html_path, artifact_path="monitoring/reports")
                    mlflow.log_artifact(json_path, artifact_path="monitoring/metrics")
                    
                    # Log the datasets used for this check
                    mlflow.log_artifact(ref_path, artifact_path="monitoring/datasets")
                    mlflow.log_artifact(prod_path, artifact_path="monitoring/datasets")
            else:
                mlflow.log_artifact(html_path, artifact_path="monitoring/reports")
                mlflow.log_artifact(json_path, artifact_path="monitoring/metrics")
                mlflow.log_artifact(ref_path, artifact_path="monitoring/datasets")
                mlflow.log_artifact(prod_path, artifact_path="monitoring/datasets")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
