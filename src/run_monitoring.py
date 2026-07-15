import logging
from src.monitoring.report_generator import ReportGenerator
from src.monitoring.drift_monitor import DriftMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Runs the monitoring pipeline, designed to be executed via scheduled cron."""
    logger.info("Starting scheduled monitoring pipeline...")
    
    try:
        generator = ReportGenerator()
        monitor = DriftMonitor()
        
        # Generate reports and log to MLflow
        result = generator.generate_reports()
        
        # Append to historical metrics tracker
        monitor.log_metrics(result)
        
        if result.get("status") == "skipped":
            logger.info(f"Monitoring skipped: {result.get('reason')}")
        else:
            logger.info(f"Monitoring completed successfully. Drift detected: {result.get('drift_detected')}")
            
    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
