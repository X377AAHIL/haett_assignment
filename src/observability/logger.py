import os
import yaml
import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

# Context variables for global tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default=None)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default=None)

# Cache for initialized loggers
_loggers = {}
_config = None

def get_observability_config():
    global _config
    if _config is None:
        config_path = "config/observability.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                _config = yaml.safe_load(f)
        else:
            _config = {}
    return _config


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Automatically injects context variables into log records."""
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        config = get_observability_config()
        
        # Inject standard context variables
        extra["request_id"] = request_id_var.get()
        extra["correlation_id"] = correlation_id_var.get()
        extra["model_version"] = config.get("model_version", "unknown")
        extra["service_name"] = config.get("service_name", "unknown")
        
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Factory to retrieve a configured ContextLoggerAdapter."""
    if name in _loggers:
        return _loggers[name]
        
    config = get_observability_config()
    log_level_str = config.get("logging", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid attaching multiple handlers if logger is already configured
    if not logger.handlers:
        log_dir = config.get("logging", {}).get("log_directory", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Map logger names to specific files if desired, otherwise default to api.log
        if "training" in name:
            log_file = "training.log"
        elif "monitoring" in name:
            log_file = "monitoring.log"
        elif "prediction" in name or "shap" in name:
            log_file = "prediction.log"
        else:
            log_file = "api.log"
            
        file_path = os.path.join(log_dir, log_file)
        
        rotation_size = config.get("logging", {}).get("rotation_size_mb", 10) * 1024 * 1024
        backup_count = config.get("logging", {}).get("backup_count", 5)
        
        handler = RotatingFileHandler(file_path, maxBytes=rotation_size, backupCount=backup_count)
        
        if config.get("logging", {}).get("enable_json_logs", True):
            # Define JSON layout
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level"}
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add console handler for local dev
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
    adapter = ContextLoggerAdapter(logger, {})
    _loggers[name] = adapter
    return adapter
