#!/usr/bin/env python3
"""Set up comprehensive logging for Wheel Trading project."""

import os
import logging
import logging.handlers
from pathlib import Path
import json

def setup_project_logging():
    """Configure project-wide logging to files and console."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file_all": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/wheel_trading.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/errors.log",
                "maxBytes": 10485760,
                "backupCount": 3
            },
            "file_trading": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "logs/trading.log",
                "maxBytes": 10485760,
                "backupCount": 5
            }
        },
        "loggers": {
            "unity_wheel": {
                "level": "DEBUG",
                "handlers": ["console", "file_all", "file_errors"],
                "propagate": False
            },
            "unity_wheel.strategy": {
                "level": "INFO",
                "handlers": ["file_trading"],
                "propagate": True
            },
            "unity_wheel.risk": {
                "level": "INFO",
                "handlers": ["file_trading"],
                "propagate": True
            },
            "jarvis2": {
                "level": "DEBUG",
                "handlers": ["console", "file_all"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file_all"]
        }
    }
    
    # Save configuration
    config_path = Path("logging_config.json")
    with open(config_path, "w") as f:
        json.dump(logging_config, f, indent=2)
    
    print("✅ Logging configuration created at logging_config.json")
    print("📁 Log files will be written to ./logs/")
    print("")
    print("To use in your Python code:")
    print("  import logging.config")
    print("  import json")
    print("  with open('logging_config.json') as f:")
    print("      logging.config.dictConfig(json.load(f))")
    print("  logger = logging.getLogger('unity_wheel')")
    
    # Create example integration file
    integration_example = '''"""Example of how to use project logging."""
import logging.config
import json
from pathlib import Path

# Load logging configuration
config_path = Path(__file__).parent.parent / "logging_config.json"
if config_path.exists():
    with open(config_path) as f:
        logging.config.dictConfig(json.load(f))

# Get logger for this module
logger = logging.getLogger(__name__)

# Now use it
logger.info("Module initialized with project logging")
'''
    
    example_path = Path("src/unity_wheel/utils/logging_setup.py")
    example_path.parent.mkdir(parents=True, exist_ok=True)
    with open(example_path, "w") as f:
        f.write(integration_example)
    
    print("\n✅ Created example integration at src/unity_wheel/utils/logging_setup.py")

if __name__ == "__main__":
    setup_project_logging()