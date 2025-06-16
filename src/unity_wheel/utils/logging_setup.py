"""Example of how to use project logging."""
import json
import logging.config
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
