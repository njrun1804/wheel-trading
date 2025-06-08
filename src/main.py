"""Main entry point for the wheel trading application."""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the wheel trading application."""
    logger.info("Starting Wheel Trading Application")
    logger.info(f"Current time: {datetime.now()}")
    logger.info(f"Environment: {os.getenv('NODE_ENV', 'production')}")
    logger.info(f"Google Cloud Project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'not set')}")

    # TODO: Implement wheel trading logic
    logger.info("Application started successfully")


if __name__ == "__main__":
    main()
