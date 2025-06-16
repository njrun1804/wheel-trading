#!/usr/bin/env python3
"""
CoreML Installation Helper for ANE Acceleration

Installs coremltools package if not available.
"""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def install_coreml():
    """Install CoreML tools for ANE acceleration"""
    try:
        import coremltools

        logger.info("✅ CoreML tools already installed")
        return True
    except ImportError:
        pass

    logger.info("Installing CoreML tools for ANE acceleration...")

    try:
        # Install coremltools
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "coremltools>=7.0", "--quiet"]
        )

        # Verify installation
        import coremltools

        logger.info("✅ CoreML tools installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install CoreML tools: {e}")
        return False
    except ImportError as e:
        logger.error(f"❌ CoreML installation verification failed: {e}")
        return False


if __name__ == "__main__":
    success = install_coreml()
    sys.exit(0 if success else 1)
