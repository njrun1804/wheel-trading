#!/usr/bin/env python3
"""Clean startup script for Unity Wheel Trading System.

Sets up hardware optimization and validates system readiness.
"""

import logging
import multiprocessing
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def maximize_hardware():
    """Set environment variables for maximum hardware utilization."""
    cpu_count = multiprocessing.cpu_count()

    # Set thread counts for all numerical libraries
    env_vars = {
        "OMP_NUM_THREADS": str(cpu_count),
        "MKL_NUM_THREADS": str(cpu_count),
        "NUMEXPR_NUM_THREADS": str(cpu_count),
        "VECLIB_MAXIMUM_THREADS": str(cpu_count),
        "OPENBLAS_NUM_THREADS": str(cpu_count),
        "BLIS_NUM_THREADS": str(cpu_count),
        # Python optimizations
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # DuckDB optimizations
        "DUCKDB_MEMORY_LIMIT": "16GB",
        "DUCKDB_THREADS": str(cpu_count),
        # Disable debugging for performance
        "PYTHONOPTIMIZE": "2",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    logger.info(f"⚡ Hardware optimization configured for {cpu_count} cores")
    return cpu_count


def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = [
        "numpy",
        "pandas",
        "duckdb",
        "scipy",
        "rich",
        "click",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

    logger.info("✅ All core dependencies available")
    return True


def check_project_structure():
    """Verify we're in the correct project directory."""
    required_paths = [
        Path("src/unity_wheel"),
        Path("config.yaml"),
        Path("data"),
    ]

    missing = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))

    if missing:
        logger.error(f"Missing project components: {', '.join(missing)}")
        logger.error("Please run from the Unity Wheel project root directory")
        return False

    logger.info("✅ Project structure verified")
    return True


def check_environment():
    """Check environment variables and configuration."""
    # Create necessary directories
    dirs_to_create = [
        Path("logs"),
        Path("data/cache"),
        Path.home() / ".wheel_trading" / "secrets",
        Path.home() / ".wheel_trading" / "cache",
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Check for .env file
    if not Path(".env").exists() and Path(".env.example").exists():
        logger.warning("⚠️  No .env file found. Creating from template...")
        import shutil

        shutil.copy(".env.example", ".env")
        logger.warning("   Please edit .env and add your API keys")

    logger.info("✅ Environment configured")
    return True


def main():
    """Run clean startup sequence."""
    logger.info("🚀 Unity Wheel Trading System - Clean Startup")
    logger.info("━" * 50)

    # Run all checks
    all_good = True

    # 1. Maximize hardware
    cpu_count = maximize_hardware()

    # 2. Check project structure
    if not check_project_structure():
        all_good = False

    # 3. Check dependencies
    if not check_dependencies():
        all_good = False

    # 4. Check environment
    if not check_environment():
        all_good = False

    # 5. Try to import main modules
    try:
        sys.path.insert(0, "src")
        logger.info("✅ Unity Wheel package importable")
    except Exception as e:
        logger.error(f"❌ Failed to import Unity Wheel: {e}")
        all_good = False

    # Summary
    logger.info("━" * 50)
    if all_good:
        logger.info("✅ STARTUP SUCCESSFUL")
        logger.info(f"   Running with {cpu_count} CPU cores")
        logger.info("   All optimizations enabled")
        return 0
    else:
        logger.error("❌ STARTUP FAILED")
        logger.error("   Please fix the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
