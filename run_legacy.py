#!/usr/bin/env python3
"""Legacy entry point for Unity Wheel Trading Bot."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.cli.run_legacy import main

if __name__ == "__main__":
    main()
