#!/usr/bin/env python3
"""Main CLI entry point for Bolt system."""

import sys
from pathlib import Path

# Add bolt directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main CLI
from bolt.solve import main

if __name__ == "__main__":
    main()
