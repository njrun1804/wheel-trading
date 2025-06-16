#!/usr/bin/env python3
"""
BOB Unified Entry Point

This is the main entry point for the BOB (Bolt Orchestrator Bootstrap) system.
It provides a unified interface that replaces the fragmented CLI tools:

- bolt_cli.py -> bob solve
- bob_cli.py -> bob (natural language)
- unified_cli.py -> bob (all features)

Usage:
    python bob_unified.py solve "optimize the trading system"
    python bob_unified.py "fix authentication issues"
    python bob_unified.py --interactive
"""

import sys
from pathlib import Path

# Add BOB to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the unified CLI
from bob.cli.main import sync_main

if __name__ == "__main__":
    sys.exit(sync_main())