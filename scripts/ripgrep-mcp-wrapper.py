#!/usr/bin/env python3
"""Ripgrep MCP wrapper to handle paths with spaces."""

import sys
import os

# Add the scripts directory to Python path
scripts_dir = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts"
sys.path.insert(0, scripts_dir)

# Import and run the actual ripgrep MCP
import importlib.util
spec = importlib.util.spec_from_file_location("ripgrep_mcp", os.path.join(scripts_dir, "ripgrep-mcp.py"))
ripgrep_mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ripgrep_mcp)
