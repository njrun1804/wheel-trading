#!/usr/bin/env python3
"""
Bolt executable wrapper script.
This provides the main entry point for the Bolt CLI system.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# Import and run the main CLI
if __name__ == "__main__":
    # Change to the bolt_cli script and execute it
    bolt_cli_path = current_dir / "bolt_cli"
    
    if bolt_cli_path.exists():
        # Execute the bolt_cli script with the same arguments
        os.execv(str(bolt_cli_path), [str(bolt_cli_path)] + sys.argv[1:])
    else:
        print("Error: bolt_cli script not found", file=sys.stderr)
        sys.exit(1)