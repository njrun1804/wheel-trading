#!/usr/bin/env python3
"""Fix all get_config imports to use the correct module."""

import os
import re

# Find all Python files
for root, dirs, files in os.walk("."):
    # Skip venv and __pycache__
    if "venv" in root or "__pycache__" in root:
        continue

    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)

            try:
                with open(filepath, "r") as f:
                    content = f.read()

                # Replace the import
                new_content = content.replace(
                    "from src.config import get_config", "from src.config import get_config"
                )

                # Only write if changed
                if new_content != content:
                    with open(filepath, "w") as f:
                        f.write(new_content)
                    print(f"Fixed: {filepath}")

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

print("Done!")
