#!/usr/bin/env python3
"""Fix import issues in test files."""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix relative imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    changes = []
    
    # Fix patterns
    patterns = [
        # Fix triple/quad dots
        (r'from \.\.\.\.?(\w+)', r'from unity_wheel.\1'),
        (r'from \.\.\.(\w+)', r'from unity_wheel.\1'),
        
        # Fix test imports
        (r'from unity_wheel\.(\w+) import', r'from src.unity_wheel.\1 import'),
        (r'import unity_wheel\.', r'import src.unity_wheel.'),
        
        # Fix common issues
        (r'from src\.src\.', r'from src.'),
        (r'from ..config.loader import get_config', 
         r'from src.unity_wheel.config.loader import get_config'),
    ]
    
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append(f"Applied: {pattern} -> {replacement}")
            content = new_content
    
    # Only write if changed
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True, changes
    
    return False, []

def main():
    """Fix imports in all test files."""
    print("Fixing imports in test files...")
    
    test_dirs = [
        "src/unity_wheel/tests",
        "tests",
        "src/unity_wheel/*/tests"
    ]
    
    fixed_count = 0
    
    for pattern in test_dirs:
        for test_file in Path(".").glob(f"{pattern}/**/*.py"):
            if test_file.is_file():
                fixed, changes = fix_imports_in_file(test_file)
                if fixed:
                    print(f"\n✓ Fixed {test_file}")
                    for change in changes[:3]:  # Show first 3 changes
                        print(f"  - {change}")
                    if len(changes) > 3:
                        print(f"  ... and {len(changes)-3} more")
                    fixed_count += 1
    
    print(f"\n✅ Fixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()