#!/usr/bin/env python3
"""
Simple import fix for the most common issues.
Fixes src.config.loader imports to relative imports.
"""
import re
from pathlib import Path


# Files to fix (the most critical ones)
CRITICAL_FILES = [
    "src/unity_wheel/api/advisor.py",
    "src/unity_wheel/strategy/wheel.py", 
    "src/unity_wheel/risk/analytics.py",
    "src/unity_wheel/risk/advanced_financial_modeling.py",
    "src/unity_wheel/portfolio/single_account.py",
    "src/unity_wheel/cli/run.py",
    "src/unity_wheel/storage/optimized_storage.py",
    "src/unity_wheel/backtesting/wheel_backtester.py",
]

def fix_file_imports(file_path: Path) -> int:
    """Fix imports in a single file."""
    fixes = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the most common import issues
        replacements = [
            (r'from src\.config\.loader import', 'from ...config.loader import'),
            (r'from unity_wheel\.([^.\s]+)', r'from ..\1'),
            # Fix specific problematic imports
            (r'import src\.', 'import ..'),
        ]
        
        for pattern, replacement in replacements:
            old_content = content
            content = re.sub(pattern, replacement, content)
            if content != old_content:
                fixes += 1
        
        # Remove trailing whitespace
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)
        
        # Ensure single newline at end
        content = content.rstrip() + '\n'
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return fixes
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return 0


def main():
    """Fix the most critical import issues."""
    base_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
    
    print("Fixing critical import issues...")
    print("=" * 50)
    
    total_fixes = 0
    
    for file_rel_path in CRITICAL_FILES:
        file_path = base_path / file_rel_path
        
        if file_path.exists():
            print(f"Processing: {file_rel_path}")
            fixes = fix_file_imports(file_path)
            total_fixes += fixes
            print(f"  Applied {fixes} fixes")
        else:
            print(f"  File not found: {file_rel_path}")
    
    print("\n" + "=" * 50)
    print(f"Total fixes applied: {total_fixes}")
    print("Next steps:")
    print("1. Run ruff check --fix to apply automated fixes")
    print("2. Run ruff check to see remaining issues")
    print("=" * 50)


if __name__ == "__main__":
    main()