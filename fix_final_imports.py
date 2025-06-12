#!/usr/bin/env python3
"""Fix final remaining src.unity_wheel imports."""

import os
import re

def fix_imports(file_path):
    """Fix src.unity_wheel imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix various patterns
    patterns = [
        (r'from src\.unity_wheel\.', 'from unity_wheel.'),
        (r'import src\.unity_wheel\.', 'import unity_wheel.'),
        (r'from src\.unity_wheel ', 'from unity_wheel '),
        (r'import src\.unity_wheel', 'import unity_wheel'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Main function to fix imports."""
    fixed_files = []
    
    # Walk through all Python files
    for root, dirs, files in os.walk('src'):
        # Skip __pycache__ directories
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports(file_path):
                    fixed_files.append(file_path)
    
    print(f"Fixed {len(fixed_files)} files:")
    for file in sorted(fixed_files):
        print(f"  ✓ {file}")

if __name__ == '__main__':
    main()