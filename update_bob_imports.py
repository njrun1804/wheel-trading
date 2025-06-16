#!/usr/bin/env python3
"""
Script to update import paths in migrated BOB files.

This script updates import statements to use the new BOB structure:
- bolt.* imports -> bob.* imports
- einstein.* imports -> bob.search.* imports
- Relative import fixes within BOB
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path: Path):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update bolt imports to bob
        content = re.sub(r'from bolt\.', 'from bob.', content)
        content = re.sub(r'import bolt\.', 'import bob.', content)
        
        # Update einstein imports to bob.search
        content = re.sub(r'from einstein\.', 'from bob.search.', content)
        content = re.sub(r'import einstein\.', 'import bob.search.', content)
        
        # Update specific bolt core imports
        content = re.sub(r'from \.\.error_handling', 'from .error_handling', content)
        content = re.sub(r'from bolt\.core\.', 'from bob.integration.', content)
        content = re.sub(r'from bolt\.agents\.', 'from bob.agents.', content)
        content = re.sub(r'from bolt\.hardware\.', 'from bob.hardware.', content)
        
        # Fix some common relative import issues
        content = re.sub(r'from \.\.\.', 'from bob.', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in: {file_path.relative_to(Path.cwd())}")
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def update_all_bob_imports():
    """Update imports in all BOB Python files."""
    bob_dir = Path("bob")
    
    if not bob_dir.exists():
        print("BOB directory not found!")
        return
    
    # Find all Python files in BOB directory
    python_files = list(bob_dir.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to update...")
    
    for file_path in python_files:
        update_imports_in_file(file_path)
    
    print("Import update complete!")

if __name__ == "__main__":
    update_all_bob_imports()