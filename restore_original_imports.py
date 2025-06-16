#!/usr/bin/env python3
"""
Restore Original Imports Script

Used during rollback to restore original import statements.
"""

import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_imports_in_file(file_path: Path) -> bool:
    """Restores original import statements in a file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Reverse the import transformations
        restore_transforms = [
            (r"from bob\.search\.", "from einstein."),
            (r"import bob\.search\.", "import einstein."),
            (r"from bob\.search ", "from einstein "),
            (r"bob\.search\.engine", "einstein.unified_index"),
            (r"bob\.search\.query_processor", "einstein.query_router"),
            (r"bob\.search\.result_aggregator", "einstein.result_merger"),
            (r"bob\.config\.search_config", "einstein.einstein_config"),
            (r"bob\.performance\.adaptive_concurrency", "einstein.adaptive_concurrency"),
            (r"bob\.hardware\.m4_pro_optimizer", "einstein.m4_pro_optimizer"),
        ]
        
        original_content = content
        for pattern, replacement in restore_transforms:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"✅ Restored imports in {file_path}")
            return True
        else:
            logger.info(f"ℹ️  No imports to restore in {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to restore imports in {file_path}: {e}")
        return False

def main():
    """Restore imports in all Python files."""
    project_root = Path.cwd()
    
    python_files = list(project_root.rglob("*.py"))
    logger.info(f"Restoring imports in {len(python_files)} Python files...")
    
    success_count = 0
    for py_file in python_files:
        if restore_imports_in_file(py_file):
            success_count += 1
    
    logger.info(f"✅ Successfully restored imports in {success_count}/{len(python_files)} files")

if __name__ == "__main__":
    main()