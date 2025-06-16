#!/usr/bin/env python3
"""
Targeted linting fixes to address the most common issues.
This approach manually applies the most effective fixes.
"""
import os
import re
from pathlib import Path
from typing import List, Tuple


def fix_import_issues(file_path: Path) -> List[str]:
    """Fix common import issues."""
    fixes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix unused imports - look for simple cases
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Skip lines that are clearly used imports (in __all__, function names, etc.)
        if line.strip().startswith('from ') or line.strip().startswith('import '):
            # Basic heuristic: if import appears elsewhere in file, keep it
            import_parts = line.replace('from ', '').replace('import ', '').replace(' as ', ' ')
            import_names = [name.strip() for name in import_parts.split(',')]
            
            # Check if any import name appears in the rest of the file
            rest_of_file = '\n'.join(lines[lines.index(line)+1:])
            
            used = False
            for name in import_names:
                if '.' in name:
                    name = name.split('.')[-1]  # Get the actual imported name
                if name in rest_of_file:
                    used = True
                    break
            
            if used or '__all__' in rest_of_file:
                new_lines.append(line)
            else:
                fixes.append(f"Removed unused import: {line.strip()}")
        else:
            new_lines.append(line)
    
    # Fix import organization - basic sorting
    import_section = []
    other_lines = []
    in_imports = True
    
    for line in new_lines:
        if line.strip().startswith(('import ', 'from ')) and in_imports:
            import_section.append(line)
        elif line.strip() == '' and in_imports:
            import_section.append(line)
        else:
            if in_imports and line.strip():  # First non-import line
                in_imports = False
            other_lines.append(line)
    
    # Basic import sorting
    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    
    for line in import_section:
        if line.strip().startswith('from __future__'):
            stdlib_imports.insert(0, line)  # Keep __future__ at top
        elif line.strip().startswith(('import os', 'import sys', 'from os', 'from sys', 'import re', 'from re', 'import json', 'from json')):
            stdlib_imports.append(line)
        elif line.strip().startswith(('import numpy', 'from numpy', 'import pandas', 'from pandas', 'import scipy', 'from scipy')):
            third_party_imports.append(line)
        elif line.strip().startswith(('from .', 'from ..')):
            local_imports.append(line)
        elif line.strip() == '':
            continue  # Remove extra blank lines, we'll add them back
        else:
            # Default to third party
            third_party_imports.append(line)
    
    # Reassemble with proper spacing
    new_content_lines = []
    if stdlib_imports:
        new_content_lines.extend(stdlib_imports)
        new_content_lines.append('')
    if third_party_imports:
        new_content_lines.extend(third_party_imports)
        new_content_lines.append('')
    if local_imports:
        new_content_lines.extend(local_imports)
        new_content_lines.append('')
    
    new_content_lines.extend(other_lines)
    new_content = '\n'.join(new_content_lines)
    
    # Remove excessive blank lines
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
    
    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        fixes.append("Organized imports and removed blank lines")
    
    return fixes


def fix_unused_variables(file_path: Path) -> List[str]:
    """Fix unused variables by adding underscore prefix."""
    fixes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Simple pattern for unused variables assigned but not used
    # This is a basic heuristic and won't catch all cases
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_line = line
        
        # Look for simple variable assignments
        if '=' in line and not line.strip().startswith('#'):
            # Extract variable name (basic case)
            match = re.match(r'(\s*)(\w+)\s*=', line)
            if match:
                indent, var_name = match.groups()
                
                # Check if variable is used later in the function/method
                # Look ahead in same indentation block
                used = False
                for j in range(i + 1, len(lines)):
                    future_line = lines[j]
                    
                    # Stop at end of block (different indentation)
                    if future_line.strip() and not future_line.startswith(indent + '    ') and not future_line.startswith(indent + '\t'):
                        if len(future_line) - len(future_line.lstrip()) <= len(indent):
                            break
                    
                    # Check if variable is used
                    if var_name in future_line and f'{var_name}=' not in future_line:
                        used = True
                        break
                
                # If not used and not already prefixed with underscore, add it
                if not used and not var_name.startswith('_') and var_name not in ['self', 'cls']:
                    new_line = line.replace(f'{var_name}=', f'_{var_name}=', 1)
                    fixes.append(f"Prefixed unused variable with underscore: {var_name}")
        
        new_lines.append(new_line)
    
    new_content = '\n'.join(new_lines)
    
    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return fixes


def fix_whitespace_issues(file_path: Path) -> List[str]:
    """Fix whitespace and formatting issues."""
    fixes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Remove trailing whitespace
    lines = content.split('\n')
    new_lines = [line.rstrip() for line in lines]
    
    # Remove excessive blank lines at end of file
    while new_lines and new_lines[-1] == '':
        new_lines.pop()
    
    # Ensure single newline at end of file
    new_lines.append('')
    
    new_content = '\n'.join(new_lines)
    
    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        fixes.append("Fixed trailing whitespace and end-of-file newlines")
    
    return fixes


def fix_simple_syntax_issues(file_path: Path) -> List[str]:
    """Fix simple syntax issues."""
    fixes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix lambda assignments (convert to def)
    content = re.sub(
        r'^(\s*)(\w+)\s*=\s*lambda\s*([^:]*?):\s*(.+)$',
        r'\1def \2(\3):\n\1    return \4',
        content,
        flags=re.MULTILINE
    )
    
    # Fix comparison to None (use 'is' instead of '==')
    content = re.sub(r'==\s*None\b', 'is None', content)
    content = re.sub(r'!=\s*None\b', 'is not None', content)
    
    # Fix comparison to True/False
    content = re.sub(r'==\s*True\b', 'is True', content)
    content = re.sub(r'==\s*False\b', 'is False', content)
    content = re.sub(r'!=\s*True\b', 'is not True', content)
    content = re.sub(r'!=\s*False\b', 'is not False', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        fixes.append("Fixed comparison operators and lambda assignments")
    
    return fixes


def process_python_files():
    """Process all Python files in src/ directory."""
    base_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
    src_path = base_path / "src"
    
    total_fixes = 0
    files_processed = 0
    
    print("Processing Python files for linting fixes...")
    print("=" * 60)
    
    for py_file in src_path.rglob("*.py"):
        try:
            print(f"\nProcessing: {py_file.relative_to(base_path)}")
            
            file_fixes = []
            file_fixes.extend(fix_import_issues(py_file))
            file_fixes.extend(fix_unused_variables(py_file))
            file_fixes.extend(fix_whitespace_issues(py_file))
            file_fixes.extend(fix_simple_syntax_issues(py_file))
            
            if file_fixes:
                for fix in file_fixes:
                    print(f"  ✓ {fix}")
                total_fixes += len(file_fixes)
            else:
                print("  ✓ No fixes needed")
            
            files_processed += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {py_file}: {e}")
    
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print(f"Files processed: {files_processed}")
    print(f"Total fixes applied: {total_fixes}")
    print("=" * 60)


if __name__ == "__main__":
    process_python_files()