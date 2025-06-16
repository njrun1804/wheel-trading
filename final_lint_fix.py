#!/usr/bin/env python3
"""
Final comprehensive linting fix to get from current state to under 500 issues.
This script applies remaining import fixes and then runs all automated fixes.
"""
import subprocess
import sys
import os
from pathlib import Path
import re

def fix_remaining_imports():
    """Fix remaining critical import issues manually."""
    base_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
    
    # Remaining critical files to fix
    remaining_fixes = [
        ("src/unity_wheel/risk/borrowing_cost_analyzer.py", "from src.config.loader import", "from ...config.loader import"),
        ("src/unity_wheel/risk/unity_margin.py", "from src.config.loader import", "from ...config.loader import"),
        ("src/unity_wheel/monitoring/diagnostics.py", "from src.config.loader import", "from ...config.loader import"),
        ("src/unity_wheel/analytics/decision_engine.py", "from src.config.loader import", "from ...config.loader import"),
        ("src/unity_wheel/analytics/dynamic_optimizer.py", "from src.config.loader import", "from ...config.loader import"),
        ("src/unity_wheel/data_providers/databento/client.py", "from src.config.loader import", "from ....config.loader import"),
    ]
    
    print("Fixing remaining critical imports...")
    fixes_applied = 0
    
    for file_path_str, old_pattern, new_pattern in remaining_fixes:
        file_path = base_path / file_path_str
        
        if not file_path.exists():
            print(f"  Skip (not found): {file_path_str}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  ✓ Fixed: {file_path_str}")
                fixes_applied += 1
            else:
                print(f"  - Pattern not found in: {file_path_str}")
                
        except Exception as e:
            print(f"  ✗ Error fixing {file_path_str}: {e}")
    
    print(f"Applied {fixes_applied} additional import fixes")
    return fixes_applied

def run_automated_fixes():
    """Run all automated ruff fixes."""
    cwd = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
    
    commands = [
        # Phase 1: Safe fixes
        (["ruff", "check", "--fix", "src/"], "Apply safe automated fixes"),
        
        # Phase 2: Format code
        (["ruff", "format", "src/"], "Format code"),
        
        # Phase 3: Unsafe fixes (unused imports, variables)
        (["ruff", "check", "--fix", "--unsafe-fixes", "src/"], "Apply unsafe fixes"),
        
        # Phase 4: Final status
        (["ruff", "check", "--statistics", "src/"], "Check final status"),
    ]
    
    for cmd, description in commands:
        print(f"\n{description}...")
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout
            )
            
            if result.stdout.strip():
                print("Output:")
                # Limit output to avoid too much noise
                lines = result.stdout.split('\n')[:50]  # First 50 lines
                for line in lines:
                    print(f"  {line}")
                if len(result.stdout.split('\n')) > 50:
                    print(f"  ... ({len(result.stdout.split('\n')) - 50} more lines)")
            
            if result.stderr.strip():
                print("Errors:")
                error_lines = result.stderr.split('\n')[:20]  # First 20 error lines
                for line in error_lines:
                    print(f"  {line}")
            
            print(f"Return code: {result.returncode}")
            
        except subprocess.TimeoutExpired:
            print("  Command timed out")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Execute comprehensive linting fix."""
    print("="*60)
    print("FINAL LINTING FIX - TARGET: <500 ISSUES")
    print("="*60)
    
    # Step 1: Fix remaining imports manually
    print("\nSTEP 1: Fix remaining critical imports")
    import_fixes = fix_remaining_imports()
    
    # Step 2: Run automated fixes
    print("\nSTEP 2: Run automated fixes")
    run_automated_fixes()
    
    print("\n" + "="*60)
    print("LINTING FIX COMPLETE")
    print("="*60)
    print(f"Additional import fixes applied: {import_fixes}")
    print("\nTo verify results:")
    print("1. Run: ruff check --statistics src/")
    print("2. Check if total violations < 500")
    print("3. Review any remaining F401 (unused imports)")
    print("4. Review any remaining F841 (unused variables)")
    print("5. Add # noqa comments where needed")
    print("\nIf >500 issues remain:")
    print("- Focus on F401/F841 errors first")
    print("- Add per-file-ignores to pyproject.toml")
    print("- Consider # noqa: F401 for intentional imports")

if __name__ == "__main__":
    main()