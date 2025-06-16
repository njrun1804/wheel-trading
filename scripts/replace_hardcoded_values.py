#!/usr/bin/env python3
"""Replace all hardcoded values with configuration references."""

import re
from pathlib import Path

# Define replacement patterns
REPLACEMENTS = [
    # Symbol replacements
    (r'symbol\s*=\s*["\']U["\']', "symbol = config.trading.symbol"),
    (r'ticker\s*=\s*["\']U["\']', "ticker = config.trading.symbol"),
    (r'"U"(?=\s*#.*symbol)', "config.trading.symbol"),
    # Portfolio value replacements
    (
        r"portfolio_value\s*=\s*100000",
        "portfolio_value = config.trading.portfolio_value",
    ),
    (
        r"portfolio_value\s*=\s*200000",
        "portfolio_value = config.trading.portfolio_value",
    ),
    (r"portfolio\s*=\s*\d+", "portfolio = config.trading.portfolio_value"),
    # Delta replacements
    (r"target_delta\s*=\s*0\.\d+", "target_delta = config.trading.target_delta"),
    (r"delta\s*=\s*0\.\d+(?=\s*#.*target)", "delta = config.trading.target_delta"),
    (r"delta_target\s*=\s*0\.\d+", "delta_target = config.trading.target_delta"),
    # DTE replacements
    (r"target_dte\s*=\s*\d+", "target_dte = config.trading.target_dte"),
    (r"dte\s*=\s*30", "dte = config.trading.target_dte"),
    (r"days_to_expiry\s*=\s*30", "days_to_expiry = config.trading.target_dte"),
    # Position size replacements
    (
        r"max_position_size\s*=\s*0\.\d+",
        "max_position_size = config.trading.max_position_size",
    ),
    # Risk replacements
    (r"max_var_95\s*=\s*0\.\d+", "max_var_95 = config.risk.max_var_95"),
    (r"max_cvar_95\s*=\s*0\.\d+", "max_cvar_95 = config.risk.max_cvar_95"),
    # Database path replacements
    (
        r'["\']~/\.wheel_trading/cache/wheel_cache\.duckdb["\']',
        "config.storage.database_path",
    ),
    (r'["\']wheel_cache\.duckdb["\']', "Path(config.storage.database_path).name"),
    # Config file replacements
    (
        r'["\']config\.yaml["\']',
        'os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml")',
    ),
    (
        r'["\']config_unified\.yaml["\']',
        'os.getenv("WHEEL_CONFIG_PATH", "config/unified.yaml")',
    ),
]

# Files to add config import to
CONFIG_IMPORT = (
    "from unity_wheel.config.unified_config import get_config\nconfig = get_config()\n"
)


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped."""
    skip_patterns = [
        "__pycache__",
        ".git",
        "venv",
        "archive",
        "mcp-servers",
        ".pyc",
        "replace_hardcoded_values.py",  # Don't modify this script
        "unified_config.py",  # Don't modify the config file itself
    ]

    return any(pattern in str(file_path) for pattern in skip_patterns)


def add_config_import(content: str, file_path: Path) -> str:
    """Add config import if needed."""
    # Skip if already has config import
    if "from unity_wheel.config" in content or "get_config" in content:
        return content

    # Skip test files and scripts
    if "test_" in file_path.name or file_path.parent.name == "scripts":
        return content

    # Find the right place to add import (after other imports)
    lines = content.split("\n")
    import_line = -1

    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_line = i

    if import_line >= 0:
        # Add after last import
        lines.insert(import_line + 1, "\n" + CONFIG_IMPORT)
        return "\n".join(lines)

    # Add after module docstring
    in_docstring = False
    docstring_end = -1

    for i, line in enumerate(lines):
        if line.strip().startswith('"""'):
            if not in_docstring:
                in_docstring = True
            else:
                docstring_end = i
                break

    if docstring_end >= 0:
        lines.insert(docstring_end + 1, "\n" + CONFIG_IMPORT)
        return "\n".join(lines)

    # Add at the beginning
    return CONFIG_IMPORT + "\n" + content


def replace_hardcoded_values(file_path: Path) -> tuple[bool, int]:
    """Replace hardcoded values in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        replacements_made = 0

        # Apply replacements
        for pattern, replacement in REPLACEMENTS:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                replacements_made += count

        # Add config import if we made replacements
        if replacements_made > 0:
            content = add_config_import(content, file_path)

        # Only write if changes were made
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, replacements_made

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    """Replace hardcoded values in all Python files."""
    project_root = Path(__file__).parent.parent

    total_files = 0
    modified_files = 0
    total_replacements = 0

    print("🔄 Replacing hardcoded values with configuration references...")

    # Process all Python files
    for py_file in project_root.rglob("*.py"):
        if should_skip_file(py_file):
            continue

        total_files += 1
        modified, replacements = replace_hardcoded_values(py_file)

        if modified:
            modified_files += 1
            total_replacements += replacements
            print(
                f"✅ Modified {py_file.relative_to(project_root)} ({replacements} replacements)"
            )

    print("\n📊 Summary:")
    print(f"  Files scanned: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total replacements: {total_replacements}")

    return 0


if __name__ == "__main__":
    main()
