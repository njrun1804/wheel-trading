#!/usr/bin/env python3
"""
Targeted cleanup script for AsyncIO unused imports.
"""

import json
from pathlib import Path


def find_asyncio_files():
    """Find files with AsyncIO unused imports from the analysis."""
    with open("smart_import_analysis.json") as f:
        data = json.load(f)

    asyncio_files = []
    for result in data["all_results"]:
        if result.get("error"):
            continue

        file_path = result["file"]
        for imp in result["unused_imports"]:
            if imp["category"] == "asyncio":
                asyncio_files.append({"file": file_path, "import": imp})

    return asyncio_files


def clean_asyncio_import(file_path: str, import_info: dict) -> bool:
    """Remove a specific AsyncIO import from a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        line_no = import_info["line_no"]
        if 1 <= line_no <= len(lines):
            line_content = lines[line_no - 1]

            # Check if this is the import we want to remove
            import_name = import_info["name"]

            if import_info["import_type"] == "import":
                # Handle: import asyncio
                if f'import {import_info["module"]}' in line_content:
                    lines.pop(line_no - 1)

            elif import_info["import_type"] == "from_import":
                # Handle: from asyncio import something
                if "," in line_content:
                    # Multi-import line
                    new_line = remove_from_line(line_content, import_name)
                    if new_line.strip().endswith("import"):
                        # Remove entire line if no imports left
                        lines.pop(line_no - 1)
                    else:
                        lines[line_no - 1] = new_line
                else:
                    # Single import
                    lines.pop(line_no - 1)

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            return True

    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False


def remove_from_line(line: str, name_to_remove: str) -> str:
    """Remove a specific import from a multi-import line."""
    if "from " in line and " import " in line:
        parts = line.split(" import ", 1)
        if len(parts) == 2:
            prefix = parts[0] + " import "
            imports_part = parts[1]

            # Split imports and remove the target
            imports = [imp.strip() for imp in imports_part.split(",")]
            imports = [imp for imp in imports if imp != name_to_remove]

            if imports:
                return prefix + ", ".join(imports) + "\n"
            else:
                return prefix + "\n"  # Will be detected as empty and removed

    return line


def main():
    asyncio_files = find_asyncio_files()

    print(f"Found {len(asyncio_files)} AsyncIO imports to clean up")

    cleaned_count = 0
    for item in asyncio_files:
        file_path = item["file"]
        import_info = item["import"]

        if clean_asyncio_import(file_path, import_info):
            print(f"✅ Cleaned {Path(file_path).name}: removed {import_info['name']}")
            cleaned_count += 1
        else:
            print(f"❌ Failed to clean {Path(file_path).name}")

    print(f"\n✨ AsyncIO cleanup complete! Cleaned {cleaned_count} imports")


if __name__ == "__main__":
    main()
