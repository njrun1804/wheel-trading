#!/usr/bin/env python3
"""Archive all deprecated files to the archive directory."""

import shutil
from datetime import datetime
from pathlib import Path

# Define deprecated files to archive
DEPRECATED_FILES = [
    'src/unity_wheel/risk/analytics_deprecated.py',
    'src/unity_wheel/risk/advanced_financial_modeling_deprecated.py',
    'src/unity_wheel/utils/position_sizing_deprecated.py',
    'src/unity_wheel/math/options_deprecated.py',
    'src/unity_wheel/analytics/decision_engine_deprecated.py',
    'src/unity_wheel/analytics/decision_engine.py.DEPRECATED',
    'src/unity_wheel/math/options.py.DEPRECATED',
    'src/unity_wheel/risk/analytics.py.DEPRECATED',
    'src/unity_wheel/risk/advanced_financial_modeling.py.DEPRECATED',
    'src/unity_wheel/utils/position_sizing.py.DEPRECATED',
]


def archive_deprecated_files():
    """Move deprecated files to archive directory."""
    project_root = Path(__file__).parent.parent
    archive_dir = project_root / 'archive' / f'{datetime.now().strftime("%Y%m%d")}_unified_refactor' / 'deprecated_components'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    print("🗄️ Archiving deprecated files...")
    
    for file_path in DEPRECATED_FILES:
        source = project_root / file_path
        
        if source.exists():
            # Create subdirectory structure in archive
            relative_path = Path(file_path).relative_to('src/unity_wheel')
            dest_dir = archive_dir / relative_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest = dest_dir / relative_path.name
            
            # Move the file
            shutil.move(str(source), str(dest))
            moved_count += 1
            
            print(f"  ✅ Archived: {file_path}")
            print(f"     → {dest.relative_to(project_root)}")
        else:
            print(f"  ⏭️ Skipped (not found): {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"  Files archived: {moved_count}")
    print(f"  Archive location: {archive_dir.relative_to(project_root)}")
    
    # Create archive README
    readme_path = archive_dir / 'README.md'
    readme_content = f"""# Deprecated Components Archive

Archived on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

These files have been deprecated as part of the unified refactor.
They are kept here for reference and should not be used in production.

## Archived Files
"""
    
    for file_path in archive_dir.rglob('*.py'):
        readme_content += f"- {file_path.relative_to(archive_dir)}\n"
    
    readme_path.write_text(readme_content)
    print(f"\n✅ Created archive README: {readme_path.relative_to(project_root)}")


if __name__ == '__main__':
    archive_deprecated_files()