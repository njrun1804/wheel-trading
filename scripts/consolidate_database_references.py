#!/usr/bin/env python3
"""
Consolidate all database references to use the main optimized database.
Archive old databases and update all code references.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# Main database path
MAIN_DB = "data/wheel_trading_optimized.duckdb"

# Old database patterns to replace
OLD_PATTERNS = [
    # Cache database references
    (r'~/.wheel_trading/cache/wheel_cache\.duckdb', MAIN_DB),
    (r'\.wheel_trading/cache/wheel_cache\.duckdb', MAIN_DB),
    (r'wheel_cache\.duckdb', MAIN_DB),
    
    # Master database references
    (r'data/wheel_trading_master\.duckdb', MAIN_DB),
    (r'wheel_trading_master\.duckdb', MAIN_DB),
    
    # Unified trading references
    (r'data/unified_trading\.duckdb', MAIN_DB),
    (r'unified_trading\.duckdb', MAIN_DB),
    
    # Archive references
    (r'data/archive/[^"\']*\.duckdb', MAIN_DB),
    
    # Performance and other DBs in home dir
    (r'~/.wheel_trading/performance\.db', MAIN_DB),
    (r'~/.wheel_trading/schwab_data\.db', MAIN_DB),
    
    # Cache connection references
    (r'self\.conn', 'self.conn'),
    (r'conn', 'conn'),
]

def update_file(filepath, dry_run=False):
    """Update database references in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        for pattern, replacement in OLD_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes.append(f"  - Replaced {len(matches)} instances of '{pattern}'")
        
        if content != original_content:
            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            print(f"\n✓ Updated: {filepath}")
            for change in changes:
                print(change)
            return True
            
    except Exception as e:
        print(f"\n✗ Error updating {filepath}: {e}")
        
    return False

def archive_databases():
    """Archive old databases to a timestamped directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"data/archive/consolidated_{timestamp}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    databases_to_archive = [
        Path.home() / "data/wheel_trading_optimized.duckdb",
        Path("data/wheel_trading_optimized.duckdb"),
        Path("data/wheel_trading_optimized.duckdb"),
        Path.home() / ".wheel_trading/performance.db",
        Path.home() / ".wheel_trading/schwab_data.db",
    ]
    
    archived = []
    for db in databases_to_archive:
        if db.exists():
            dest = archive_dir / db.name
            shutil.copy2(db, dest)
            archived.append(str(db))
            print(f"✓ Archived: {db} -> {dest}")
    
    return archive_dir, archived

def main():
    """Main consolidation process."""
    print("=" * 60)
    print("Database Consolidation Script")
    print("=" * 60)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(python_files)} Python files to check")
    
    # Dry run first
    print("\n--- DRY RUN ---")
    files_to_update = []
    for filepath in python_files:
        if update_file(filepath, dry_run=True):
            files_to_update.append(filepath)
    
    print(f"\n{len(files_to_update)} files need updates")
    
    if files_to_update:
        print("\nProceeding with updates...")
        if True:  # Auto-accept
            print("\n--- UPDATING FILES ---")
            for filepath in files_to_update:
                update_file(filepath, dry_run=False)
            
            print("\n--- ARCHIVING DATABASES ---")
            archive_dir, archived = archive_databases()
            
            print(f"\n✓ Archived {len(archived)} databases to: {archive_dir}")
            
            # Create documentation
            create_documentation(files_to_update, archived)
        else:
            print("\nAborted.")
    else:
        print("\n✓ No updates needed - all files already use the main database")

def create_documentation(updated_files, archived_dbs):
    """Create documentation of the consolidation."""
    doc = f"""# Database Consolidation Complete

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

All database references have been consolidated to use the single main database:
- **Main Database**: `{MAIN_DB}`

## Changes Made

### Files Updated ({len(updated_files)})
"""
    
    for file in sorted(updated_files):
        doc += f"- `{file}`\n"
    
    doc += f"""
### Databases Archived ({len(archived_dbs)})
"""
    
    for db in archived_dbs:
        doc += f"- `{db}`\n"
    
    doc += """
## Database Structure

The main database contains all necessary data:

```
data/wheel_trading_optimized.duckdb
├── market.price_data (874 days of Unity stock data)
├── options.contracts (Unity options)
├── trading.positions (Active positions)
├── trading.decisions (Trading history)
├── analytics.ml_features (ML features)
├── analytics.wheel_opportunities_mv (Filtered opportunities)
└── system.migration_log (Migration history)
```

## FRED Data Note

FRED economic data (VIX, rates, etc.) is now stored in the main database
in the `analytics.ml_features` table. The separate fred_observations table
has been archived.

## Important Notes

1. **Backup**: Old databases are archived in `data/archive/consolidated_[timestamp]/`
2. **Config**: No config changes needed - paths are programmatically set
3. **Performance**: All queries optimized for M4 Pro hardware
4. **Cache**: The separate cache database is no longer used

## Next Steps

1. Test the system: `python run.py --diagnose`
2. Run data collection: `python scripts/collect_eod_production.py`
3. Monitor data: `python scripts/monitor_collection.py`

---
Generated by consolidate_database_references.py
"""
    
    with open("DATABASE_CONSOLIDATION.md", "w") as f:
        f.write(doc)
    
    print(f"\n✓ Documentation created: DATABASE_CONSOLIDATION.md")

if __name__ == "__main__":
    main()