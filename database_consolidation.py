#!/usr/bin/env python3
"""
Database Consolidation Tool

Consolidates multiple database files and removes unused ones.
"""

import shutil
import sqlite3
from pathlib import Path

from database_manager import get_database_manager
from unified_config import get_unified_config


def consolidate_databases():
    """Consolidate multiple database files into unified structure."""

    config = get_unified_config()
    db_manager = get_database_manager()

    # Map old database files to new unified structure
    consolidation_map = {
        "jarvis2_audit.db": config.meta.evolution_db,
        "jarvis2_reaudit.db": config.meta.evolution_db,  # Merge with evolution
        "jarvis2_strategy.db": config.meta.evolution_db,  # Merge with evolution
        "meta_daemon_continuous.db": config.meta.monitoring_db,
        "meta_evolution.db": config.meta.evolution_db,  # Keep as is
        "meta_monitoring.db": config.meta.monitoring_db,  # Keep as is
        "meta_reality_learning.db": config.meta.reality_db,  # Keep as is
    }

    consolidated_dbs = set(consolidation_map.values())

    print("🗄️ Database Consolidation Starting...")
    print(
        f"Consolidating {len(consolidation_map)} files into {len(consolidated_dbs)} databases"
    )

    # Create backup directory
    backup_dir = Path("backups/database_consolidation")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Process each old database
    for old_db, new_db in consolidation_map.items():
        old_path = Path(old_db)
        if not old_path.exists():
            continue

        print(f"📦 Processing {old_db} -> {new_db}")

        # Backup original
        backup_path = backup_dir / old_db
        shutil.copy2(old_path, backup_path)
        print(f"   ✅ Backed up to {backup_path}")

        # If it's not being renamed, skip the migration
        if old_db == new_db:
            continue

        # Migrate data to new database
        try:
            migrate_database_data(old_path, new_db)
            print(f"   ✅ Migrated data to {new_db}")

            # Remove old database after successful migration
            old_path.unlink()
            print(f"   🗑️ Removed {old_db}")

        except Exception as e:
            print(f"   ❌ Migration failed: {e}")
            continue

    print("\n📊 Final Database Structure:")
    stats = db_manager.get_database_stats()
    for name, data in stats.items():
        if "error" not in data:
            print(f"   {name}: {data['size_mb']}MB, {data['table_count']} tables")
        else:
            print(f"   {name}: Error - {data['error']}")

    print("\n✅ Database consolidation complete!")


def migrate_database_data(source_path: Path, target_db: str):
    """Migrate data from source database to target database."""

    # Connect to source database
    source_conn = sqlite3.connect(source_path)
    source_conn.row_factory = sqlite3.Row

    # Get database manager connection to target
    db_manager = get_database_manager()

    with db_manager.get_connection(target_db.replace(".db", "")) as target_conn:
        # Get all tables from source
        cursor = source_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            print(f"     Migrating table: {table}")

            # Get table schema
            cursor = source_conn.execute(
                f"SELECT sql FROM sqlite_master WHERE name = '{table}'"
            )
            schema_row = cursor.fetchone()
            if not schema_row:
                continue

            schema = schema_row[0]

            # Create table in target (ignore if exists)
            try:
                target_conn.execute(schema)
            except sqlite3.Error:
                pass  # Table might already exist

            # Copy data
            cursor = source_conn.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

            if rows:
                # Get column names
                columns = [description[0] for description in cursor.description]
                placeholders = ",".join(["?" for _ in columns])

                # Insert data (ignore conflicts)
                for row in rows:
                    try:
                        target_conn.execute(
                            f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",
                            tuple(row),
                        )
                    except sqlite3.Error as e:
                        print(f"       Warning: Failed to insert row: {e}")

        target_conn.commit()

    source_conn.close()


def cleanup_orphaned_files():
    """Clean up any orphaned database files."""

    # Look for any remaining .db files that aren't in our unified structure
    config = get_unified_config()
    expected_dbs = {
        config.meta.evolution_db,
        config.meta.monitoring_db,
        config.meta.reality_db,
    }

    orphaned = []
    for db_file in Path(".").glob("*.db"):
        if db_file.name not in expected_dbs:
            orphaned.append(db_file)

    if orphaned:
        print(f"\n🧹 Found {len(orphaned)} orphaned database files:")
        for db_file in orphaned:
            backup_path = Path("backups/database_consolidation") / db_file.name
            shutil.move(db_file, backup_path)
            print(f"   Moved {db_file.name} to backups")
    else:
        print("\n✅ No orphaned database files found")


if __name__ == "__main__":
    consolidate_databases()
    cleanup_orphaned_files()
    print("\n🎉 Database consolidation complete!")
