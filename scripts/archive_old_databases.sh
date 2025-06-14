#!/bin/bash
# Archive old market databases after successful migration

set -e  # Exit on error

ARCHIVE_DIR="$HOME/wheel_trading_archives/$(date +%Y%m%d_%H%M%S)"
ARCHIVE_FILE="wheel_trading_old_dbs_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "🗄️  Archiving old market databases..."
echo "   Archive directory: $ARCHIVE_DIR"

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Function to safely move and compress databases
archive_database() {
    local db_path=$1
    local db_name=$(basename "$db_path")
    
    if [ -f "$db_path" ]; then
        echo "  📦 Archiving $db_name..."
        cp -v "$db_path" "$ARCHIVE_DIR/"
        
        # Get file size
        size=$(du -h "$db_path" | cut -f1)
        echo "     Size: $size"
        
        # Create checksum
        if command -v shasum >/dev/null 2>&1; then
            shasum -a 256 "$db_path" > "$ARCHIVE_DIR/$db_name.sha256"
            echo "     Checksum created"
        fi
    else
        echo "  ⏭️  Skipping $db_name (not found)"
    fi
}

# Archive market databases (not system databases)
echo ""
echo "📊 Archiving market databases..."
archive_database "data/wheel_trading_master.duckdb"
archive_database "data/unified_trading.duckdb"
archive_database "data/cache/wheel_cache.duckdb"

# Archive parquet files
if [ -d "data/unity-options" ]; then
    echo ""
    echo "📈 Archiving parquet files..."
    cp -r "data/unity-options" "$ARCHIVE_DIR/"
    echo "  ✓ Unity options parquet files archived"
fi

# Create archive metadata
cat > "$ARCHIVE_DIR/ARCHIVE_INFO.txt" << EOF
Wheel Trading Database Archive
==============================
Created: $(date)
Purpose: Archive of old databases after migration to optimized structure

Contents:
---------
$(ls -lh "$ARCHIVE_DIR" | grep -v ARCHIVE_INFO)

Migration Summary:
-----------------
- Migrated to: data/wheel_trading_optimized.duckdb
- Migration date: $(date)
- Old structure: 3 databases, 30+ tables
- New structure: 1 database, 12 optimized tables

Original Locations:
------------------
- wheel_trading_master.duckdb -> data/
- unified_trading.duckdb -> data/
- wheel_cache.duckdb -> data/cache/
- unity-options/ -> data/

Notes:
------
- System databases (slice_cache.duckdb, etc.) were NOT archived
- These remain in their original locations for continued use
- The home directory cache (~/.wheel_trading/) was NOT moved

To Restore:
----------
cd $ARCHIVE_DIR
cp wheel_trading_master.duckdb ../../data/
cp unified_trading.duckdb ../../data/
cp wheel_cache.duckdb ../../data/cache/
cp -r unity-options ../../data/
EOF

# Create compressed archive
echo ""
echo "🗜️  Creating compressed archive..."
cd "$HOME/wheel_trading_archives"
tar -czf "$ARCHIVE_FILE" "$(basename "$ARCHIVE_DIR")"
ARCHIVE_SIZE=$(du -h "$ARCHIVE_FILE" | cut -f1)

echo "  ✓ Created $ARCHIVE_FILE ($ARCHIVE_SIZE)"

# Optionally remove uncompressed files
echo ""
read -p "Remove uncompressed archive directory? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$ARCHIVE_DIR"
    echo "  ✓ Removed uncompressed files"
fi

# Show next steps
echo ""
echo "✅ Archive complete!"
echo ""
echo "📍 Archive location: $HOME/wheel_trading_archives/$ARCHIVE_FILE"
echo ""
echo "🗑️  To remove old databases (after confirming new system works):"
echo "   rm data/wheel_trading_master.duckdb"
echo "   rm data/unified_trading.duckdb"
echo "   rm data/cache/wheel_cache.duckdb"
echo "   rm -rf data/unity-options"
echo ""
echo "🔄 To restore from archive:"
echo "   cd $HOME/wheel_trading_archives"
echo "   tar -xzf $ARCHIVE_FILE"
echo "   cd $(basename "$ARCHIVE_DIR" .tar.gz)"
echo "   # Follow instructions in ARCHIVE_INFO.txt"