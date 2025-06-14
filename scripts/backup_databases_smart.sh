#!/bin/bash
# Smart backup script that distinguishes between market and system databases

set -e  # Exit on error

BACKUP_DIR="$HOME/wheel_trading_backups/$(date +%Y%m%d_%H%M%S)"
MARKET_BACKUP_DIR="$BACKUP_DIR/market_data"
SYSTEM_BACKUP_DIR="$BACKUP_DIR/system_data"

echo "🔵 Creating backup directories..."
mkdir -p "$MARKET_BACKUP_DIR"
mkdir -p "$SYSTEM_BACKUP_DIR"

# Backup market data databases
echo "📊 Backing up MARKET DATA databases..."
if [ -f "data/wheel_trading_master.duckdb" ]; then
    cp -v data/wheel_trading_master.duckdb "$MARKET_BACKUP_DIR/"
    echo "  ✓ wheel_trading_master.duckdb ($(du -h data/wheel_trading_master.duckdb | cut -f1))"
fi

if [ -f "data/unified_trading.duckdb" ]; then
    cp -v data/unified_trading.duckdb "$MARKET_BACKUP_DIR/"
    echo "  ✓ unified_trading.duckdb ($(du -h data/unified_trading.duckdb | cut -f1))"
fi

if [ -f "data/cache/wheel_cache.duckdb" ]; then
    cp -v data/cache/wheel_cache.duckdb "$MARKET_BACKUP_DIR/wheel_cache_project.duckdb"
    echo "  ✓ wheel_cache.duckdb from project ($(du -h data/cache/wheel_cache.duckdb | cut -f1))"
fi

# Backup parquet files
if [ -d "data/unity-options" ]; then
    echo "📈 Backing up parquet files..."
    cp -r data/unity-options "$MARKET_BACKUP_DIR/"
    echo "  ✓ Unity options parquet files"
fi

# Check the home wheel_cache - it might contain market data
if [ -f "$HOME/.wheel_trading/cache/wheel_cache.duckdb" ]; then
    echo "🔍 Analyzing ~/.wheel_trading/cache/wheel_cache.duckdb..."
    # This could be market or system data - backup to both for safety
    cp -v "$HOME/.wheel_trading/cache/wheel_cache.duckdb" "$MARKET_BACKUP_DIR/wheel_cache_home.duckdb"
    cp -v "$HOME/.wheel_trading/cache/wheel_cache.duckdb" "$SYSTEM_BACKUP_DIR/wheel_cache_home.duckdb"
    echo "  ✓ Backed up to both locations for safety ($(du -h "$HOME/.wheel_trading/cache/wheel_cache.duckdb" | cut -f1))"
fi

# Backup system databases (if they exist)
echo "🖥️  Backing up SYSTEM databases (will not be migrated)..."
if [ -f "$HOME/.wheel_trading/cache/slice_cache.duckdb" ]; then
    cp -v "$HOME/.wheel_trading/cache/slice_cache.duckdb" "$SYSTEM_BACKUP_DIR/"
    echo "  ✓ slice_cache.duckdb (code embeddings)"
fi

# Look for any code graph databases
find . -name "code_graph.duckdb" -type f 2>/dev/null | while read -r db; do
    cp -v "$db" "$SYSTEM_BACKUP_DIR/$(basename "$db")"
    echo "  ✓ $(basename "$db") (code analysis)"
done

# Create restore scripts
cat > "$MARKET_BACKUP_DIR/restore_market_data.sh" << 'EOF'
#!/bin/bash
echo "🔄 Restoring market data databases..."
cp -v wheel_trading_master.duckdb ../../../data/
cp -v unified_trading.duckdb ../../../data/
cp -v wheel_cache_project.duckdb ../../../data/cache/wheel_cache.duckdb
[ -d unity-options ] && cp -r unity-options ../../../data/
echo "✅ Market data restore complete!"
EOF

cat > "$SYSTEM_BACKUP_DIR/restore_system_data.sh" << 'EOF'
#!/bin/bash
echo "🔄 Restoring system databases..."
[ -f slice_cache.duckdb ] && cp -v slice_cache.duckdb ~/.wheel_trading/cache/
[ -f wheel_cache_home.duckdb ] && cp -v wheel_cache_home.duckdb ~/.wheel_trading/cache/wheel_cache.duckdb
echo "✅ System data restore complete!"
EOF

chmod +x "$MARKET_BACKUP_DIR/restore_market_data.sh"
chmod +x "$SYSTEM_BACKUP_DIR/restore_system_data.sh"

# Create summary
cat > "$BACKUP_DIR/backup_summary.txt" << EOF
Backup created at: $(date)
=================================

Market Data (will be migrated):
- wheel_trading_master.duckdb
- unified_trading.duckdb  
- wheel_cache.duckdb (project)
- unity-options parquet files

System Data (will NOT be migrated):
- slice_cache.duckdb (if exists)
- code_graph.duckdb (if exists)
- wheel_cache.duckdb (home) - backed up to both locations

Restore scripts:
- $MARKET_BACKUP_DIR/restore_market_data.sh
- $SYSTEM_BACKUP_DIR/restore_system_data.sh

Total backup size: $(du -sh "$BACKUP_DIR" | cut -f1)
EOF

echo ""
echo "✅ Backup complete!"
echo "📁 Location: $BACKUP_DIR"
echo "📄 Summary: $BACKUP_DIR/backup_summary.txt"
echo ""
echo "To restore market data: cd $MARKET_BACKUP_DIR && ./restore_market_data.sh"
echo "To restore system data: cd $SYSTEM_BACKUP_DIR && ./restore_system_data.sh"