#!/bin/bash
# Cleanup script for file system artifacts

echo "=== Unity Wheel Trading Bot - File System Cleanup ==="
echo "This script will remove duplicate files and redundant directories"
echo ""

# Create backup directory for safety
BACKUP_DIR=".cleanup_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 1. List all duplicate files with " 2" suffix
echo "Finding duplicate files with ' 2' suffix..."
find . -name "* 2.*" -type f > "$BACKUP_DIR/duplicate_files.txt"
COUNT=$(wc -l < "$BACKUP_DIR/duplicate_files.txt")
echo "Found $COUNT duplicate files"

# 2. Remove duplicate files
if [ $COUNT -gt 0 ]; then
    echo "Removing duplicate files..."
    while IFS= read -r file; do
        echo "  Removing: $file"
        rm -f "$file"
    done < "$BACKUP_DIR/duplicate_files.txt"
    echo "✓ Removed $COUNT duplicate files"
fi

# 3. Remove redundant engine directories
echo ""
echo "Removing redundant engine directories..."
for dir in ml_engine risk_engine strategy_engine; do
    if [ -d "$dir" ]; then
        echo "  Removing: $dir/"
        rm -rf "$dir"
        echo "✓ Removed $dir/"
    fi
done

# 4. Remove nested duplicate structure
echo ""
echo "Checking for nested duplicate structure..."
if [ -d "Documents/com~apple~CloudDocs" ]; then
    echo "  Removing: Documents/com~apple~CloudDocs/"
    rm -rf "Documents/com~apple~CloudDocs"
    echo "✓ Removed nested duplicate"
fi

# 5. Clean up __pycache__ and .pyc files
echo ""
echo "Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ Cleaned Python cache files"

# 6. Clean up .DS_Store files (macOS)
echo ""
echo "Cleaning .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null
echo "✓ Cleaned .DS_Store files"

echo ""
echo "=== Cleanup Complete ==="
echo "Backup information saved to: $BACKUP_DIR/"
echo ""
echo "Summary:"
echo "- Removed $COUNT duplicate files with ' 2' suffix"
echo "- Removed redundant engine directories"
echo "- Cleaned Python cache files"
echo "- Cleaned macOS system files"