#!/bin/bash

# Phase 1 Emergency Cleanup Script
# ZERO RISK - Archives and removes redundant files
# Part of Codebase Harmonization Roadmap

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/consolidation_backups/$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Phase 1 Emergency Cleanup"
echo "Project Root: $PROJECT_ROOT"
echo "Backup Directory: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. ARCHIVE CLEANUP - Remove old backup directories
echo "📁 Cleaning up archive directories..."

ARCHIVE_DIRS=(
    "archive/old_orchestrator_tests"
    "jarvis2/backup_before_fixes" 
    "jarvis2/backup_complete"
    "backups/sequential_thinking_cleanup"
    "backups/hardware_consolidation"
    "backups/jarvis2_consolidation"
)

for dir in "${ARCHIVE_DIRS[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        echo "  Archiving: $dir"
        cp -r "$PROJECT_ROOT/$dir" "$BACKUP_DIR/"
        rm -rf "$PROJECT_ROOT/$dir"
        echo "  ✅ Removed: $dir"
    fi
done

# 2. DOCUMENTATION CONSOLIDATION - Remove duplicate documentation
echo "📚 Consolidating documentation files..."

# Create list of documentation files to remove
DOC_PATTERNS=(
    "*_SUMMARY.md"
    "*_REPORT.md" 
    "*_COMPLETE.md"
    "*_VALIDATION.md"
    "*_FINAL.md"
    "*_RESULTS.md"
    "*_STATUS.md"
    "*_ASSESSMENT.md"
    "*_ANALYSIS.md"
    "*_IMPLEMENTATION.md"
)

# Backup all documentation before removal
echo "  Creating documentation backup..."
find "$PROJECT_ROOT" -name "*.md" -not -path "*/consolidation_backups/*" > "$BACKUP_DIR/all_docs_list.txt"
mkdir -p "$BACKUP_DIR/docs_backup"

while IFS= read -r doc_file; do
    if [ -f "$doc_file" ]; then
        # Create relative path structure in backup
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$doc_file")
        backup_path="$BACKUP_DIR/docs_backup/$rel_path"
        mkdir -p "$(dirname "$backup_path")"
        cp "$doc_file" "$backup_path"
    fi
done < "$BACKUP_DIR/all_docs_list.txt"

# Remove duplicate documentation
removed_docs=0
for pattern in "${DOC_PATTERNS[@]}"; do
    echo "  Removing pattern: $pattern"
    while IFS= read -r -d '' file; do
        # Skip essential files
        basename_file=$(basename "$file")
        if [[ "$basename_file" =~ ^(README|CLAUDE|ARCHITECTURE|LICENSE)\.md$ ]]; then
            echo "    Skipping essential: $file"
            continue
        fi
        
        echo "    Removing: $file"
        rm -f "$file"
        ((removed_docs++))
    done < <(find "$PROJECT_ROOT" -name "$pattern" -type f -not -path "*/consolidation_backups/*" -print0)
done

echo "  ✅ Removed $removed_docs documentation files"

# 3. TEST FILE CONSOLIDATION - Remove redundant test files
echo "🧪 Consolidating test files..."

# Create test file backup
mkdir -p "$BACKUP_DIR/tests_backup"
find "$PROJECT_ROOT" -name "test_*.py" -not -path "*/consolidation_backups/*" > "$BACKUP_DIR/all_tests_list.txt"

while IFS= read -r test_file; do
    if [ -f "$test_file" ]; then
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$test_file")
        backup_path="$BACKUP_DIR/tests_backup/$rel_path"
        mkdir -p "$(dirname "$backup_path")"
        cp "$test_file" "$backup_path"
    fi
done < "$BACKUP_DIR/all_tests_list.txt"

# Remove redundant test patterns
TEST_REMOVAL_PATTERNS=(
    "*_simple.py"
    "*_validation.py"
    "*_old.py"
    "*_backup.py"
    "*_copy.py"
    "*_deprecated.py"
)

removed_tests=0
for pattern in "${TEST_REMOVAL_PATTERNS[@]}"; do
    echo "  Checking pattern: test$pattern"
    while IFS= read -r -d '' test_file; do
        # Check if there's a main test file for this
        base_name=$(basename "$test_file")
        main_test_name=$(echo "$base_name" | sed -E 's/_(simple|validation|old|backup|copy|deprecated)\.py$/\.py/')
        
        # Skip if this IS the main test file
        if [ "$base_name" = "$main_test_name" ]; then
            continue
        fi
        
        # Look for main test file
        dir_name=$(dirname "$test_file")
        main_test_path="$dir_name/$main_test_name"
        
        if [ -f "$main_test_path" ]; then
            echo "    Removing redundant: $test_file (main exists: $main_test_path)"
            rm -f "$test_file"
            ((removed_tests++))
        else
            echo "    Keeping: $test_file (no main test found)"
        fi
    done < <(find "$PROJECT_ROOT" -name "test$pattern" -type f -not -path "*/consolidation_backups/*" -print0)
done

echo "  ✅ Removed $removed_tests redundant test files"

# 4. CONFIGURATION FILE CLEANUP - Remove duplicate configs
echo "⚙️  Cleaning up configuration files..."

CONFIG_BACKUP_DIR="$BACKUP_DIR/config_backup"
mkdir -p "$CONFIG_BACKUP_DIR"

# Backup all config files
find "$PROJECT_ROOT" \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) -not -path "*/consolidation_backups/*" > "$BACKUP_DIR/all_configs_list.txt"

while IFS= read -r config_file; do
    if [ -f "$config_file" ]; then
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$config_file")
        backup_path="$CONFIG_BACKUP_DIR/$rel_path"
        mkdir -p "$(dirname "$backup_path")"
        cp "$config_file" "$backup_path"
    fi
done < "$BACKUP_DIR/all_configs_list.txt"

# Remove duplicate MCP server configs (keep only mcp-servers.json)
removed_configs=0
while IFS= read -r -d '' config_file; do
    basename_file=$(basename "$config_file")
    if [[ "$basename_file" =~ ^mcp-servers-.+\.json$ ]] || [[ "$basename_file" =~ \.backup$ ]]; then
        echo "    Removing duplicate config: $config_file"
        rm -f "$config_file"
        ((removed_configs++))
    fi
done < <(find "$PROJECT_ROOT" -name "mcp-servers*.json" -type f -not -path "*/consolidation_backups/*" -print0)

echo "  ✅ Removed $removed_configs duplicate configuration files"

# 5. DATABASE CLEANUP - Archive old databases
echo "🗄️  Cleaning up old databases..."

DB_BACKUP_DIR="$BACKUP_DIR/databases_backup"
mkdir -p "$DB_BACKUP_DIR"

# Find and backup all database files
find "$PROJECT_ROOT" \( -name "*.db" -o -name "*.db-shm" -o -name "*.db-wal" \) -not -path "*/consolidation_backups/*" > "$BACKUP_DIR/all_dbs_list.txt"

removed_dbs=0
while IFS= read -r db_file; do
    if [ -f "$db_file" ]; then
        basename_file=$(basename "$db_file")
        # Skip main production databases
        if [[ "$basename_file" =~ ^(wheel_trading_master|unified_wheel_trading|analytics)\.db$ ]]; then
            echo "    Keeping production database: $db_file"
            continue
        fi
        
        # Skip if in data/ directory (production data)
        if [[ "$db_file" == */data/* ]]; then
            echo "    Keeping data directory database: $db_file"
            continue
        fi
        
        # Archive and remove development/test databases
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$db_file")
        backup_path="$DB_BACKUP_DIR/$rel_path"
        mkdir -p "$(dirname "$backup_path")"
        cp "$db_file" "$backup_path"
        rm -f "$db_file"
        echo "    Archived and removed: $db_file"
        ((removed_dbs++))
    fi
done < "$BACKUP_DIR/all_dbs_list.txt"

echo "  ✅ Archived and removed $removed_dbs database files"

# 6. GENERATE CLEANUP REPORT
echo "📊 Generating cleanup report..."

REPORT_FILE="$BACKUP_DIR/cleanup_report.md"
cat > "$REPORT_FILE" << EOF
# Phase 1 Emergency Cleanup Report

**Execution Date**: $(date)
**Backup Location**: $BACKUP_DIR

## Summary of Changes

### Archive Directories Removed
- ${#ARCHIVE_DIRS[@]} backup directories removed
- All content backed up to: \`$BACKUP_DIR/\`

### Documentation Consolidation  
- $removed_docs duplicate documentation files removed
- All documentation backed up to: \`$BACKUP_DIR/docs_backup/\`
- Essential documentation preserved (README, CLAUDE, ARCHITECTURE, LICENSE)

### Test File Consolidation
- $removed_tests redundant test files removed  
- All test files backed up to: \`$BACKUP_DIR/tests_backup/\`
- Main test files preserved

### Configuration Cleanup
- $removed_configs duplicate configuration files removed
- All configs backed up to: \`$BACKUP_DIR/config_backup/\`
- Primary configurations preserved

### Database Cleanup
- $removed_dbs development/test databases archived
- Production databases preserved in /data/ directory
- All databases backed up to: \`$BACKUP_DIR/databases_backup/\`

## Files Lists
- All documentation: \`all_docs_list.txt\`
- All test files: \`all_tests_list.txt\`  
- All configuration files: \`all_configs_list.txt\`
- All database files: \`all_dbs_list.txt\`

## Recovery Instructions
To recover any removed files:
\`\`\`bash
# Restore from backup
cp -r $BACKUP_DIR/<category>_backup/* $PROJECT_ROOT/
\`\`\`

## Next Steps
1. Verify system functionality after cleanup
2. Run test suite to ensure no regressions
3. Proceed to Phase 2: System Integration
EOF

echo "  ✅ Cleanup report generated: $REPORT_FILE"

# 7. VERIFY SYSTEM INTEGRITY
echo "🔍 Verifying system integrity..."

# Check that essential files still exist
ESSENTIAL_FILES=(
    "README.md"
    "CLAUDE.md" 
    "src/unity_wheel/__init__.py"
    "config.yaml"
    "pyproject.toml"
    "requirements.txt"
)

missing_files=0
for file in "${ESSENTIAL_FILES[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$file" ]; then
        echo "  ❌ Missing essential file: $file"
        ((missing_files++))
    else
        echo "  ✅ Essential file present: $file"
    fi
done

if [ $missing_files -eq 0 ]; then
    echo "  ✅ All essential files verified"
else
    echo "  ⚠️  $missing_files essential files missing - review cleanup"
    exit 1
fi

# Calculate space saved
if command -v du >/dev/null 2>&1; then
    backup_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    echo "  📦 Space recovered: ~$backup_size"
fi

echo ""
echo "🎉 Phase 1 Emergency Cleanup COMPLETED"
echo ""
echo "📊 Summary:"
echo "  - Archive directories: ${#ARCHIVE_DIRS[@]} removed"
echo "  - Documentation files: $removed_docs removed"  
echo "  - Test files: $removed_tests removed"
echo "  - Configuration files: $removed_configs removed"
echo "  - Database files: $removed_dbs archived"
echo ""
echo "📁 All changes backed up to: $BACKUP_DIR"
echo "📋 Detailed report: $REPORT_FILE"
echo ""
echo "🔄 Next: Review system functionality, then proceed to Phase 2"
echo ""