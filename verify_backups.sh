#!/bin/bash
# Backup Verification Script
# Automatically verify database backup integrity

BACKUP_DIR="backups/database_consolidation"
LOG_FILE="backup_verification.log"

echo "$(date): Starting backup verification" >> $LOG_FILE

if [ ! -d "$BACKUP_DIR" ]; then
    echo "$(date): ERROR - Backup directory not found" >> $LOG_FILE
    exit 1
fi

VERIFIED=0
FAILED=0

for backup_file in "$BACKUP_DIR"/*.db; do
    if [ -f "$backup_file" ]; then
        # Test SQLite database integrity
        if sqlite3 "$backup_file" "PRAGMA integrity_check;" > /dev/null 2>&1; then
            echo "$(date): VERIFIED - $(basename "$backup_file")" >> $LOG_FILE
            ((VERIFIED++))
        else
            echo "$(date): FAILED - $(basename "$backup_file")" >> $LOG_FILE
            ((FAILED++))
        fi
    fi
done

echo "$(date): Verification complete - $VERIFIED verified, $FAILED failed" >> $LOG_FILE

if [ $FAILED -gt 0 ]; then
    echo "$(date): WARNING - $FAILED backup files failed verification" >> $LOG_FILE
    exit 1
fi

exit 0
