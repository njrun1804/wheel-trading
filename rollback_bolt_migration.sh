#!/bin/bash
# BOLT to BOB Migration Rollback Script
# Generated: 2025-06-16T15:38:48.959035

set -e

echo "🚨 EMERGENCY ROLLBACK: Restoring BOLT system..."

# Stop all running processes
pkill -f "bob_" || true
pkill -f "bolt_cli" || true

# Restore from backup
if [ -d "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848" ]; then
    echo "Restoring from backup: /Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848"
    
    # Restore BOLT directory
    if [ -d "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848/bolt_original" ]; then
        rm -rf "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt"
        cp -r "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848/bolt_original" "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt"
        echo "✅ BOLT directory restored"
    fi
    
    # Restore BOB directory to pre-migration state
    if [ -d "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848/bob_original" ]; then
        rm -rf "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bob"
        cp -r "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848/bob_original" "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bob" 
        echo "✅ BOB directory restored"
    fi
    
    echo "🔄 Rollback complete. System restored to pre-migration state."
else
    echo "❌ Backup directory not found: /Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/bolt_migration_backup_20250616_153848"
    echo "Manual recovery required."
    exit 1
fi
