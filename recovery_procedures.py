#!/usr/bin/env python3
"""
Recovery Procedures for BOB Migration Cleanup
=============================================

Provides comprehensive recovery capabilities for files moved or deleted
during the cleanup process. Multiple recovery sources and validation.

Usage:
    python recovery_procedures.py --list-archives          # Show available archives
    python recovery_procedures.py --recover-all            # Recover all from latest archive
    python recovery_procedures.py --recover-category old_einstein
    python recovery_procedures.py --recover-file path/to/file.py
    python recovery_procedures.py --emergency-rollback     # Full system rollback
"""

import os
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Base directory
BASE_DIR = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")

class RecoveryManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.recovery_log = {
            "recovery_date": datetime.now().isoformat(),
            "base_directory": str(base_dir),
            "recovery_sources": [],
            "recovered_items": [],
            "errors": [],
            "validation_results": []
        }
    
    def find_cleanup_archives(self) -> List[Dict[str, Any]]:
        """Find all available cleanup archives"""
        archives = []
        
        # Safe cleanup archives
        archive_pattern = self.base_dir / "archive" / "cleanup_*"
        for archive_dir in self.base_dir.glob("archive/cleanup_*"):
            if archive_dir.is_dir():
                manifest_file = archive_dir / "cleanup_manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file) as f:
                            manifest = json.load(f)
                        
                        archives.append({
                            "type": "safe_cleanup",
                            "path": archive_dir,
                            "manifest": manifest,
                            "date": datetime.fromisoformat(manifest["cleanup_date"]),
                            "categories": list(manifest.get("categories", {}).keys()),
                            "file_count": sum(cat.get("file_count", 0) for cat in manifest.get("categories", {}).values())
                        })
                    except Exception as e:
                        print(f"⚠️  Could not read manifest from {archive_dir}: {e}")
        
        # Migration backups
        backup_patterns = [
            "bob_migration_backup_*.tar.gz", 
            "bolt_migration_backup_*/",
            ".bob_migration_backup_*/",
            "phase1_cleanup_backup_*.tar.gz"
        ]
        
        for pattern in backup_patterns:
            for backup_path in self.base_dir.glob(pattern):
                if backup_path.exists():
                    archives.append({
                        "type": "migration_backup",
                        "path": backup_path,
                        "date": datetime.fromtimestamp(backup_path.stat().st_mtime),
                        "size": backup_path.stat().st_size if backup_path.is_file() else None
                    })
        
        # Sort by date (newest first)
        archives.sort(key=lambda x: x["date"], reverse=True)
        return archives
    
    def find_deletion_logs(self) -> List[Path]:
        """Find deletion logs from aggressive cleanup"""
        return list(self.base_dir.glob("deletion_log_*.json"))
    
    def list_recovery_sources(self) -> Dict[str, Any]:
        """List all available recovery sources"""
        archives = self.find_cleanup_archives()
        deletion_logs = self.find_deletion_logs()
        
        print("🗂️  Available Recovery Sources:")
        print("=" * 50)
        
        # Safe cleanup archives
        safe_archives = [a for a in archives if a["type"] == "safe_cleanup"]
        if safe_archives:
            print(f"\n📦 Safe Cleanup Archives ({len(safe_archives)}):")
            for i, archive in enumerate(safe_archives, 1):
                age_hours = (datetime.now() - archive["date"]).total_seconds() / 3600
                print(f"   {i}. {archive['path'].name}")
                print(f"      Date: {archive['date'].strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f}h ago)")
                print(f"      Categories: {', '.join(archive['categories'])}")
                print(f"      Files: {archive['file_count']}")
        
        # Migration backups
        migration_backups = [a for a in archives if a["type"] == "migration_backup"]
        if migration_backups:
            print(f"\n💾 Migration Backups ({len(migration_backups)}):")
            for i, backup in enumerate(migration_backups, 1):
                age_hours = (datetime.now() - backup["date"]).total_seconds() / 3600
                size_str = self._format_bytes(backup["size"]) if backup["size"] else "directory"
                print(f"   {i}. {backup['path'].name}")
                print(f"      Date: {backup['date'].strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f}h ago)")
                print(f"      Size: {size_str}")
        
        # Deletion logs
        if deletion_logs:
            print(f"\n🗑️  Deletion Logs ({len(deletion_logs)}):")
            for i, log_file in enumerate(deletion_logs, 1):
                age_hours = (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).total_seconds() / 3600
                print(f"   {i}. {log_file.name} ({age_hours:.1f}h ago)")
        
        return {
            "safe_archives": safe_archives,
            "migration_backups": migration_backups,
            "deletion_logs": deletion_logs
        }
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
    
    def recover_from_safe_archive(self, archive_path: Path, category: Optional[str] = None, 
                                 specific_file: Optional[str] = None, dry_run: bool = False) -> bool:
        """Recover files from safe cleanup archive"""
        manifest_file = archive_path / "cleanup_manifest.json"
        
        if not manifest_file.exists():
            print(f"❌ No manifest found in {archive_path}")
            return False
        
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"❌ Could not read manifest: {e}")
            return False
        
        print(f"🔄 Recovering from archive: {archive_path.name}")
        
        categories_to_recover = []
        if category:
            if category in manifest["categories"]:
                categories_to_recover = [category]
            else:
                print(f"❌ Category '{category}' not found in archive")
                return False
        else:
            categories_to_recover = list(manifest["categories"].keys())
        
        recovered_count = 0
        
        for cat_name in categories_to_recover:
            cat_data = manifest["categories"][cat_name]
            print(f"\n📂 Recovering category: {cat_name}")
            
            for file_info in cat_data.get("moved_files", []):
                source_path = Path(file_info["destination"])  # Where it was moved to
                dest_path = Path(file_info["source"])         # Where to restore it
                
                # Handle specific file recovery
                if specific_file and not str(dest_path).endswith(specific_file):
                    continue
                
                if not source_path.exists():
                    print(f"   ⚠️  Archive file not found: {source_path}")
                    continue
                
                if dry_run:
                    print(f"   📋 Would recover: {dest_path.relative_to(self.base_dir)}")
                else:
                    try:
                        # Create destination directory
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if source_path.is_file():
                            shutil.copy2(source_path, dest_path)
                        else:
                            if dest_path.exists():
                                shutil.rmtree(dest_path)
                            shutil.copytree(source_path, dest_path)
                        
                        print(f"   ✅ Recovered: {dest_path.relative_to(self.base_dir)}")
                        recovered_count += 1
                        
                        self.recovery_log["recovered_items"].append({
                            "source": str(source_path),
                            "destination": str(dest_path),
                            "category": cat_name,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        error_msg = f"Failed to recover {dest_path}: {str(e)}"
                        print(f"   ❌ {error_msg}")
                        self.recovery_log["errors"].append(error_msg)
        
        if not dry_run:
            print(f"\n✅ Recovery completed: {recovered_count} items restored")
        
        return True
    
    def emergency_rollback(self, dry_run: bool = False) -> bool:
        """Emergency rollback to pre-cleanup state"""
        print("🚨 EMERGENCY ROLLBACK PROCEDURE")
        print("=" * 40)
        
        archives = self.find_cleanup_archives()
        if not archives:
            print("❌ No recovery archives found!")
            return False
        
        # Find most recent safe cleanup archive
        safe_archives = [a for a in archives if a["type"] == "safe_cleanup"]
        if not safe_archives:
            print("❌ No safe cleanup archives found!")
            return False
        
        latest_archive = safe_archives[0]  # Already sorted by date
        age_hours = (datetime.now() - latest_archive["date"]).total_seconds() / 3600
        
        print(f"📦 Latest safe archive: {latest_archive['path'].name}")
        print(f"   Created: {latest_archive['date'].strftime('%Y-%m-%d %H:%M:%S')} ({age_hours:.1f}h ago)")
        print(f"   Categories: {', '.join(latest_archive['categories'])}")
        print(f"   Files: {latest_archive['file_count']}")
        
        if not dry_run:
            confirm = input("\n⚠️  Proceed with emergency rollback? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Emergency rollback cancelled")
                return False
        
        # Recover all categories from latest archive
        return self.recover_from_safe_archive(latest_archive["path"], dry_run=dry_run)
    
    def validate_recovery(self) -> Dict[str, Any]:
        """Validate system health after recovery"""
        print("🔍 Validating system health after recovery...")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "overall_health": True
        }
        
        # Check BOB system
        bob_main = self.base_dir / "bob_main"
        validation_results["checks"].append({
            "name": "BOB main CLI exists",
            "passed": bob_main.exists(),
            "details": str(bob_main) if bob_main.exists() else "File not found"
        })
        
        # Check critical directories
        critical_dirs = [
            "bob/",
            "bob/search/", 
            "bob/agents/",
            "bob/config/"
        ]
        
        for dir_path in critical_dirs:
            full_path = self.base_dir / dir_path
            passed = full_path.exists() and full_path.is_dir()
            validation_results["checks"].append({
                "name": f"Critical directory {dir_path}",
                "passed": passed,
                "details": f"Directory {'exists' if passed else 'missing'}"
            })
        
        # Check configuration
        config_file = self.base_dir / "bob" / "config" / "unified_config.yaml"
        passed = config_file.exists()
        validation_results["checks"].append({
            "name": "Unified configuration file",
            "passed": passed,
            "details": str(config_file) if passed else "Config file missing"
        })
        
        # Overall health
        validation_results["overall_health"] = all(check["passed"] for check in validation_results["checks"])
        
        # Print results
        passed_checks = sum(1 for check in validation_results["checks"] if check["passed"])
        total_checks = len(validation_results["checks"])
        
        print(f"Validation results: {passed_checks}/{total_checks} checks passed")
        for check in validation_results["checks"]:
            status = "✅" if check["passed"] else "❌"
            print(f"   {status} {check['name']}: {check['details']}")
        
        if validation_results["overall_health"]:
            print("✅ System health validation PASSED")
        else:
            print("❌ System health validation FAILED")
        
        self.recovery_log["validation_results"].append(validation_results)
        return validation_results
    
    def save_recovery_log(self):
        """Save recovery log"""
        log_file = self.base_dir / f"recovery_log_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.recovery_log, f, indent=2)
        print(f"📄 Recovery log saved: {log_file}")

def main():
    parser = argparse.ArgumentParser(description="Recovery procedures for BOB cleanup")
    parser.add_argument("--list-archives", action="store_true", help="List available recovery sources")
    parser.add_argument("--recover-all", action="store_true", help="Recover all from latest archive")
    parser.add_argument("--recover-category", help="Recover specific category")
    parser.add_argument("--recover-file", help="Recover specific file")
    parser.add_argument("--archive", help="Specific archive to recover from")
    parser.add_argument("--emergency-rollback", action="store_true", help="Emergency rollback to pre-cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Preview recovery without executing")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation checks")
    
    args = parser.parse_args()
    
    recovery = RecoveryManager(BASE_DIR)
    
    if args.list_archives:
        recovery.list_recovery_sources()
        return 0
    
    if args.validate_only:
        recovery.validate_recovery()
        return 0
    
    if args.emergency_rollback:
        success = recovery.emergency_rollback(dry_run=args.dry_run)
        if success and not args.dry_run:
            recovery.validate_recovery()
            recovery.save_recovery_log()
        return 0 if success else 1
    
    # Find archives
    archives = recovery.find_cleanup_archives()
    safe_archives = [a for a in archives if a["type"] == "safe_cleanup"]
    
    if not safe_archives:
        print("❌ No safe cleanup archives found for recovery")
        return 1
    
    # Determine archive to use
    archive_to_use = None
    if args.archive:
        # Find specific archive
        for archive in safe_archives:
            if args.archive in str(archive["path"]):
                archive_to_use = archive
                break
        if not archive_to_use:
            print(f"❌ Archive not found: {args.archive}")
            return 1
    else:
        # Use latest archive
        archive_to_use = safe_archives[0]
    
    print(f"📦 Using archive: {archive_to_use['path'].name}")
    
    # Perform recovery
    if args.recover_all:
        success = recovery.recover_from_safe_archive(
            archive_to_use["path"], 
            dry_run=args.dry_run
        )
    elif args.recover_category:
        success = recovery.recover_from_safe_archive(
            archive_to_use["path"], 
            category=args.recover_category,
            dry_run=args.dry_run
        )
    elif args.recover_file:
        success = recovery.recover_from_safe_archive(
            archive_to_use["path"],
            specific_file=args.recover_file,
            dry_run=args.dry_run
        )
    else:
        # Interactive recovery
        recovery.list_recovery_sources()
        
        print(f"\nRecovery options for {archive_to_use['path'].name}:")
        categories = archive_to_use["categories"]
        for i, category in enumerate(categories, 1):
            print(f"   {i}. {category}")
        
        selection = input("Enter category number, 'all', or 'cancel': ").strip().lower()
        
        if selection == 'cancel':
            return 0
        elif selection == 'all':
            success = recovery.recover_from_safe_archive(archive_to_use["path"], dry_run=args.dry_run)
        else:
            try:
                index = int(selection) - 1
                category = categories[index]
                success = recovery.recover_from_safe_archive(
                    archive_to_use["path"], 
                    category=category, 
                    dry_run=args.dry_run
                )
            except (ValueError, IndexError):
                print("Invalid selection")
                return 1
    
    # Post-recovery validation
    if success and not args.dry_run:
        recovery.validate_recovery()
        recovery.save_recovery_log()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())