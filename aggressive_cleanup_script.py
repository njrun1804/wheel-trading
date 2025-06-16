#!/usr/bin/env python3
"""
Aggressive Cleanup Script for BOB Migration
==========================================

Permanently deletes legacy files after multiple confirmations.
Only use after safe_cleanup_script.py validation is complete.

⚠️  WARNING: This script PERMANENTLY DELETES files!
Only run after successful validation of safe cleanup.

Usage:
    python aggressive_cleanup_script.py --dry-run           # Preview deletions
    python aggressive_cleanup_script.py --category backups  # Delete specific category
    python aggressive_cleanup_script.py --force-all         # Delete everything (dangerous)
"""

import os
import shutil
import json
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Base directory
BASE_DIR = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
DELETION_LOG = BASE_DIR / f"deletion_log_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"

# High-risk large backups that can be safely deleted after validation 
AGGRESSIVE_CATEGORIES = {
    "large_backups": {
        "description": "Large backup files (>100MB) from completed migrations",
        "paths": [
            "phase1_cleanup_backup_20250616_025345.tar.gz",  # 1.5GB
            "bob_migration_backup_20250616_150750.tar.gz",   # 99MB
            "consolidated_backup_archive_20250616_025101.tar.gz",
        ],
        "space_estimate": "1.6GB",
        "safety_level": "LOW",  # Can be deleted after validation
        "requires_confirmation": True
    },
    "migration_artifacts": {
        "description": "Migration-related files no longer needed",
        "paths": [
            "bolt_migration_backup_20250616_155451/",
            ".bob_migration_backup_20250616_154502/",
            "validate_bob_migration.py",
            "validate_bolt_migration.py",
            "execute_bolt_migration.py",
            "migration.log",
            "migration_report.md",
            "bolt_to_bob_migration_plan.py",
            "einstein_to_bob_migration.py"
        ],
        "space_estimate": "10MB",
        "safety_level": "MEDIUM",
        "requires_confirmation": True
    },
    "old_logs": {
        "description": "Historical log files and temporary data",
        "paths": [
            "logs/*.log",
            "*.log",
            "data_audit/data_audit_2025-06-13.jsonl",
            "data_audit/data_audit_2025-06-14.jsonl", 
            "logs/memory_analysis_*.jsonl",
            "test_results/",
            "profile_output.txt"
        ],
        "space_estimate": "50MB",
        "safety_level": "MEDIUM",
        "requires_confirmation": True
    },
    "development_artifacts": {
        "description": "Development and testing artifacts",
        "paths": [
            "build/",
            "bolt_solver.egg-info/",
            "venv/",
            "*.egg-info/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".coverage*"
        ],
        "space_estimate": "100MB",
        "safety_level": "HIGH",  # Regenerable
        "requires_confirmation": False
    },
    "deprecated_executables": {
        "description": "Deprecated executable files and symlinks",
        "paths": [
            "bolt_cli",
            "boltcli",
            "bob_unified",
            "unified",
            "bolt_executable",
            "bob_cli_new",
            "bob_simple",
            "bob_unified_main"
        ],
        "space_estimate": "1MB",
        "safety_level": "MEDIUM",
        "requires_confirmation": False
    }
}

class AggressiveCleanup:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.deletion_log = {
            "cleanup_date": datetime.now().isoformat(),
            "base_directory": str(base_dir),
            "deleted_categories": {},
            "deleted_files": [],
            "errors": [],
            "safety_checks": []
        }
        
    def validate_safe_cleanup_completed(self) -> bool:
        """Verify that safe cleanup was run and validated first"""
        print("🔍 Validating safe cleanup prerequisite...")
        
        # Look for safe cleanup archives
        archive_dirs = list(self.base_dir.glob("archive/cleanup_*"))
        if not archive_dirs:
            print("❌ ERROR: No safe cleanup archive found")
            print("   Please run safe_cleanup_script.py first for safety")
            return False
        
        # Check for recent cleanup manifest
        latest_archive = max(archive_dirs, key=lambda p: p.stat().st_mtime)
        manifest_file = latest_archive / "cleanup_manifest.json"
        
        if not manifest_file.exists():
            print("❌ ERROR: Safe cleanup manifest not found")
            return False
        
        # Load and validate manifest
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            cleanup_date = datetime.fromisoformat(manifest["cleanup_date"])
            hours_since = (datetime.now() - cleanup_date).total_seconds() / 3600
            
            if hours_since > 72:  # 3 days
                print(f"⚠️  WARNING: Safe cleanup is {hours_since:.1f} hours old")
                print("   Consider running fresh validation")
            
            print(f"✅ Safe cleanup validated (archive: {latest_archive.name})")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Could not validate safe cleanup: {e}")
            return False
    
    def run_safety_checks(self) -> bool:
        """Run comprehensive safety checks before aggressive deletion"""
        print("🛡️  Running safety checks...")
        
        checks = []
        
        # 1. BOB system functionality
        bob_main = self.base_dir / "bob_main"
        if not bob_main.exists():
            checks.append(("BOB main CLI missing", False))
        else:
            checks.append(("BOB main CLI present", True))
        
        # 2. Critical BOB directories
        critical_dirs = [
            "bob/search/",
            "bob/agents/", 
            "bob/config/",
            "bob/integration/"
        ]
        
        for dir_path in critical_dirs:
            full_path = self.base_dir / dir_path
            if full_path.exists():
                checks.append((f"Critical directory {dir_path} present", True))
            else:
                checks.append((f"Critical directory {dir_path} MISSING", False))
        
        # 3. System performance test
        # This would ideally run bob_main --health-check
        # For now, just check if it's executable
        if bob_main.exists() and os.access(bob_main, os.X_OK):
            checks.append(("BOB main executable", True))
        else:
            checks.append(("BOB main not executable", False))
        
        # Record checks
        self.deletion_log["safety_checks"] = [
            {"check": desc, "passed": passed, "timestamp": datetime.now().isoformat()}
            for desc, passed in checks
        ]
        
        # Print results
        passed_checks = sum(1 for _, passed in checks if passed)
        total_checks = len(checks)
        
        print(f"Safety checks: {passed_checks}/{total_checks} passed")
        for desc, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {desc}")
        
        if passed_checks < total_checks:
            print("❌ Safety checks failed - aborting aggressive cleanup")
            return False
        
        print("✅ All safety checks passed")
        return True
    
    def calculate_deletion_impact(self, categories: List[str]) -> Dict[str, Any]:
        """Calculate what will be permanently deleted"""
        total_size = 0
        file_count = 0
        category_details = {}
        
        for category in categories:
            if category not in AGGRESSIVE_CATEGORIES:
                continue
                
            category_size = 0
            category_files = []
            
            for path_pattern in AGGRESSIVE_CATEGORIES[category]["paths"]:
                # Handle glob patterns
                if '*' in path_pattern:
                    for path in self.base_dir.glob(path_pattern):
                        if path.exists():
                            if path.is_file():
                                size = path.stat().st_size
                                category_size += size
                                category_files.append(str(path.relative_to(self.base_dir)))
                            elif path.is_dir():
                                for file_path in path.rglob("*"):
                                    if file_path.is_file():
                                        size = file_path.stat().st_size
                                        category_size += size
                                category_files.append(str(path.relative_to(self.base_dir)))
                else:
                    path = self.base_dir / path_pattern
                    if path.exists():
                        if path.is_file():
                            size = path.stat().st_size
                            category_size += size
                            category_files.append(path_pattern)
                        elif path.is_dir():
                            for file_path in path.rglob("*"):
                                if file_path.is_file():
                                    size = file_path.stat().st_size
                                    category_size += size
                            category_files.append(path_pattern)
            
            category_details[category] = {
                "files": category_files,
                "file_count": len(category_files),
                "size_bytes": category_size,
                "size_human": self._format_bytes(category_size),
                "safety_level": AGGRESSIVE_CATEGORIES[category]["safety_level"],
                "requires_confirmation": AGGRESSIVE_CATEGORIES[category]["requires_confirmation"]
            }
            
            total_size += category_size
            file_count += len(category_files)
        
        return {
            "total_size_bytes": total_size,
            "total_size_human": self._format_bytes(total_size),
            "total_files": file_count,
            "categories": category_details
        }
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
    
    def delete_category(self, category: str, dry_run: bool = True) -> bool:
        """Permanently delete files in category"""
        if category not in AGGRESSIVE_CATEGORIES:
            print(f"❌ Unknown category: {category}")
            return False
        
        cat_info = AGGRESSIVE_CATEGORIES[category]
        print(f"\n🗑️  Processing category: {category}")
        print(f"   Description: {cat_info['description']}")
        print(f"   Safety level: {cat_info['safety_level']}")
        
        deleted_files = []
        
        for path_pattern in cat_info["paths"]:
            # Handle glob patterns
            if '*' in path_pattern:
                paths_to_delete = list(self.base_dir.glob(path_pattern))
            else:
                path = self.base_dir / path_pattern
                paths_to_delete = [path] if path.exists() else []
            
            for path in paths_to_delete:
                if not path.exists():
                    continue
                
                if dry_run:
                    if path.is_dir():
                        file_count = len(list(path.rglob("*")))
                        print(f"   🗂️  Would delete directory: {path.relative_to(self.base_dir)} ({file_count} files)")
                    else:
                        size = self._format_bytes(path.stat().st_size)
                        print(f"   📄 Would delete file: {path.relative_to(self.base_dir)} ({size})")
                else:
                    try:
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            shutil.rmtree(path)
                        
                        deleted_files.append({
                            "path": str(path.relative_to(self.base_dir)),
                            "type": "directory" if path.is_dir() else "file",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        print(f"   ✅ Deleted: {path.relative_to(self.base_dir)}")
                        
                    except Exception as e:
                        error_msg = f"Failed to delete {path.relative_to(self.base_dir)}: {str(e)}"
                        print(f"   ❌ {error_msg}")  
                        self.deletion_log["errors"].append(error_msg)
        
        if not dry_run:
            self.deletion_log["deleted_categories"][category] = {
                "description": cat_info["description"],
                "deleted_files": deleted_files,
                "file_count": len(deleted_files)
            }
        
        return True
    
    def save_deletion_log(self):
        """Save deletion log for record keeping"""
        with open(DELETION_LOG, 'w') as f:
            json.dump(self.deletion_log, f, indent=2)
        print(f"📄 Deletion log saved: {DELETION_LOG}")

def main():
    parser = argparse.ArgumentParser(description="Aggressive cleanup script - PERMANENTLY DELETES FILES")
    parser.add_argument("--dry-run", action="store_true", help="Preview deletions without executing")
    parser.add_argument("--category", action="append", help="Specific category to delete")
    parser.add_argument("--force-all", action="store_true", help="Delete all categories (DANGEROUS)")
    parser.add_argument("--skip-safety", action="store_true", help="Skip safety checks (NOT RECOMMENDED)")
    parser.add_argument("--i-understand-this-deletes-files", action="store_true", 
                       help="Required confirmation flag")
    
    args = parser.parse_args()
    
    # Safety check - require explicit confirmation
    if not args.i_understand_this_deletes_files and not args.dry_run:
        print("❌ This script PERMANENTLY DELETES files!")
        print("   Add --i-understand-this-deletes-files flag to proceed")
        print("   Or use --dry-run to preview actions")
        return 1
    
    cleanup = AggressiveCleanup(BASE_DIR)
    
    # Validate prerequisites
    if not args.skip_safety:
        if not cleanup.validate_safe_cleanup_completed():
            return 1
        
        if not cleanup.run_safety_checks():
            return 1
    
    # Determine categories  
    categories = []
    if args.force_all:
        categories = list(AGGRESSIVE_CATEGORIES.keys())
        print("⚠️  WARNING: --force-all specified - will process ALL categories")
    elif args.category:
        categories = args.category
    else:
        # Interactive selection
        print("Available aggressive cleanup categories:")
        for i, (cat, info) in enumerate(AGGRESSIVE_CATEGORIES.items(), 1):
            confirmation = "⚠️ REQUIRES CONFIRMATION" if info['requires_confirmation'] else "Auto-approve"
            print(f"{i}. {cat}: {info['description']}")
            print(f"   Safety: {info['safety_level']}, {confirmation}")
        
        selection = input("\nEnter category numbers (comma-separated) or 'all': ").strip()
        if selection.lower() == 'all':
            categories = list(AGGRESSIVE_CATEGORIES.keys())
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                categories = [list(AGGRESSIVE_CATEGORIES.keys())[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid selection")
                return 1
    
    # Calculate impact
    impact = cleanup.calculate_deletion_impact(categories)
    
    print(f"\n💥 DELETION IMPACT ANALYSIS:")
    print(f"   Categories: {len(categories)}")
    print(f"   Total files to DELETE: {impact['total_files']}")
    print(f"   Space to recover: {impact['total_size_human']}")
    
    # Show per-category breakdown
    for category, details in impact['categories'].items():
        safety_icon = "🔒" if details['safety_level'] == 'HIGH' else "⚠️" if details['safety_level'] == 'MEDIUM' else "✅"
        print(f"   {safety_icon} {category}: {details['file_count']} files, {details['size_human']}")
        
        if details['requires_confirmation']:
            print(f"      ⚠️  Requires individual confirmation")
    
    # Final confirmation for execution
    if not args.dry_run:
        print(f"\n⚠️  ⚠️  ⚠️  FINAL WARNING ⚠️  ⚠️  ⚠️")
        print(f"This will PERMANENTLY DELETE {impact['total_files']} files")
        print(f"Space recovered: {impact['total_size_human']}")
        print("This action CANNOT be undone!")
        
        confirm = input("\nType 'DELETE' to confirm permanent deletion: ").strip()
        if confirm != 'DELETE':
            print("Deletion cancelled")
            return 0
        
        # Individual confirmations for high-risk categories
        for category in categories:
            if AGGRESSIVE_CATEGORIES[category]['requires_confirmation']:
                cat_details = impact['categories'][category]
                print(f"\n⚠️  Category '{category}' requires individual confirmation")
                print(f"   Files: {cat_details['file_count']}")
                print(f"   Size: {cat_details['size_human']}")
                
                confirm_cat = input(f"Delete category '{category}'? (yes/no): ").strip().lower()
                if confirm_cat != 'yes':
                    print(f"Skipping category: {category}")
                    categories.remove(category)
    
    # Execute deletion
    print(f"\n🗑️  {'[DRY RUN] ' if args.dry_run else ''}Starting aggressive cleanup...")
    
    for category in categories:
        cleanup.delete_category(category, dry_run=args.dry_run)
    
    if not args.dry_run:
        cleanup.save_deletion_log()
        
        print(f"\n✅ Aggressive cleanup completed!")
        print(f"   Files permanently deleted: {impact['total_files']}")
        print(f"   Space recovered: {impact['total_size_human']}")
        print(f"   Deletion log: {DELETION_LOG}")
        print(f"\n⚠️  Remember: These files are PERMANENTLY DELETED!")
    else:
        print(f"\n📋 Dry run completed - no files were deleted")
    
    return 0

if __name__ == "__main__":
    exit(main())