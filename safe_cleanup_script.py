#!/usr/bin/env python3
"""
Safe Cleanup Script for BOB Migration
=====================================

Moves legacy files to archive directory instead of deleting them.
Provides full recovery capability and detailed manifest logging.

Usage:
    python safe_cleanup_script.py --dry-run    # Preview what will be moved
    python safe_cleanup_script.py --execute    # Actually move files
    python safe_cleanup_script.py --category migration_backups  # Specific category
"""

import os
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Base directory
BASE_DIR = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
ARCHIVE_DIR = BASE_DIR / "archive" / f"cleanup_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
MANIFEST_FILE = ARCHIVE_DIR / "cleanup_manifest.json"

# Categories for cleanup
CLEANUP_CATEGORIES = {
    "old_einstein": {
        "description": "Old Einstein search directory (migrated to /bob/search/)",
        "paths": ["einstein/"],
        "space_estimate": "1.4MB",
        "safety_level": "HIGH"
    },
    "old_bolt": {
        "description": "Old BOLT agent directory (migrated to /bob/agents/)",
        "paths": ["bolt/"],
        "space_estimate": "2.8MB", 
        "safety_level": "HIGH"
    },
    "legacy_cli": {
        "description": "Legacy CLI files (replaced by bob_main)",
        "paths": [
            "bob_cli.py", "bolt_cli.py", "unified_cli.py", "bob_unified_cli.py",
            "bolt_einstein_cli.py", "unified_cli_demo.py", "bob_cli_examples.py",
            "bolt_cli", "boltcli", "bob_unified", "unified"
        ],
        "space_estimate": "2MB",
        "safety_level": "MEDIUM"
    },
    "migration_backups": {
        "description": "Migration backup files and directories (completed migrations)",
        "paths": [
            "bob_migration_backup_20250616_150750.tar.gz",
            "bolt_migration_backup_20250616_155451/",
            ".bob_migration_backup_20250616_154502/",
            "consolidated_backup_archive_20250616_025101.tar.gz",
            "validate_bob_migration.py",
            "validate_bolt_migration.py", 
            "execute_bolt_migration.py",
            "migration.log",
            "einstein_to_bob_migration.py",
            "bolt_to_bob_migration_plan.py"
        ],
        "space_estimate": "1.6GB",
        "safety_level": "LOW"  # Can be safely removed after validation
    },
    "legacy_docs": {
        "description": "Superseded documentation files",
        "paths": [
            # Will be populated by finding *_MIGRATION_*.md, EINSTEIN_*.md, BOLT_*.md files
        ],
        "space_estimate": "50MB",
        "safety_level": "MEDIUM"
    },
    "config_duplicates": {
        "description": "Duplicate configuration files (consolidated into /bob/config/)",
        "paths": [
            "config_backup_agent4/",
            "meta_backups/",
            "config_unified.yaml"  # If superseded by bob/config/unified_config.yaml
        ],
        "space_estimate": "1MB",
        "safety_level": "MEDIUM"
    }
}

class SafeCleanup:
    def __init__(self, base_dir: Path, archive_dir: Path):
        self.base_dir = base_dir
        self.archive_dir = archive_dir
        self.manifest = {
            "cleanup_date": datetime.now().isoformat(),
            "base_directory": str(base_dir),
            "archive_directory": str(archive_dir),
            "categories": {},
            "moved_files": [],
            "errors": [],
            "recovery_info": {}
        }
        
    def find_legacy_docs(self) -> List[str]:
        """Find legacy documentation files to clean up"""
        patterns = [
            "*_MIGRATION_*.md",
            "EINSTEIN_*.md", 
            "BOLT_*.md",
            "BOB_MIGRATION_*.md",
            "EINSTEIN_TO_BOB_*.md",
            "BOLT_TO_BOB_*.md"
        ]
        
        legacy_docs = []
        for pattern in patterns:
            legacy_docs.extend([
                str(p.relative_to(self.base_dir)) 
                for p in self.base_dir.glob(pattern)
                if p.is_file()
            ])
        
        return legacy_docs
    
    def validate_prerequisites(self) -> bool:
        """Validate system is ready for cleanup"""
        print("🔍 Validating prerequisites...")
        
        # Check BOB system exists
        bob_dir = self.base_dir / "bob"
        if not bob_dir.exists():
            print("❌ ERROR: /bob/ directory not found - migration may not be complete")
            return False
            
        # Check BOB main exists
        bob_main = self.base_dir / "bob_main"
        if not bob_main.exists():
            print("❌ ERROR: bob_main CLI not found - system may not be ready")
            return False
            
        # Check critical BOB components
        critical_components = [
            "bob/search/engine.py",
            "bob/agents/orchestrator.py", 
            "bob/config/unified_config.yaml"
        ]
        
        for component in critical_components:
            if not (self.base_dir / component).exists():
                print(f"❌ ERROR: Critical component missing: {component}")
                return False
                
        print("✅ Prerequisites validation passed")
        return True
    
    def estimate_space_recovery(self, categories: List[str]) -> Dict[str, Any]:
        """Calculate actual space that will be recovered"""
        total_size = 0
        file_count = 0
        category_details = {}
        
        for category in categories:
            if category not in CLEANUP_CATEGORIES:
                continue
                
            category_size = 0
            category_files = 0
            
            # Update legacy docs paths
            if category == "legacy_docs":
                CLEANUP_CATEGORIES[category]["paths"] = self.find_legacy_docs()
            
            for path_str in CLEANUP_CATEGORIES[category]["paths"]:
                path = self.base_dir / path_str
                if path.exists():
                    if path.is_file():
                        size = path.stat().st_size
                        category_size += size
                        category_files += 1
                    elif path.is_dir():
                        for file_path in path.rglob("*"):
                            if file_path.is_file():
                                size = file_path.stat().st_size
                                category_size += size
                                category_files += 1
            
            category_details[category] = {
                "files": category_files,
                "size_bytes": category_size,
                "size_human": self._format_bytes(category_size)
            }
            
            total_size += category_size
            file_count += category_files
        
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
    
    def move_to_archive(self, categories: List[str], dry_run: bool = True) -> bool:
        """Move files to archive directory"""
        if not dry_run:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Update legacy docs
        if "legacy_docs" in categories:
            CLEANUP_CATEGORIES["legacy_docs"]["paths"] = self.find_legacy_docs()
        
        for category in categories:
            if category not in CLEANUP_CATEGORIES:
                print(f"⚠️  Unknown category: {category}")
                continue
                
            print(f"\n📂 Processing category: {category}")
            print(f"   Description: {CLEANUP_CATEGORIES[category]['description']}")
            
            category_moved = []
            
            for path_str in CLEANUP_CATEGORIES[category]["paths"]:
                source_path = self.base_dir / path_str
                
                if not source_path.exists():
                    print(f"   ⚠️  Path not found: {path_str}")
                    continue
                
                # Create destination path maintaining directory structure
                dest_path = self.archive_dir / category / path_str
                
                if dry_run:
                    print(f"   📋 Would move: {path_str} -> archive/{category}/{path_str}")
                    if source_path.is_dir():
                        file_count = len(list(source_path.rglob("*")))
                        print(f"      📁 Directory with {file_count} files")
                else:
                    try:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if source_path.is_file():
                            shutil.move(str(source_path), str(dest_path))
                        else:
                            shutil.move(str(source_path), str(dest_path))
                        
                        category_moved.append({
                            "source": str(source_path),
                            "destination": str(dest_path),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        print(f"   ✅ Moved: {path_str}")
                        
                    except Exception as e:
                        error_msg = f"Failed to move {path_str}: {str(e)}"
                        print(f"   ❌ {error_msg}")
                        self.manifest["errors"].append(error_msg)
            
            if not dry_run:
                self.manifest["categories"][category] = {
                    "description": CLEANUP_CATEGORIES[category]["description"],
                    "moved_files": category_moved,
                    "file_count": len(category_moved)
                }
        
        return True
    
    def save_manifest(self):
        """Save cleanup manifest for recovery"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"📄 Cleanup manifest saved: {self.manifest_file}")
    
    def create_recovery_script(self):
        """Create script to recover moved files"""
        recovery_script = self.archive_dir / "recover_files.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Recovery Script for Cleanup Archive
==================================

Auto-generated recovery script to restore files moved during cleanup.

Usage:
    python recover_files.py --all                    # Restore all files
    python recover_files.py --category old_einstein  # Restore specific category
    python recover_files.py --file path/to/file.py   # Restore specific file
"""

import shutil
import json
from pathlib import Path

MANIFEST_FILE = Path(__file__).parent / "cleanup_manifest.json"
BASE_DIR = Path("{self.base_dir}")

def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)

def recover_category(category_name: str):
    manifest = load_manifest()
    if category_name not in manifest["categories"]:
        print(f"Category not found: {{category_name}}")
        return
    
    category = manifest["categories"][category_name]
    for file_info in category["moved_files"]:
        source = Path(file_info["destination"])
        dest = Path(file_info["source"])
        
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            print(f"Recovered: {{dest}}")
        else:
            print(f"Archive file not found: {{source}}")

def recover_all():
    manifest = load_manifest()
    for category_name in manifest["categories"]:
        print(f"Recovering category: {{category_name}}")
        recover_category(category_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python recover_files.py --all | --category <name>")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        recover_all()
    elif sys.argv[1] == "--category" and len(sys.argv) > 2:
        recover_category(sys.argv[2])
    else:
        print("Invalid arguments")
'''
        
        with open(recovery_script, 'w') as f:
            f.write(script_content)
        
        recovery_script.chmod(0o755)
        print(f"🔧 Recovery script created: {recovery_script}")

def main():
    parser = argparse.ArgumentParser(description="Safe cleanup script for BOB migration")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("--execute", action="store_true", help="Actually execute the cleanup")
    parser.add_argument("--category", action="append", help="Specific category to clean (can use multiple times)")
    parser.add_argument("--all-categories", action="store_true", help="Clean all categories")
    parser.add_argument("--skip-validation", action="store_true", help="Skip prerequisite validation")
    
    args = parser.parse_args()
    
    if not (args.dry_run or args.execute):
        print("Must specify either --dry-run or --execute")
        return 1
    
    # Determine categories to process
    categories = []
    if args.all_categories:
        categories = list(CLEANUP_CATEGORIES.keys())
    elif args.category:
        categories = args.category
    else:
        # Default to safe categories for interactive selection
        print("Available cleanup categories:")
        for i, (cat, info) in enumerate(CLEANUP_CATEGORIES.items(), 1):
            print(f"{i}. {cat}: {info['description']} (Safety: {info['safety_level']})")
        
        selection = input("Enter category numbers (comma-separated) or 'all': ").strip()
        if selection.lower() == 'all':
            categories = list(CLEANUP_CATEGORIES.keys())
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                categories = [list(CLEANUP_CATEGORIES.keys())[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid selection")
                return 1
    
    cleanup = SafeCleanup(BASE_DIR, ARCHIVE_DIR)
    
    # Validation
    if not args.skip_validation:
        if not cleanup.validate_prerequisites():
            print("❌ Prerequisites validation failed - aborting cleanup")
            return 1
    
    # Space estimation
    space_info = cleanup.estimate_space_recovery(categories)
    print(f"\n📊 Cleanup Summary:")
    print(f"   Categories: {len(categories)}")
    print(f"   Total files: {space_info['total_files']}")
    print(f"   Space to recover: {space_info['total_size_human']}")
    
    for category, details in space_info['categories'].items():
        safety = CLEANUP_CATEGORIES[category]['safety_level']
        print(f"   • {category}: {details['files']} files, {details['size_human']} (Safety: {safety})")
    
    # Confirmation for execution
    if args.execute and not args.dry_run:
        print(f"\n⚠️  This will move {space_info['total_files']} files to:")
        print(f"   {ARCHIVE_DIR}")
        
        confirm = input("Continue with cleanup? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cleanup cancelled")
            return 0
    
    # Execute cleanup
    print(f"\n🧹 {'[DRY RUN] ' if args.dry_run else ''}Starting cleanup...")
    
    success = cleanup.move_to_archive(categories, dry_run=args.dry_run)
    
    if success and not args.dry_run:
        cleanup.save_manifest()
        cleanup.create_recovery_script()
        
        print(f"\n✅ Cleanup completed successfully!")
        print(f"   Files moved to: {ARCHIVE_DIR}")
        print(f"   Recovery available via: {ARCHIVE_DIR / 'recover_files.py'}")
        print(f"   Space recovered: {space_info['total_size_human']}")
    elif args.dry_run:
        print(f"\n📋 Dry run completed - no files were moved")
    
    return 0

if __name__ == "__main__":
    exit(main())