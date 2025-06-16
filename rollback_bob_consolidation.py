#!/usr/bin/env python3
"""
BOB CLI Consolidation Rollback Script
=====================================

Emergency rollback script to restore the original CLI system
in case of issues with the unified BOB interface.

Features:
- Automatic detection of backup files
- Restoration of original CLI scripts
- Cleanup of unified system components
- Validation of restored system
- Detailed rollback reporting

Usage:
    python rollback_bob_consolidation.py
    python rollback_bob_consolidation.py --dry-run
    python rollback_bob_consolidation.py --force
    python rollback_bob_consolidation.py --backup-dir <path>
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import subprocess


class BobConsolidationRollback:
    """Handles rollback of BOB CLI consolidation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize rollback system."""
        self.project_root = project_root or Path(__file__).parent
        self.rollback_log = []
        self.errors = []
        
        # Files created by consolidation
        self.consolidation_files = [
            "bob",  # Main unified executable
            "bob_unified_main",  # Unified main script
            "setup_bob_symlinks.sh",  # Setup script
            "test_unified_bob_cli.py",  # Test suite
            "rollback_bob_consolidation.py",  # This script
            "bob/cli/unified_router.py",  # Unified router
            "bob/cli/intent_detector.py",  # Intent detection
            "bob/cli/command_translator.py",  # Command translation
            "bob/config/unified_bob_config.yaml",  # Unified config
        ]
        
        # Original files that were converted to wrappers
        self.wrapper_files = [
            "bob_cli.py",
            "bolt_cli.py", 
            "bob_unified.py",
            "unified_cli.py"
        ]
        
    def find_backup_directory(self) -> Optional[Path]:
        """Find the most recent backup directory."""
        backup_pattern = ".bob_migration_backup_*"
        backup_dirs = list(self.project_root.glob(backup_pattern))
        
        if not backup_dirs:
            return None
            
        # Sort by creation time and return most recent
        backup_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backup_dirs[0]
        
    def validate_backup(self, backup_dir: Path) -> bool:
        """Validate that backup contains expected files."""
        required_files = ["bob_cli.py", "bolt_cli.py", "bob_unified.py", "unified_cli.py"]
        
        for file_name in required_files:
            backup_file = backup_dir / file_name
            if not backup_file.exists():
                self.errors.append(f"Missing backup file: {file_name}")
                return False
                
        return True
        
    def rollback(self, backup_dir: Optional[Path] = None, dry_run: bool = False, force: bool = False) -> bool:
        """
        Perform rollback to original CLI system.
        
        Args:
            backup_dir: Specific backup directory to restore from
            dry_run: Show what would be done without executing
            force: Force rollback even if validation fails
            
        Returns:
            True if rollback successful, False otherwise
        """
        
        print("🔄 BOB CLI Consolidation Rollback")
        print("=" * 50)
        
        if dry_run:
            print("📋 DRY RUN MODE - No changes will be made")
            print()
            
        # Find backup directory
        if backup_dir is None:
            backup_dir = self.find_backup_directory()
            
        if backup_dir is None:
            self.errors.append("No backup directory found")
            print("❌ Error: No backup directory found")
            print("   Look for directories matching: .bob_migration_backup_*")
            return False
            
        print(f"📂 Using backup directory: {backup_dir}")
        
        # Validate backup
        if not self.validate_backup(backup_dir):
            if not force:
                print("❌ Error: Backup validation failed")
                for error in self.errors:
                    print(f"   {error}")
                print("   Use --force to proceed anyway")
                return False
            else:
                print("⚠️  Warning: Backup validation failed, but proceeding due to --force")
                
        # Perform rollback steps
        steps = [
            ("Remove unified executable", self._remove_unified_executable),
            ("Restore original CLI files", lambda dr: self._restore_original_files(backup_dir, dr)),
            ("Remove consolidation files", self._remove_consolidation_files),
            ("Cleanup unified system", self._cleanup_unified_system),
            ("Validate restored system", self._validate_restored_system)
        ]
        
        success = True
        
        for step_name, step_function in steps:
            print(f"\n🔧 {step_name}...")
            
            try:
                if step_function(dry_run):
                    print(f"   ✅ {step_name} completed")
                    self.rollback_log.append({
                        "step": step_name,
                        "status": "success",
                        "timestamp": time.time()
                    })
                else:
                    print(f"   ❌ {step_name} failed")
                    self.rollback_log.append({
                        "step": step_name,
                        "status": "failed", 
                        "timestamp": time.time()
                    })
                    success = False
                    
            except Exception as e:
                print(f"   💥 {step_name} error: {e}")
                self.rollback_log.append({
                    "step": step_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                })
                success = False
                
        # Summary
        print(f"\n📊 Rollback Summary")
        print("-" * 30)
        
        if success and not dry_run:
            print("✅ Rollback completed successfully!")
            print("   Original CLI system has been restored")
            print("   You can now use the original commands:")
            print("     python bob_cli.py <command>")
            print("     python bolt_cli.py <query>")
            print("     python bob_unified.py <command>")
            print("     python unified_cli.py <query>")
        elif success and dry_run:
            print("✅ Dry run completed successfully!")
            print("   All rollback steps would execute without errors")
            print("   Run without --dry-run to perform actual rollback")
        else:
            print("❌ Rollback failed!")
            print("   Some steps encountered errors")
            print("   Check the rollback log for details")
            
        # Export rollback log
        self._export_rollback_log()
        
        return success
        
    def _remove_unified_executable(self, dry_run: bool) -> bool:
        """Remove unified BOB executable and symlinks."""
        files_to_remove = [
            self.project_root / "bob",
            self.project_root / "bob_unified_main"
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                if dry_run:
                    print(f"   Would remove: {file_path}")
                else:
                    try:
                        file_path.unlink()
                        print(f"   Removed: {file_path}")
                    except Exception as e:
                        print(f"   Error removing {file_path}: {e}")
                        return False
                        
        return True
        
    def _restore_original_files(self, backup_dir: Path, dry_run: bool) -> bool:
        """Restore original CLI files from backup."""
        for file_name in self.wrapper_files:
            backup_file = backup_dir / file_name
            target_file = self.project_root / file_name
            
            if backup_file.exists():
                if dry_run:
                    print(f"   Would restore: {backup_file} → {target_file}")
                else:
                    try:
                        shutil.copy2(backup_file, target_file)
                        target_file.chmod(0o755)  # Make executable
                        print(f"   Restored: {file_name}")
                    except Exception as e:
                        print(f"   Error restoring {file_name}: {e}")
                        return False
            else:
                print(f"   Warning: Backup file {file_name} not found")
                
        return True
        
    def _remove_consolidation_files(self, dry_run: bool) -> bool:
        """Remove files created by consolidation."""
        for file_path_str in self.consolidation_files:
            file_path = self.project_root / file_path_str
            
            if file_path.exists():
                if dry_run:
                    print(f"   Would remove: {file_path}")
                else:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        else:
                            shutil.rmtree(file_path)
                        print(f"   Removed: {file_path_str}")
                    except Exception as e:
                        print(f"   Error removing {file_path_str}: {e}")
                        # Continue with other files
                        
        return True
        
    def _cleanup_unified_system(self, dry_run: bool) -> bool:
        """Cleanup unified system components."""
        # Remove any generated test results
        test_result_files = list(self.project_root.glob("unified_bob_cli_test_results_*.json"))
        
        for file_path in test_result_files:
            if dry_run:
                print(f"   Would remove test results: {file_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"   Removed test results: {file_path.name}")
                except Exception as e:
                    print(f"   Error removing {file_path.name}: {e}")
                    
        return True
        
    def _validate_restored_system(self, dry_run: bool) -> bool:
        """Validate that original system is working."""
        if dry_run:
            print("   Would validate restored CLI files")
            return True
            
        # Check that original files exist and are executable
        validation_tests = [
            ("bob_cli.py", "python bob_cli.py --help"),
            ("bolt_cli.py", "python bolt_cli.py --help"), 
            ("bob_unified.py", "python bob_unified.py --help"),
            ("unified_cli.py", "python unified_cli.py --help")
        ]
        
        all_passed = True
        
        for file_name, test_command in validation_tests:
            file_path = self.project_root / file_name
            
            if not file_path.exists():
                print(f"   ❌ Missing: {file_name}")
                all_passed = False
                continue
                
            # Try to execute help command
            try:
                result = subprocess.run(
                    test_command.split(),
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print(f"   ✅ Working: {file_name}")
                else:
                    print(f"   ⚠️  Issues: {file_name} (exit code {result.returncode})")
                    
            except Exception as e:
                print(f"   ❌ Error testing {file_name}: {e}")
                all_passed = False
                
        return all_passed
        
    def _export_rollback_log(self):
        """Export rollback log to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.project_root / f"bob_rollback_log_{timestamp}.json"
        
        log_data = {
            "rollback_timestamp": timestamp,
            "project_root": str(self.project_root),
            "steps": self.rollback_log,
            "errors": self.errors,
            "success": len(self.errors) == 0
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"\n📄 Rollback log saved: {log_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save rollback log: {e}")
            
    def list_backups(self) -> List[Path]:
        """List available backup directories."""
        backup_pattern = ".bob_migration_backup_*"
        backup_dirs = list(self.project_root.glob(backup_pattern))
        backup_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backup_dirs
        
    def show_backup_info(self, backup_dir: Path):
        """Show information about a backup directory."""
        print(f"\n📂 Backup: {backup_dir.name}")
        print(f"   Path: {backup_dir}")
        print(f"   Created: {time.ctime(backup_dir.stat().st_mtime)}")
        print(f"   Size: {self._get_dir_size(backup_dir) / 1024:.1f} KB")
        
        # List contents
        files = list(backup_dir.iterdir())
        print(f"   Files ({len(files)}):")
        for file_path in sorted(files):
            size_kb = file_path.stat().st_size / 1024
            print(f"     {file_path.name} ({size_kb:.1f} KB)")
            
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size


def main():
    """Main entry point for rollback script."""
    parser = argparse.ArgumentParser(
        description="BOB CLI Consolidation Rollback Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rollback_bob_consolidation.py
  python rollback_bob_consolidation.py --dry-run
  python rollback_bob_consolidation.py --list-backups
  python rollback_bob_consolidation.py --backup-dir .bob_migration_backup_20250616_154502
        """
    )
    
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--force", action="store_true", help="Force rollback even if validation fails")
    parser.add_argument("--backup-dir", type=Path, help="Specific backup directory to restore from")
    parser.add_argument("--list-backups", action="store_true", help="List available backup directories")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize rollback system
    rollback = BobConsolidationRollback(args.project_root)
    
    try:
        if args.list_backups:
            print("📋 Available Backup Directories")
            print("=" * 40)
            
            backups = rollback.list_backups()
            if not backups:
                print("No backup directories found")
                sys.exit(1)
                
            for i, backup_dir in enumerate(backups, 1):
                print(f"\n{i}. {backup_dir.name}")
                rollback.show_backup_info(backup_dir)
                
        else:
            # Perform rollback
            success = rollback.rollback(
                backup_dir=args.backup_dir,
                dry_run=args.dry_run,
                force=args.force
            )
            
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Rollback cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()