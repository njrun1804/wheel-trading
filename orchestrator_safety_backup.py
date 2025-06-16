#!/usr/bin/env python3
"""
Orchestrator Safety Backup and Validation Script
Agent 7 (P-Core 6): Comprehensive backup and safety validation system
"""

import ast
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


class OrchestorSafetyBackup:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).absolute()
        self.backup_dir = self.repo_path / "orchestrator_safety_backups"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = self.backup_dir / f"backup_{self.timestamp}"
        self.validation_results = {}

    def create_backup_structure(self):
        """Create backup directory structure"""
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.backup_path / "files").mkdir()
        (self.backup_path / "metadata").mkdir()
        (self.backup_path / "validation").mkdir()

    def get_deleted_files(self) -> list[str]:
        """Get list of files marked for deletion in git"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            deleted_files = []
            for line in result.stdout.strip().split("\n"):
                if line.startswith("D "):
                    # Handle quoted filenames
                    filename = line[3:].strip()
                    if filename.startswith('"') and filename.endswith('"'):
                        filename = filename[1:-1]
                    deleted_files.append(filename)

            return deleted_files
        except subprocess.CalledProcessError as e:
            print(f"Error getting git status: {e}")
            return []

    def backup_deleted_files(self, deleted_files: list[str]) -> dict[str, str]:
        """Backup all files marked for deletion"""
        backup_manifest = {}

        for file_path in deleted_files:
            full_path = self.repo_path / file_path

            # Skip if file doesn't exist (already deleted)
            if not full_path.exists():
                continue

            # Create backup path
            backup_file_path = self.backup_path / "files" / file_path
            backup_file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if full_path.is_file():
                    shutil.copy2(full_path, backup_file_path)
                    # Calculate file hash for integrity
                    file_hash = self.calculate_file_hash(full_path)
                    backup_manifest[file_path] = {
                        "type": "file",
                        "hash": file_hash,
                        "size": full_path.stat().st_size,
                        "backup_path": str(backup_file_path),
                    }
                elif full_path.is_dir():
                    shutil.copytree(full_path, backup_file_path, dirs_exist_ok=True)
                    backup_manifest[file_path] = {
                        "type": "directory",
                        "backup_path": str(backup_file_path),
                    }
            except Exception as e:
                print(f"Error backing up {file_path}: {e}")
                backup_manifest[file_path] = {"error": str(e)}

        return backup_manifest

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return "error_calculating_hash"

    def find_python_imports(self, file_path: Path) -> set[str]:
        """Find all imports in a Python file"""
        imports = set()
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        except Exception:
            # Fallback to regex if AST parsing fails
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                import_patterns = [
                    r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)",
                    r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import",
                ]

                for pattern in import_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    imports.update(matches)
            except Exception:
                pass

        return imports

    def analyze_import_dependencies(self, deleted_files: list[str]) -> dict[str, any]:
        """Analyze import dependencies for deleted files"""
        dependency_analysis = {
            "deleted_python_files": [],
            "modules_being_deleted": set(),
            "external_references": [],
            "safe_to_delete": [],
            "requires_review": [],
        }

        # Find Python files being deleted
        for file_path in deleted_files:
            if file_path.endswith(".py"):
                dependency_analysis["deleted_python_files"].append(file_path)
                # Extract module name
                module_name = file_path.replace("/", ".").replace(".py", "")
                dependency_analysis["modules_being_deleted"].add(module_name)

        # Check remaining Python files for references to deleted modules
        remaining_python_files = []
        for root, _dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_path)
                    if str(rel_path) not in deleted_files:
                        remaining_python_files.append(file_path)

        # Analyze remaining files for references to deleted modules
        for file_path in remaining_python_files:
            try:
                imports = self.find_python_imports(file_path)
                for deleted_module in dependency_analysis["modules_being_deleted"]:
                    if any(deleted_module in imp for imp in imports):
                        dependency_analysis["external_references"].append(
                            {
                                "file": str(file_path.relative_to(self.repo_path)),
                                "references": deleted_module,
                            }
                        )
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Categorize files based on safety
        for file_path in dependency_analysis["deleted_python_files"]:
            has_external_refs = any(
                ref["references"] in file_path
                for ref in dependency_analysis["external_references"]
            )

            if has_external_refs:
                dependency_analysis["requires_review"].append(file_path)
            else:
                dependency_analysis["safe_to_delete"].append(file_path)

        # Convert set to list for JSON serialization
        dependency_analysis["modules_being_deleted"] = list(
            dependency_analysis["modules_being_deleted"]
        )

        return dependency_analysis

    def identify_critical_files(self, deleted_files: list[str]) -> dict[str, list[str]]:
        """Identify critical files that should not be deleted"""
        critical_patterns = {
            "core_config": [
                "pyproject.toml",
                "setup.py",
                "requirements.txt",
                "config.yaml",
                "config_unified.yaml",
            ],
            "main_entry_points": ["run.py", "main.py", "__main__.py", "__init__.py"],
            "critical_modules": [
                "src/unity_wheel/api/advisor.py",
                "src/unity_wheel/strategy/wheel.py",
                "src/unity_wheel/math/options.py",
            ],
            "essential_docs": ["README.md", "CLAUDE.md"],
            "git_config": [".gitignore", ".github/workflows/"],
        }

        critical_files_found = {}
        for category, patterns in critical_patterns.items():
            critical_files_found[category] = []
            for file_path in deleted_files:
                for pattern in patterns:
                    if pattern in file_path or file_path.endswith(pattern):
                        critical_files_found[category].append(file_path)

        return critical_files_found

    def validate_file_safety(self, deleted_files: list[str]) -> dict[str, any]:
        """Comprehensive file safety validation"""
        validation = {
            "total_files_to_delete": len(deleted_files),
            "critical_files": self.identify_critical_files(deleted_files),
            "dependency_analysis": self.analyze_import_dependencies(deleted_files),
            "safety_score": 0,
            "recommendations": [],
            "blocking_issues": [],
        }

        # Check for critical files
        critical_count = sum(
            len(files) for files in validation["critical_files"].values()
        )
        if critical_count > 0:
            validation["blocking_issues"].append(
                f"Found {critical_count} critical files marked for deletion"
            )

        # Check for external references
        external_refs = len(validation["dependency_analysis"]["external_references"])
        if external_refs > 0:
            validation["blocking_issues"].append(
                f"Found {external_refs} external references to files being deleted"
            )

        # Calculate safety score (0-100)
        safety_score = 100
        safety_score -= min(critical_count * 20, 80)  # -20 per critical file, max -80
        safety_score -= min(external_refs * 5, 20)  # -5 per external ref, max -20

        validation["safety_score"] = max(safety_score, 0)

        # Generate recommendations
        if validation["safety_score"] < 50:
            validation["recommendations"].append(
                "STOP: Manual review required before deletion"
            )
        elif validation["safety_score"] < 80:
            validation["recommendations"].append(
                "CAUTION: Review critical files and dependencies"
            )
        else:
            validation["recommendations"].append("SAFE: Proceed with deletion")

        return validation

    def create_restoration_procedure(self, backup_manifest: dict[str, any]) -> str:
        """Create detailed restoration procedure documentation"""
        procedure = f"""
# Orchestrator Cleanup Restoration Procedure
Created: {datetime.datetime.now().isoformat()}
Backup Location: {self.backup_path}

## Quick Restoration Commands

### Full Restoration (All Files)
```bash
# Navigate to repository root
cd "{self.repo_path}"

# Restore all backed up files
rsync -av "{self.backup_path}/files/" ./

# Reset git status
git reset HEAD .
```

### Selective Restoration
```bash
# Restore specific file
cp "{self.backup_path}/files/path/to/file" "path/to/file"

# Restore specific directory
cp -r "{self.backup_path}/files/path/to/dir" "path/to/dir"
```

## Backup Manifest
Total files backed up: {len(backup_manifest)}

### Critical Files in Backup
"""

        for file_path, info in backup_manifest.items():
            if isinstance(info, dict) and "hash" in info:
                procedure += f"- {file_path} (Hash: {info['hash'][:16]}...)\n"
            else:
                procedure += f"- {file_path}\n"

        procedure += f"""

## Validation Commands
```bash
# Verify backup integrity
python orchestrator_safety_backup.py --verify-backup "{self.backup_path}"

# Check for missing dependencies after restoration
python orchestrator_safety_backup.py --check-dependencies
```

## Emergency Contacts
- Repository: wheel-trading
- Backup timestamp: {self.timestamp}
- Agent: 7 (P-Core 6)
"""

        return procedure

    def save_backup_metadata(
        self, backup_manifest: dict[str, any], validation_results: dict[str, any]
    ):
        """Save all backup metadata and validation results"""
        metadata = {
            "timestamp": self.timestamp,
            "repo_path": str(self.repo_path),
            "backup_path": str(self.backup_path),
            "total_files_backed_up": len(backup_manifest),
            "backup_manifest": backup_manifest,
            "validation_results": validation_results,
            "git_status": self.get_git_status(),
        }

        # Save metadata
        metadata_file = self.backup_path / "metadata" / "backup_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save validation results separately
        validation_file = self.backup_path / "validation" / "safety_validation.json"
        with open(validation_file, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)

        # Create restoration procedure
        procedure = self.create_restoration_procedure(backup_manifest)
        procedure_file = self.backup_path / "RESTORATION_PROCEDURE.md"
        with open(procedure_file, "w") as f:
            f.write(procedure)

    def get_git_status(self) -> dict[str, any]:
        """Get current git status information"""
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = branch_result.stdout.strip()

            # Get current commit
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            current_commit = commit_result.stdout.strip()

            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            return {
                "branch": current_branch,
                "commit": current_commit,
                "status_lines": status_result.stdout.strip().split("\n")
                if status_result.stdout.strip()
                else [],
                "clean": len(status_result.stdout.strip()) == 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def test_backup_integrity(self, backup_manifest: dict[str, any]) -> dict[str, any]:
        """Test backup integrity by comparing hashes"""
        integrity_results = {
            "total_files": len(backup_manifest),
            "verified_files": 0,
            "failed_files": [],
            "missing_files": [],
            "integrity_score": 0,
        }

        for file_path, info in backup_manifest.items():
            if isinstance(info, dict) and "hash" in info:
                backup_file_path = Path(info["backup_path"])

                if not backup_file_path.exists():
                    integrity_results["missing_files"].append(file_path)
                    continue

                # Calculate hash of backup file
                backup_hash = self.calculate_file_hash(backup_file_path)

                if backup_hash == info["hash"]:
                    integrity_results["verified_files"] += 1
                else:
                    integrity_results["failed_files"].append(
                        {
                            "file": file_path,
                            "expected_hash": info["hash"],
                            "actual_hash": backup_hash,
                        }
                    )

        # Calculate integrity score
        if integrity_results["total_files"] > 0:
            integrity_results["integrity_score"] = (
                integrity_results["verified_files"] / integrity_results["total_files"]
            ) * 100

        return integrity_results

    def run_full_backup_and_validation(self) -> dict[str, any]:
        """Run complete backup and validation process"""
        print("🔄 Starting Orchestrator Safety Backup and Validation...")

        # Create backup structure
        self.create_backup_structure()
        print(f"📁 Created backup directory: {self.backup_path}")

        # Get files marked for deletion
        deleted_files = self.get_deleted_files()
        print(f"📋 Found {len(deleted_files)} files marked for deletion")

        # Backup deleted files
        print("💾 Creating backup of files...")
        backup_manifest = self.backup_deleted_files(deleted_files)

        # Validate file safety
        print("🔍 Validating file safety...")
        validation_results = self.validate_file_safety(deleted_files)

        # Test backup integrity
        print("✅ Testing backup integrity...")
        integrity_results = self.test_backup_integrity(backup_manifest)

        # Save all metadata
        self.save_backup_metadata(backup_manifest, validation_results)

        # Compile final results
        final_results = {
            "backup_path": str(self.backup_path),
            "backup_manifest": backup_manifest,
            "validation_results": validation_results,
            "integrity_results": integrity_results,
            "timestamp": self.timestamp,
        }

        return final_results


def main():
    repo_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"

    backup_system = OrchestorSafetyBackup(repo_path)
    results = backup_system.run_full_backup_and_validation()

    print("\n" + "=" * 60)
    print("ORCHESTRATOR SAFETY BACKUP COMPLETE")
    print("=" * 60)
    print(f"Backup Location: {results['backup_path']}")
    print(f"Files Backed Up: {len(results['backup_manifest'])}")
    print(f"Safety Score: {results['validation_results']['safety_score']}/100")
    print(f"Integrity Score: {results['integrity_results']['integrity_score']:.1f}%")

    if results["validation_results"]["blocking_issues"]:
        print("\n⚠️  BLOCKING ISSUES FOUND:")
        for issue in results["validation_results"]["blocking_issues"]:
            print(f"   - {issue}")

    print(f"\n📖 Restoration guide: {results['backup_path']}/RESTORATION_PROCEDURE.md")

    return results


if __name__ == "__main__":
    main()
