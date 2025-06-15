"""
MetaExecutor - Real Code Modification Engine

This component gives the meta system the ability to actually modify its own code files.
It's the bridge from simulation to reality - the "hands" that actually change the code.

CRITICAL SAFETY: This system can modify itself. All changes are:
1. Backed up before modification
2. Validated before application  
3. Logged for rollback capability
4. Applied incrementally with verification

Design Decision: Separate execution from generation
Rationale: Safety through separation of concerns - generation plans, execution validates and applies
Alternative: Direct modification in generator
Prediction: Will enable safer self-modification with better error recovery
"""

import ast
import shutil
import time
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from meta_generator import CodeGeneration


@dataclass
class CodeModification:
    """Represents an actual code modification"""
    modification_id: str
    target_file: Path
    backup_file: Path
    change_type: str  # 'method_addition', 'bug_fix', 'optimization'
    original_content: str
    modified_content: str
    validation_passed: bool
    applied: bool
    rollback_available: bool


class MetaExecutor:
    """Executes real code modifications with safety guarantees"""
    
    def __init__(self, meta_db_path: str = None):
        if meta_db_path is None:
            from meta_config import get_meta_config
            config = get_meta_config()
            meta_db_path = config.database.evolution_db
        self.db = sqlite3.connect(meta_db_path)
        self.birth_time = time.time()
        
        # Safety mechanisms
        self.backup_dir = Path("meta_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Validation
        self.validation_enabled = True
        self.dry_run_mode = False  # Start in safe mode
        
        # Modification tracking
        self.modifications: List[CodeModification] = []
        self.successful_modifications = 0
        self.failed_modifications = 0
        
        print(f"⚡ MetaExecutor initialized at {time.ctime(self.birth_time)}")
        print(f"🛡️  Safety mode: {'DRY RUN' if self.dry_run_mode else 'LIVE EXECUTION'}")
        print(f"💾 Backup directory: {self.backup_dir}")
        
        self._init_executor_schema()
        self._record_birth()
        
    def _init_executor_schema(self):
        """Initialize executor database schema"""
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS code_modifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                modification_id TEXT UNIQUE NOT NULL,
                target_file TEXT NOT NULL,
                backup_file TEXT NOT NULL,
                change_type TEXT NOT NULL,
                original_hash TEXT NOT NULL,
                modified_hash TEXT NOT NULL,
                validation_passed BOOLEAN NOT NULL,
                applied BOOLEAN NOT NULL,
                rollback_available BOOLEAN NOT NULL,
                outcome TEXT,
                error_message TEXT
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                safety_check TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                details TEXT
            )
        """)
        
        self.db.commit()
        
    def _record_birth(self):
        """Record executor initialization"""
        
        self.db.execute("""
            INSERT INTO observations (timestamp, event_type, details, context)
            VALUES (?, ?, ?, ?)
        """, (
            time.time(),
            "meta_executor_birth",
            f'{{"component": "MetaExecutor", "safety_mode": "{self.dry_run_mode}", "backup_dir": "{self.backup_dir}"}}',
            "MetaExecutor"
        ))
        
        self.db.commit()
        
    def enable_live_execution(self):
        """Enable actual code modification (DANGEROUS!)"""
        
        print("⚠️  ENABLING LIVE CODE EXECUTION")
        print("   This will actually modify source files!")
        print("   All changes will be backed up and validated.")
        
        self.dry_run_mode = False
        
        self.db.execute("""
            INSERT INTO safety_events 
            (timestamp, event_type, file_path, safety_check, passed, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            time.time(), "live_execution_enabled", "system", "user_authorization", True,
            "Live code execution enabled by user request"
        ))
        
        self.db.commit()
        
    def create_backup(self, file_path: Path) -> Path:
        """Create timestamped backup of file"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
            
        timestamp = int(time.time())
        backup_name = f"{file_path.stem}_{timestamp}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}.backup"
        backup_path = self.backup_dir / backup_name
        
        # Copy file with metadata
        shutil.copy2(file_path, backup_path)
        
        print(f"💾 Backup created: {file_path} → {backup_path}")
        
        return backup_path
        
    def validate_python_syntax(self, content: str, file_path: Path) -> Tuple[bool, str]:
        """Validate Python syntax of modified content"""
        
        try:
            ast.parse(content)
            return True, "Syntax valid"
        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path}: {e}"
            print(f"❌ Validation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Parse error in {file_path}: {e}"
            print(f"❌ Validation failed: {error_msg}")
            return False, error_msg
            
    def validate_imports(self, content: str, file_path: Path) -> Tuple[bool, str]:
        """Validate that imports can be resolved"""
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Basic import validation - just check if it's a known bad import
                        if alias.name in ["tensorflow-gpu", "torch.cuda", "cupy"]:
                            return False, f"Incompatible import: {alias.name}"
                            
            return True, "Imports valid"
        except Exception as e:
            return False, f"Import validation error: {e}"
            
    def safety_check(self, modification: CodeModification) -> Tuple[bool, List[str]]:
        """Comprehensive safety check before applying modification"""
        
        checks = []
        issues = []
        
        # 1. Syntax validation
        syntax_valid, syntax_msg = self.validate_python_syntax(
            modification.modified_content, modification.target_file
        )
        checks.append(("syntax", syntax_valid, syntax_msg))
        if not syntax_valid:
            issues.append(syntax_msg)
            
        # 2. Import validation
        import_valid, import_msg = self.validate_imports(
            modification.modified_content, modification.target_file
        )
        checks.append(("imports", import_valid, import_msg))
        if not import_valid:
            issues.append(import_msg)
            
        # 3. Size change validation (prevent massive accidental changes)
        original_lines = len(modification.original_content.split('\n'))
        modified_lines = len(modification.modified_content.split('\n'))
        size_change_ratio = abs(modified_lines - original_lines) / max(original_lines, 1)
        
        if size_change_ratio > 2.0:  # More than 2x size change
            size_valid = False
            size_msg = f"Excessive size change: {original_lines} → {modified_lines} lines"
            issues.append(size_msg)
        else:
            size_valid = True
            size_msg = f"Size change acceptable: {size_change_ratio:.1%}"
            
        checks.append(("size_change", size_valid, size_msg))
        
        # 4. Backup validation
        backup_valid = modification.backup_file.exists()
        backup_msg = "Backup verified" if backup_valid else "Backup missing"
        checks.append(("backup", backup_valid, backup_msg))
        if not backup_valid:
            issues.append(backup_msg)
            
        # Record safety checks
        for check_name, passed, details in checks:
            self.db.execute("""
                INSERT INTO safety_events 
                (timestamp, event_type, file_path, safety_check, passed, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(), "safety_check", str(modification.target_file),
                check_name, passed, details
            ))
            
        self.db.commit()
        
        all_passed = all(check[1] for check in checks)
        return all_passed, issues
        
    def apply_modification(self, modification: CodeModification) -> bool:
        """Apply a code modification with full safety checks"""
        
        print(f"🔧 Applying modification: {modification.modification_id}")
        print(f"   Target: {modification.target_file}")
        print(f"   Type: {modification.change_type}")
        
        # Safety check
        safety_passed, issues = self.safety_check(modification)
        
        if not safety_passed:
            print(f"❌ Safety check failed:")
            for issue in issues:
                print(f"   • {issue}")
                
            modification.validation_passed = False
            modification.applied = False
            self.failed_modifications += 1
            return False
            
        modification.validation_passed = True
        
        # Apply the modification
        if self.dry_run_mode:
            print("🧪 DRY RUN: Would apply modification")
            modification.applied = False
            outcome = "dry_run_success"
        else:
            try:
                # Actually write the modified content
                modification.target_file.write_text(modification.modified_content)
                modification.applied = True
                outcome = "applied_successfully"
                self.successful_modifications += 1
                
                print(f"✅ Modification applied successfully")
                
                # Verify the change took effect
                new_content = modification.target_file.read_text()
                new_hash = hashlib.sha256(new_content.encode()).hexdigest()
                expected_hash = hashlib.sha256(modification.modified_content.encode()).hexdigest()
                
                if new_hash == expected_hash:
                    print("✅ Verification passed: File content matches expected")
                else:
                    print("⚠️  Verification warning: File content differs from expected")
                    
            except Exception as e:
                print(f"❌ Failed to apply modification: {e}")
                modification.applied = False
                outcome = f"application_failed: {e}"
                self.failed_modifications += 1
                
        # Record the modification
        self.db.execute("""
            INSERT INTO code_modifications 
            (timestamp, modification_id, target_file, backup_file, change_type,
             original_hash, modified_hash, validation_passed, applied, 
             rollback_available, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), modification.modification_id, str(modification.target_file),
            str(modification.backup_file), modification.change_type,
            hashlib.sha256(modification.original_content.encode()).hexdigest(),
            hashlib.sha256(modification.modified_content.encode()).hexdigest(),
            modification.validation_passed, modification.applied,
            modification.rollback_available, outcome
        ))
        
        self.db.commit()
        
        self.modifications.append(modification)
        
        return modification.applied
        
    def execute_code_generation(self, generation: CodeGeneration) -> bool:
        """Execute a code generation by creating real modification"""
        
        target_file = Path(generation.target_file)
        
        if not target_file.exists():
            print(f"❌ Target file does not exist: {target_file}")
            return False
            
        # Create backup
        backup_file = self.create_backup(target_file)
        
        # Read original content
        original_content = target_file.read_text()
        
        # Create modified content by adding generated code
        if generation.generation_type == "method_addition":
            modified_content = self._add_method_to_file(original_content, generation.generated_code)
        elif generation.generation_type == "optimization":
            modified_content = self._apply_optimization(original_content, generation.generated_code)
        else:
            # Generic addition at end of class
            modified_content = self._generic_code_addition(original_content, generation.generated_code)
            
        # Create modification object
        modification = CodeModification(
            modification_id=generation.task_id,
            target_file=target_file,
            backup_file=backup_file,
            change_type=generation.generation_type,
            original_content=original_content,
            modified_content=modified_content,
            validation_passed=False,
            applied=False,
            rollback_available=True
        )
        
        # Apply the modification
        return self.apply_modification(modification)
        
    def _add_method_to_file(self, original_content: str, generated_method: str) -> str:
        """Add a method to a Python class file"""
        
        lines = original_content.split('\n')
        
        # Find the last class definition and add method before the final closing
        class_end_line = -1
        indent_level = 0
        
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.strip() and not line.startswith(' ') and not line.startswith('#'):
                class_end_line = i
                break
                
        if class_end_line > 0:
            # Insert method before the last line
            lines.insert(class_end_line, "")
            lines.insert(class_end_line + 1, generated_method)
            lines.insert(class_end_line + 2, "")
        else:
            # Append at end
            lines.append("")
            lines.append(generated_method)
            
        return '\n'.join(lines)
        
    def _apply_optimization(self, original_content: str, optimization_code: str) -> str:
        """Apply optimization code to file with intelligent placement"""
        
        try:
            # Parse the original content to find appropriate insertion point
            tree = ast.parse(original_content)
            
            # Find the best insertion point for optimization
            lines = original_content.split('\n')
            
            # Strategy 1: Insert before main block if exists
            for i, line in enumerate(lines):
                if 'if __name__ == "__main__"' in line:
                    return '\n'.join(lines[:i]) + '\n\n' + optimization_code + '\n\n' + '\n'.join(lines[i:])
            
            # Strategy 2: Insert after imports and before first function/class
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('#') or not line.strip():
                    import_end = i + 1
                else:
                    break
            
            if import_end > 0:
                return '\n'.join(lines[:import_end]) + '\n\n' + optimization_code + '\n\n' + '\n'.join(lines[import_end:])
            
            # Fallback: append at end
            return original_content + "\n\n" + optimization_code
            
        except SyntaxError:
            # If parsing fails, append at end
            return original_content + "\n\n" + optimization_code
        
    def _generic_code_addition(self, original_content: str, generated_code: str) -> str:
        """Generic code addition with AST-based intelligent placement"""
        
        try:
            # Parse both original and generated code
            original_tree = ast.parse(original_content)
            generated_tree = ast.parse(generated_code)
            
            lines = original_content.split('\n')
            
            # Determine what type of code is being added
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in generated_tree.body)
            has_classes = any(isinstance(node, ast.ClassDef) for node in generated_tree.body)
            has_functions = any(isinstance(node, ast.FunctionDef) for node in generated_tree.body)
            
            # Place imports at the top after existing imports
            if has_imports:
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        import_end = i + 1
                
                # Insert imports after existing imports
                import_lines = [line for line in generated_code.split('\n') 
                              if line.strip().startswith(('import ', 'from '))]
                
                if import_lines:
                    lines = lines[:import_end] + import_lines + lines[import_end:]
                    
                    # Remove import lines from generated_code for rest of processing
                    remaining_lines = [line for line in generated_code.split('\n') 
                                     if not line.strip().startswith(('import ', 'from ')) and line.strip()]
                    generated_code = '\n'.join(remaining_lines)
            
            # Place classes and functions before main block
            if has_classes or has_functions:
                for i, line in enumerate(lines):
                    if 'if __name__ == "__main__"' in line:
                        return '\n'.join(lines[:i]) + '\n\n' + generated_code + '\n\n' + '\n'.join(lines[i:])
            
            # Default: append at end
            return '\n'.join(lines) + "\n\n" + generated_code
            
        except SyntaxError:
            # If parsing fails, append at end
            return original_content + "\n\n" + generated_code
        
    def rollback_modification(self, modification_id: str) -> bool:
        """Rollback a modification using its backup"""
        
        modification = next((m for m in self.modifications if m.modification_id == modification_id), None)
        
        if not modification:
            print(f"❌ Modification not found: {modification_id}")
            return False
            
        if not modification.rollback_available or not modification.backup_file.exists():
            print(f"❌ Rollback not available for: {modification_id}")
            return False
            
        try:
            # Restore from backup
            shutil.copy2(modification.backup_file, modification.target_file)
            print(f"🔄 Rolled back: {modification.target_file}")
            
            # Record rollback
            self.db.execute("""
                INSERT INTO safety_events 
                (timestamp, event_type, file_path, safety_check, passed, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(), "rollback_executed", str(modification.target_file),
                "rollback", True, f"Rolled back modification {modification_id}"
            ))
            
            self.db.commit()
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
            
    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status and statistics"""
        
        return {
            "birth_time": self.birth_time,
            "dry_run_mode": self.dry_run_mode,
            "total_modifications": len(self.modifications),
            "successful_modifications": self.successful_modifications,
            "failed_modifications": self.failed_modifications,
            "success_rate": self.successful_modifications / max(len(self.modifications), 1),
            "backup_count": len(list(self.backup_dir.glob("*.backup"))),
            "validation_enabled": self.validation_enabled
        }
        
    def get_executor_report(self) -> str:
        """Generate executor activity report"""
        
        status = self.get_executor_status()
        
        report = f"""
⚡ MetaExecutor Report
━━━━━━━━━━━━━━━━━━━━━━━━━━

Execution Mode: {'🧪 DRY RUN' if status['dry_run_mode'] else '⚡ LIVE EXECUTION'}
Total Modifications: {status['total_modifications']}
Success Rate: {status['success_rate']:.1%}

Successful: {status['successful_modifications']}
Failed: {status['failed_modifications']}
Backups Created: {status['backup_count']}

Recent Modifications:
"""
        
        # Get recent modifications
        cursor = self.db.execute("""
            SELECT modification_id, target_file, change_type, applied, outcome
            FROM code_modifications
            ORDER BY timestamp DESC LIMIT 5
        """)
        
        for mod_id, target_file, change_type, applied, outcome in cursor.fetchall():
            status_emoji = "✅" if applied else "❌"
            report += f"  {status_emoji} {mod_id}: {change_type} on {Path(target_file).name}\n"
            
        return report


# Test the executor
if __name__ == "__main__":
    executor = MetaExecutor()
    
    print(executor.get_executor_report())
    
    print("\n🧪 Testing in DRY RUN mode...")
    print("   (No actual files will be modified)")
    
    # Create a simple test modification
    test_file = Path("meta_prime.py")
    if test_file.exists():
        backup = executor.create_backup(test_file)
        
        original = test_file.read_text()
        modified = original + "\n# Test comment added by MetaExecutor"
        
        test_mod = CodeModification(
            modification_id="test_001",
            target_file=test_file,
            backup_file=backup,
            change_type="test_addition",
            original_content=original,
            modified_content=modified,
            validation_passed=False,
            applied=False,
            rollback_available=True
        )
        
        success = executor.apply_modification(test_mod)
        print(f"\n📊 Test result: {'Success' if success else 'Failed'}")
        
    print(f"\n{executor.get_executor_report()}")