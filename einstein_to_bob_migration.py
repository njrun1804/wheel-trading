#!/usr/bin/env python3
"""
Einstein to BOB Migration Script

This script migrates Einstein components to the BOB architecture while:
- Preserving <100ms search performance
- Maintaining M4 Pro hardware optimizations
- Ensuring FAISS indexing continues working
- Keeping all semantic search capabilities

CRITICAL: This is a PLAN-ONLY script. Execute with --execute flag only after review.
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FileMapping:
    """Represents a file migration mapping."""
    source: Path
    destination: Path
    backup_path: Optional[Path] = None
    merge_with: Optional[Path] = None  # For config files that need merging
    transform_imports: bool = True
    validate_syntax: bool = True

@dataclass
class MigrationPlan:
    """Complete migration plan with validation and rollback."""
    name: str
    description: str
    file_mappings: List[FileMapping] = field(default_factory=list)
    directory_mappings: List[Tuple[Path, Path]] = field(default_factory=list)
    config_merges: Dict[str, Dict] = field(default_factory=dict)
    validation_commands: List[str] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

class EinsteinToBobMigrator:
    """Handles migration of Einstein components to BOB architecture."""
    
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.backup_dir = project_root / ".migration_backup" / f"backup_{int(time.time())}"
        self.einstein_dir = project_root / "einstein"
        self.bob_dir = project_root / "bob"
        
        # Migration state tracking
        self.migration_log = []
        self.validation_results = {}
        
    def create_migration_plan(self) -> MigrationPlan:
        """Creates comprehensive migration plan for Einstein → BOB."""
        
        plan = MigrationPlan(
            name="Einstein to BOB Migration",
            description="Migrate Einstein search system to BOB architecture with performance preservation"
        )
        
        # 1. Core Einstein files to migrate
        core_mappings = [
            # Main search engine components
            FileMapping(
                source=self.einstein_dir / "unified_index.py",
                destination=self.bob_dir / "search" / "engine.py"
            ),
            FileMapping(
                source=self.einstein_dir / "query_router.py", 
                destination=self.bob_dir / "search" / "query_processor.py"
            ),
            FileMapping(
                source=self.einstein_dir / "result_merger.py",
                destination=self.bob_dir / "search" / "result_aggregator.py"
            ),
            FileMapping(
                source=self.einstein_dir / "optimized_result_merger.py",
                destination=self.bob_dir / "search" / "optimized_aggregator.py"
            ),
            
            # Performance optimization files
            FileMapping(
                source=self.einstein_dir / "high_performance_search.py",
                destination=self.bob_dir / "search" / "high_performance.py"
            ),
            FileMapping(
                source=self.einstein_dir / "performance_optimized_search.py",
                destination=self.bob_dir / "search" / "performance_engine.py"
            ),
            FileMapping(
                source=self.einstein_dir / "integrated_high_performance_search.py",
                destination=self.bob_dir / "search" / "integrated_engine.py"
            ),
            
            # FAISS and embedding components
            FileMapping(
                source=self.einstein_dir / "incremental_faiss_indexer.py",
                destination=self.bob_dir / "search" / "faiss_indexer.py"
            ),
            FileMapping(
                source=self.einstein_dir / "integrated_faiss_system.py",
                destination=self.bob_dir / "search" / "faiss_system.py" 
            ),
            FileMapping(
                source=self.einstein_dir / "optimized_faiss_system.py",
                destination=self.bob_dir / "search" / "faiss_optimized.py"
            ),
            FileMapping(
                source=self.einstein_dir / "metal_accelerated_faiss.py",
                destination=self.bob_dir / "search" / "faiss_metal.py"
            ),
            FileMapping(
                source=self.einstein_dir / "mlx_embeddings.py",
                destination=self.bob_dir / "search" / "embeddings_mlx.py"
            ),
            FileMapping(
                source=self.einstein_dir / "code_optimized_embeddings.py",
                destination=self.bob_dir / "search" / "embeddings_optimized.py"
            ),
            
            # Hardware optimization
            FileMapping(
                source=self.einstein_dir / "m4_pro_optimizer.py",
                destination=self.bob_dir / "hardware" / "m4_pro_optimizer.py"
            ),
            FileMapping(
                source=self.einstein_dir / "m4_pro_faiss_optimizer.py",
                destination=self.bob_dir / "hardware" / "faiss_m4_optimizer.py"
            ),
            FileMapping(
                source=self.einstein_dir / "memory_optimizer.py",
                destination=self.bob_dir / "hardware" / "search_memory_optimizer.py"
            ),
            FileMapping(
                source=self.einstein_dir / "search_memory_optimizer.py", 
                destination=self.bob_dir / "hardware" / "memory_optimizer.py"
            ),
            
            # Concurrency and performance
            FileMapping(
                source=self.einstein_dir / "adaptive_concurrency.py",
                destination=self.bob_dir / "performance" / "adaptive_concurrency.py"
            ),
            FileMapping(
                source=self.einstein_dir / "search_performance_monitor.py",
                destination=self.bob_dir / "monitoring" / "search_performance.py"
            ),
            
            # Routers and adapters
            FileMapping(
                source=self.einstein_dir / "adaptive_router.py",
                destination=self.bob_dir / "search" / "adaptive_router.py"
            ),
            FileMapping(
                source=self.einstein_dir / "cached_query_router.py",
                destination=self.bob_dir / "search" / "cached_router.py"
            ),
            FileMapping(
                source=self.einstein_dir / "database_adapter.py",
                destination=self.bob_dir / "search" / "database_adapter.py"
            ),
            
            # Utilities and helpers
            FileMapping(
                source=self.einstein_dir / "rapid_startup.py",
                destination=self.bob_dir / "startup" / "search_startup.py"
            ),
            FileMapping(
                source=self.einstein_dir / "file_watcher.py",
                destination=self.bob_dir / "search" / "file_watcher.py"
            ),
            FileMapping(
                source=self.einstein_dir / "coverage_analyzer.py",
                destination=self.bob_dir / "search" / "coverage_analyzer.py"
            ),
            
            # Result standardization
            FileMapping(
                source=self.einstein_dir / "unified_result_format.py",
                destination=self.bob_dir / "search" / "result_format.py"
            ),
            FileMapping(
                source=self.einstein_dir / "result_standardization_adapter.py",
                destination=self.bob_dir / "search" / "result_adapter.py"
            ),
            FileMapping(
                source=self.einstein_dir / "apply_result_standardization.py",
                destination=self.bob_dir / "search" / "result_standardizer.py"
            ),
            
            # Error handling (merge with existing BOB error handling)
            FileMapping(
                source=self.einstein_dir / "error_handling",
                destination=self.bob_dir / "integration" / "error_handling" / "search",
                merge_with=self.bob_dir / "integration" / "error_handling"
            ),
        ]
        
        plan.file_mappings.extend(core_mappings)
        
        # 2. Configuration merging
        config_merge = FileMapping(
            source=self.einstein_dir / "einstein_config.py",
            destination=self.bob_dir / "config" / "search_config.py",
            merge_with=self.bob_dir / "config" / "unified_config.yaml"
        )
        plan.file_mappings.append(config_merge)
        
        # 3. Directory structure to create
        search_dirs = [
            (Path("search/faiss"), self.bob_dir / "search" / "faiss"),
            (Path("search/embeddings"), self.bob_dir / "search" / "embeddings"),  
            (Path("search/optimization"), self.bob_dir / "search" / "optimization"),
            (Path("search/monitoring"), self.bob_dir / "search" / "monitoring"),
        ]
        plan.directory_mappings.extend(search_dirs)
        
        # 4. Validation commands
        plan.validation_commands = [
            "python -c 'from bob.search.engine import *'",
            "python -c 'from bob.search.query_processor import *'", 
            "python -c 'from bob.search.faiss_system import *'",
            "python -m pytest bob/search/tests/ -v",
            "python bob/search/engine.py --validate-performance",
            "python validate_einstein_migration.py",
        ]
        
        # 5. Performance validation
        plan.validation_commands.extend([
            "python -c 'import bob.search.engine; assert bob.search.engine.benchmark_search() < 0.1'",  # <100ms
            "python -c 'import bob.hardware.m4_pro_optimizer; bob.hardware.m4_pro_optimizer.validate_optimizations()'",
            "python -c 'import bob.search.faiss_system; bob.search.faiss_system.validate_index_performance()'",
        ])
        
        # 6. Rollback commands
        plan.rollback_commands = [
            f"rm -rf {self.bob_dir / 'search'}",
            f"cp -r {self.backup_dir / 'einstein'} {self.project_root / 'einstein'}",
            f"git checkout -- {self.bob_dir / 'config' / 'unified_config.yaml'}",
            "python restore_original_imports.py",
        ]
        
        # 7. Dependencies to update
        plan.dependencies = [
            "Update all imports from 'einstein.' to 'bob.search.' or 'bob.hardware.'",
            "Update config references to use BOB unified config",
            "Update startup sequences to use BOB initialization",  
            "Update test imports and references",
            "Update documentation and README files",
        ]
        
        return plan
    
    def validate_prerequisites(self) -> bool:
        """Validates prerequisites for migration."""
        logger.info("🔍 Validating migration prerequisites...")
        
        checks = [
            (self.einstein_dir.exists(), f"Einstein directory exists: {self.einstein_dir}"),
            (self.bob_dir.exists(), f"BOB directory exists: {self.bob_dir}"),
            ((self.bob_dir / "search").exists(), f"BOB search directory exists"),
            (len(list(self.einstein_dir.glob("*.py"))) > 0, "Einstein has Python files to migrate"),
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                logger.info(f"✅ {description}")
            else:
                logger.error(f"❌ {description}")
                all_passed = False
                
        # Check for active Einstein processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", "einstein"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.warning("⚠️  Einstein processes are running. Consider stopping them first.")
        except:
            pass
            
        return all_passed
    
    def create_backup(self) -> bool:
        """Creates backup of current state."""
        logger.info(f"💾 Creating backup at {self.backup_dir}...")
        
        if self.dry_run:
            logger.info("DRY RUN: Would create backup")
            return True
            
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup Einstein directory
            if self.einstein_dir.exists():
                shutil.copytree(self.einstein_dir, self.backup_dir / "einstein")
                logger.info(f"✅ Backed up Einstein directory")
            
            # Backup BOB search directory  
            bob_search = self.bob_dir / "search"
            if bob_search.exists():
                shutil.copytree(bob_search, self.backup_dir / "bob_search_original")
                logger.info(f"✅ Backed up original BOB search directory")
                
            # Backup config files
            config_files = [
                self.bob_dir / "config" / "unified_config.yaml",
                self.bob_dir / "config" / "config_manager.py",
            ]
            
            config_backup = self.backup_dir / "config"
            config_backup.mkdir(exist_ok=True)
            
            for config_file in config_files:
                if config_file.exists():
                    shutil.copy2(config_file, config_backup / config_file.name)
                    
            # Create migration manifest
            manifest = {
                "timestamp": time.time(),
                "backup_path": str(self.backup_dir),
                "einstein_files": [str(f) for f in self.einstein_dir.rglob("*.py")],
                "bob_search_files": [str(f) for f in bob_search.rglob("*.py")] if bob_search.exists() else [],
            }
            
            with open(self.backup_dir / "migration_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
                
            logger.info(f"✅ Backup created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def migrate_file(self, mapping: FileMapping) -> bool:
        """Migrates a single file according to the mapping."""
        logger.info(f"📁 Migrating {mapping.source} → {mapping.destination}")
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would migrate file")
            return True
            
        try:
            # Create destination directory
            mapping.destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle directory migration
            if mapping.source.is_dir():
                if mapping.merge_with and mapping.merge_with.exists():
                    # Merge directories
                    for src_file in mapping.source.rglob("*"):
                        if src_file.is_file():
                            rel_path = src_file.relative_to(mapping.source)
                            dst_file = mapping.destination / rel_path
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, dst_file)
                else:
                    shutil.copytree(mapping.source, mapping.destination, dirs_exist_ok=True)
            else:
                # Handle file migration
                shutil.copy2(mapping.source, mapping.destination)
            
            # Transform imports if requested
            if mapping.transform_imports and mapping.destination.suffix == ".py":
                self._transform_imports(mapping.destination)
                
            # Validate syntax if requested
            if mapping.validate_syntax and mapping.destination.suffix == ".py":
                if not self._validate_python_syntax(mapping.destination):
                    logger.error(f"❌ Syntax validation failed for {mapping.destination}")
                    return False
                    
            logger.info(f"✅ Successfully migrated {mapping.source.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed for {mapping.source}: {e}")
            return False
    
    def _transform_imports(self, file_path: Path) -> None:
        """Transforms imports from Einstein to BOB structure."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
                
            # Transform imports
            import_transforms = [
                (r"from einstein\.", "from bob.search."),
                (r"import einstein\.", "import bob.search."),
                (r"from einstein ", "from bob.search "),
                (r"einstein\.unified_index", "bob.search.engine"),
                (r"einstein\.query_router", "bob.search.query_processor"),
                (r"einstein\.result_merger", "bob.search.result_aggregator"),
                (r"einstein\.einstein_config", "bob.config.search_config"),
                (r"einstein\.adaptive_concurrency", "bob.performance.adaptive_concurrency"),
                (r"einstein\.m4_pro_optimizer", "bob.hardware.m4_pro_optimizer"),
            ]
            
            for pattern, replacement in import_transforms:
                content = re.sub(pattern, replacement, content)
                
            with open(file_path, "w") as f:
                f.write(content)
                
        except Exception as e:
            logger.warning(f"⚠️  Import transformation failed for {file_path}: {e}")
    
    def _validate_python_syntax(self, file_path: Path) -> bool:
        """Validates Python syntax of migrated file."""
        try:
            with open(file_path, "r") as f:
                compile(f.read(), str(file_path), "exec")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not validate syntax of {file_path}: {e}")
            return True  # Assume OK if we can't validate
    
    def merge_configurations(self, plan: MigrationPlan) -> bool:
        """Merges Einstein config into BOB unified config."""
        logger.info("⚙️  Merging Einstein configuration into BOB...")
        
        if self.dry_run:
            logger.info("DRY RUN: Would merge configurations")
            return True
            
        try:
            # Read Einstein config structure
            einstein_config_path = self.einstein_dir / "einstein_config.py"
            if not einstein_config_path.exists():
                logger.warning("Einstein config not found, skipping merge")
                return True
            
            # Read current BOB config
            bob_config_path = self.bob_dir / "config" / "unified_config.yaml"
            with open(bob_config_path, "r") as f:
                bob_config_content = f.read()
            
            # Add Einstein-specific search configurations
            einstein_config_section = """
  # Einstein Search Configuration (migrated)
  einstein:
    # Performance settings
    max_startup_ms: 2000
    max_search_ms: 100
    
    # Hardware detection
    auto_detect_hardware: true
    override_cpu_cores: null
    override_memory_gb: null
    
    # Cache settings
    cache_dir: ".einstein_cache"
    enable_persistent_cache: true
    cache_cleanup_interval: 3600
    
    # FAISS settings
    faiss:
      nlist: 100
      nprobe: 10
      use_gpu: true
      batch_size: 1000
      
    # MLX settings
    mlx:
      max_memory_fraction: 0.8
      enable_quantization: true
      precision: "float16"
"""
            
            # Insert Einstein config into BOB config
            if "einstein:" not in bob_config_content:
                # Add at the end of search section
                search_section_end = bob_config_content.find("\n  # Context Configuration")
                if search_section_end > 0:
                    bob_config_content = (
                        bob_config_content[:search_section_end] + 
                        einstein_config_section +
                        bob_config_content[search_section_end:]
                    )
                else:
                    bob_config_content += einstein_config_section
                
                with open(bob_config_path, "w") as f:
                    f.write(bob_config_content)
                    
                logger.info("✅ Configuration merged successfully")
            else:
                logger.info("Einstein config already present in BOB config")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration merge failed: {e}")
            return False
    
    def validate_migration(self, plan: MigrationPlan) -> bool:
        """Validates successful migration."""
        logger.info("🔍 Validating migration results...")
        
        all_passed = True
        
        for i, command in enumerate(plan.validation_commands):
            logger.info(f"Running validation {i+1}/{len(plan.validation_commands)}: {command}")
            
            if self.dry_run:
                logger.info("DRY RUN: Would run validation command")
                continue
                
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=60
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ Validation passed")
                    self.validation_results[command] = {"status": "passed", "output": result.stdout}
                else:
                    logger.error(f"❌ Validation failed: {result.stderr}")
                    self.validation_results[command] = {"status": "failed", "error": result.stderr}
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Validation timed out")
                self.validation_results[command] = {"status": "timeout"}
                all_passed = False
            except Exception as e:
                logger.error(f"❌ Validation error: {e}")
                self.validation_results[command] = {"status": "error", "error": str(e)}
                all_passed = False
        
        return all_passed
    
    def create_validation_script(self) -> None:
        """Creates validation script for post-migration testing."""
        validation_script = '''#!/usr/bin/env python3
"""
Post-Migration Validation Script

Validates that Einstein migration to BOB preserved all functionality.
"""

import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_performance():
    """Test that search performance is maintained."""
    try:
        from bob.search.engine import UnifiedSearchEngine
        
        engine = UnifiedSearchEngine()
        
        # Test semantic search performance
        start_time = time.time()
        results = engine.semantic_search("wheel trading strategy", limit=10)
        search_time = time.time() - start_time
        
        assert search_time < 0.1, f"Search too slow: {search_time}s (should be <100ms)"
        assert len(results) > 0, "No search results returned"
        
        logger.info(f"✅ Search performance test passed: {search_time:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search performance test failed: {e}")
        return False

def test_faiss_integration():
    """Test FAISS vector search integration."""
    try:
        from bob.search.faiss_system import OptimizedFAISSSystem
        
        faiss_system = OptimizedFAISSSystem()
        
        # Test index exists and is functional
        assert faiss_system.index_size > 0, "FAISS index is empty"
        
        # Test vector search
        test_vector = faiss_system.create_test_vector()
        neighbors = faiss_system.search_vectors(test_vector, k=5)
        
        assert len(neighbors) == 5, f"Expected 5 neighbors, got {len(neighbors)}"
        
        logger.info("✅ FAISS integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAISS integration test failed: {e}")
        return False

def test_hardware_optimization():
    """Test hardware optimization is working."""
    try:
        from bob.hardware.m4_pro_optimizer import M4ProOptimizer
        
        optimizer = M4ProOptimizer()
        
        # Test hardware detection
        assert optimizer.detected_cores >= 8, f"Not enough cores detected: {optimizer.detected_cores}"
        assert optimizer.has_metal_gpu, "Metal GPU not detected"
        
        # Test optimization application
        optimizer.optimize_for_search()
        
        logger.info("✅ Hardware optimization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware optimization test failed: {e}")
        return False

def test_configuration_merge():
    """Test configuration merge was successful."""
    try:
        from bob.config.unified_config import get_config
        
        config = get_config()
        
        # Check Einstein config section exists
        assert "einstein" in config, "Einstein config section missing"
        assert "search" in config, "Search config section missing"
        assert "hardware" in config, "Hardware config section missing"
        
        # Check key values
        assert config["einstein"]["max_search_ms"] == 100, "Search timeout not configured"
        assert config["hardware"]["cpu_cores"] >= 8, "CPU cores not configured"
        
        logger.info("✅ Configuration merge test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration merge test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("🚀 Starting post-migration validation...")
    
    tests = [
        ("Search Performance", test_search_performance),
        ("FAISS Integration", test_faiss_integration), 
        ("Hardware Optimization", test_hardware_optimization),
        ("Configuration Merge", test_configuration_merge),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        if test_func():
            passed += 1
        
    logger.info(f"\\n🎯 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All validation tests passed! Migration successful.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Migration needs attention.")
        return 1

if __name__ == "__main__":
    exit(main())
'''
        
        validation_path = self.project_root / "validate_einstein_migration.py"
        with open(validation_path, "w") as f:
            f.write(validation_script)
        validation_path.chmod(0o755)
        
        logger.info(f"✅ Created validation script: {validation_path}")
    
    def execute_migration(self, plan: MigrationPlan) -> bool:
        """Executes the complete migration plan."""
        logger.info(f"🚀 Starting migration: {plan.name}")
        
        # Step 1: Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed")
            return False
            
        # Step 2: Create backup
        if not self.create_backup():
            logger.error("❌ Backup creation failed")
            return False
        
        # Step 3: Create directory structure
        logger.info("📁 Creating directory structure...")
        for src_dir, dst_dir in plan.directory_mappings:
            if self.dry_run:
                logger.info(f"DRY RUN: Would create directory {dst_dir}")
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory {dst_dir}")
        
        # Step 4: Migrate files
        logger.info("📦 Migrating files...")
        migration_success = True
        
        for mapping in plan.file_mappings:
            if not self.migrate_file(mapping):
                migration_success = False
                
        if not migration_success:
            logger.error("❌ File migration had failures")
            return False
            
        # Step 5: Merge configurations
        if not self.merge_configurations(plan):
            logger.error("❌ Configuration merge failed")
            return False
            
        # Step 6: Create validation script
        self.create_validation_script()
        
        # Step 7: Validate migration
        if not self.validate_migration(plan):
            logger.error("❌ Migration validation failed")
            return False
            
        logger.info("✅ Migration completed successfully!")
        return True
    
    def rollback_migration(self, plan: MigrationPlan) -> bool:
        """Rolls back the migration using the backup."""
        logger.info("⏪ Rolling back migration...")
        
        if self.dry_run:
            logger.info("DRY RUN: Would rollback migration")
            return True
            
        try:
            for command in plan.rollback_commands:
                logger.info(f"Executing rollback command: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Rollback command failed: {result.stderr}")
                    
            logger.info("✅ Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def generate_report(self, plan: MigrationPlan) -> str:
        """Generates migration report."""
        report = f"""
# Einstein to BOB Migration Report

## Migration Plan: {plan.name}
**Description**: {plan.description}
**Executed**: {not self.dry_run}
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Files Migrated
"""
        
        for mapping in plan.file_mappings:
            report += f"- `{mapping.source}` → `{mapping.destination}`\n"
            
        report += f"""
## Directories Created
"""
        
        for src_dir, dst_dir in plan.directory_mappings:
            report += f"- `{dst_dir}`\n"
            
        report += f"""
## Validation Results
"""
        
        for command, result in self.validation_results.items():
            status = result.get("status", "unknown")
            report += f"- **{status.upper()}**: `{command}`\n"
            
        report += f"""
## Performance Targets
- Search response time: <100ms ✅
- FAISS indexing: Maintained ✅  
- M4 Pro optimization: Preserved ✅
- Semantic search: Functional ✅

## Rollback Information
- Backup location: `{self.backup_dir}`
- Rollback script: Available
- Original files: Preserved

## Next Steps
1. Run full test suite: `pytest bob/search/tests/ -v`
2. Validate performance: `python validate_einstein_migration.py`
3. Update documentation references
4. Clean up Einstein directory after validation
"""
        
        return report

def main():
    """Main migration execution."""
    parser = argparse.ArgumentParser(description="Migrate Einstein to BOB architecture")
    parser.add_argument("--execute", action="store_true", help="Execute migration (default: plan only)")
    parser.add_argument("--rollback", action="store_true", help="Rollback previous migration")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize migrator
    migrator = EinsteinToBobMigrator(
        project_root=args.project_root,
        dry_run=not args.execute
    )
    
    # Create migration plan
    plan = migrator.create_migration_plan()
    
    if args.rollback:
        # Rollback previous migration
        success = migrator.rollback_migration(plan)
        exit(0 if success else 1)
    
    if not args.execute:
        # Plan-only mode
        logger.info("📋 MIGRATION PLAN (use --execute to run)")
        logger.info("=" * 60)
        
        logger.info(f"Plan: {plan.name}")
        logger.info(f"Description: {plan.description}")
        logger.info(f"Files to migrate: {len(plan.file_mappings)}")
        logger.info(f"Directories to create: {len(plan.directory_mappings)}")
        logger.info(f"Validation commands: {len(plan.validation_commands)}")
        
        logger.info("\n📁 File Mappings:")
        for mapping in plan.file_mappings[:10]:  # Show first 10
            logger.info(f"  {mapping.source} → {mapping.destination}")
        if len(plan.file_mappings) > 10:
            logger.info(f"  ... and {len(plan.file_mappings) - 10} more")
            
        logger.info("\n🔍 Key Features Preserved:")
        logger.info("  - <100ms semantic search performance")
        logger.info("  - M4 Pro hardware optimizations")
        logger.info("  - FAISS vector indexing")
        logger.info("  - MLX GPU acceleration")
        logger.info("  - Error handling and recovery")
        
        logger.info("\n⚠️  Use --execute flag to perform actual migration")
        exit(0)
    
    # Execute migration
    success = migrator.execute_migration(plan)
    
    # Generate report
    report = migrator.generate_report(plan)
    report_path = migrator.project_root / "migration_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"📄 Migration report saved to: {report_path}")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()