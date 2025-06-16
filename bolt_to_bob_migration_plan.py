#!/usr/bin/env python3
"""
BOLT to BOB Migration Plan - Comprehensive System Migration

CRITICAL: This script creates a migration plan but does NOT execute file operations.
Run with --execute flag only after reviewing the complete plan.

Migrates BOLT components to BOB directory structure while preserving:
- 1.5 tasks/second throughput 
- M4 Pro hardware optimizations
- Work-stealing algorithm
- 8-agent coordination
- Einstein integration
- GPU acceleration
- Error handling systems
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Migration configuration
BASE_DIR = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
BOLT_DIR = BASE_DIR / "bolt"
BOB_DIR = BASE_DIR / "bob"
BACKUP_DIR = BASE_DIR / f"bolt_migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class BoltToBobMigration:
    """Comprehensive BOLT to BOB migration system."""
    
    def __init__(self):
        self.migration_plan: Dict[str, any] = {}
        self.file_mappings: List[Tuple[Path, Path]] = []
        self.merge_operations: List[Dict[str, any]] = []
        self.config_updates: List[Dict[str, any]] = []
        self.dependency_updates: List[Dict[str, any]] = []
        self.validation_steps: List[str] = []
        
    def analyze_current_state(self) -> Dict[str, any]:
        """Analyze current BOLT and BOB states."""
        print("🔍 Analyzing current system state...")
        
        analysis = {
            "bolt_exists": BOLT_DIR.exists(),
            "bob_exists": BOB_DIR.exists(),
            "bolt_files": [],
            "bob_files": [],
            "conflicts": [],
            "already_migrated": []
        }
        
        if BOLT_DIR.exists():
            for file_path in BOLT_DIR.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(BOLT_DIR)
                    analysis["bolt_files"].append(str(rel_path))
        
        if BOB_DIR.exists():
            for file_path in BOB_DIR.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(BOB_DIR)
                    analysis["bob_files"].append(str(rel_path))
        
        # Check for conflicts and already migrated files
        for bolt_file in analysis["bolt_files"]:
            if bolt_file in analysis["bob_files"]:
                # Check if they're identical
                bolt_path = BOLT_DIR / bolt_file
                bob_path = BOB_DIR / bolt_file
                
                try:
                    bolt_content = bolt_path.read_text()
                    bob_content = bob_path.read_text()
                    
                    if bolt_content == bob_content:
                        analysis["already_migrated"].append(bolt_file)
                    else:
                        analysis["conflicts"].append({
                            "file": bolt_file,
                            "bolt_size": len(bolt_content),
                            "bob_size": len(bob_content)
                        })
                except Exception as e:
                    analysis["conflicts"].append({
                        "file": bolt_file,
                        "error": str(e)
                    })
        
        return analysis
    
    def create_migration_plan(self) -> Dict[str, any]:
        """Create comprehensive migration plan."""
        print("📋 Creating comprehensive migration plan...")
        
        # Migration mappings for key BOLT components
        self.migration_plan = {
            "phase_1_backup": {
                "description": "Create complete backup of current system",
                "operations": [
                    f"Create backup directory: {BACKUP_DIR}",
                    f"Backup BOLT directory: {BOLT_DIR} -> {BACKUP_DIR}/bolt_original",
                    f"Backup BOB directory: {BOB_DIR} -> {BACKUP_DIR}/bob_original", 
                    "Create rollback script"
                ]
            },
            
            "phase_2_directory_structure": {
                "description": "Ensure proper BOB directory structure",
                "operations": [
                    "Verify /bob/agents/ exists (already present)",
                    "Create /bob/integration/bolt/ for BOLT-specific integration",
                    "Create /bob/hardware/gpu/ for GPU acceleration",
                    "Create /bob/performance/bolt/ for BOLT performance components"
                ]
            },
            
            "phase_3_core_migration": {
                "description": "Migrate core BOLT components to BOB structure",
                "file_mappings": [
                    # Core integration (major component)
                    ("/bolt/core/integration.py", "/bob/integration/bolt/core_integration.py"),
                    ("/bolt/core/optimized_integration.py", "/bob/integration/bolt/optimized_integration.py"),
                    ("/bolt/core/ultra_fast_coordination.py", "/bob/core/ultra_fast_coordination.py"),
                    
                    # Hardware acceleration
                    ("/bolt/gpu_acceleration.py", "/bob/hardware/gpu/bolt_gpu_acceleration.py"),
                    ("/bolt/gpu_acceleration_optimized.py", "/bob/hardware/gpu/optimized_acceleration.py"),
                    ("/bolt/gpu_acceleration_final.py", "/bob/hardware/gpu/final_acceleration.py"),
                    ("/bolt/hardware_accelerated_faiss.py", "/bob/hardware/gpu/faiss_acceleration.py"),
                    
                    # Memory management
                    ("/bolt/memory_optimized_bolt.py", "/bob/hardware/memory_optimization.py"),
                    ("/bolt/unified_memory.py", "/bob/hardware/unified_memory.py"),
                    ("/bolt/optimized_memory_manager.py", "/bob/hardware/memory_manager.py"),
                    
                    # Performance components
                    ("/bolt/performance_benchmark.py", "/bob/performance/bolt/benchmarks.py"),
                    ("/bolt/m4_pro_integration.py", "/bob/performance/bolt/m4_pro_integration.py"),
                    ("/bolt/adaptive_concurrency.py", "/bob/performance/bolt/adaptive_concurrency.py"),
                ]
            },
            
            "phase_4_merge_operations": {
                "description": "Merge overlapping components", 
                "operations": [
                    {
                        "type": "merge_hardware",
                        "source": "/bolt/hardware/",
                        "target": "/bob/hardware/",
                        "strategy": "supplement_existing"
                    },
                    {
                        "type": "merge_error_handling", 
                        "source": "/bolt/error_handling/",
                        "target": "/bob/integration/error_handling/",
                        "strategy": "enhance_existing"
                    },
                    {
                        "type": "merge_utils",
                        "source": "/bolt/utils/",
                        "target": "/bob/utils/",
                        "strategy": "supplement_with_bolt_prefix"
                    }
                ]
            },
            
            "phase_5_configuration": {
                "description": "Update configuration systems",
                "operations": [
                    "Merge BOLT config into BOB unified config",
                    "Update import paths in all migrated files",
                    "Update CLI integration for bolt-specific commands",
                    "Configure startup sequence integration"
                ]
            },
            
            "phase_6_validation": {
                "description": "Comprehensive validation and testing",
                "tests": [
                    "8-agent orchestration functionality",
                    "1.5 tasks/second throughput validation", 
                    "Work-stealing algorithm verification",
                    "M4 Pro hardware optimization validation",
                    "Einstein integration compatibility",
                    "GPU acceleration functionality",
                    "Error handling and recovery systems",
                    "Memory management and optimization"
                ]
            },
            
            "phase_7_cleanup": {
                "description": "Clean up after successful migration",
                "operations": [
                    "Archive original /bolt/ directory",
                    "Update all external references to BOLT",
                    "Update documentation and README files",
                    "Create migration completion report"
                ]
            }
        }
        
        return self.migration_plan
    
    def generate_file_mappings(self) -> List[Tuple[Path, Path]]:
        """Generate detailed file mapping list."""
        print("🗂️  Generating detailed file mappings...")
        
        mappings = []
        
        # Core BOLT files that need migration
        bolt_core_files = [
            "core/integration.py",
            "core/optimized_integration.py", 
            "core/ultra_fast_coordination.py",
            "core/config.py",
            "core/system_info.py",
            "core/task_subdivision.py",
            "core/robust_tool_manager.py",
            "core/einstein_accelerator.py"
        ]
        
        for bolt_file in bolt_core_files:
            source = BOLT_DIR / bolt_file
            target = BOB_DIR / "integration" / "bolt" / Path(bolt_file).name
            if source.exists():
                mappings.append((source, target))
        
        # GPU acceleration files
        gpu_files = [
            "gpu_acceleration.py",
            "gpu_acceleration_optimized.py", 
            "gpu_acceleration_final.py",
            "hardware_accelerated_faiss.py",
            "metal_accelerated_search.py",
            "gpu_memory_optimizer.py"
        ]
        
        for gpu_file in gpu_files:
            source = BOLT_DIR / gpu_file
            target = BOB_DIR / "hardware" / "gpu" / gpu_file
            if source.exists():
                mappings.append((source, target))
        
        # Performance files  
        performance_files = [
            "performance_benchmark.py",
            "m4_pro_integration.py",
            "adaptive_concurrency.py",
            "thermal_monitor.py",
            "memory_optimization_integration.py"
        ]
        
        for perf_file in performance_files:
            source = BOLT_DIR / perf_file
            target = BOB_DIR / "performance" / "bolt" / perf_file
            if source.exists():
                mappings.append((source, target))
        
        self.file_mappings = mappings
        return mappings
    
    def create_dependency_updates(self) -> List[Dict[str, any]]:
        """Create dependency update plan."""
        print("🔗 Planning dependency updates...")
        
        updates = [
            {
                "type": "import_path_updates",
                "description": "Update all imports from bolt.* to bob.*",
                "patterns": [
                    ("from bolt.core", "from bob.integration.bolt"),
                    ("from bolt.agents", "from bob.agents"),
                    ("from bolt.hardware", "from bob.hardware"),
                    ("from bolt.error_handling", "from bob.integration.error_handling"),
                    ("import bolt.", "import bob.")
                ]
            },
            {
                "type": "configuration_integration", 
                "description": "Integrate BOLT configurations into BOB config system",
                "operations": [
                    "Merge bolt/config.yaml.example into bob/config/base.yaml",
                    "Update configuration loader to handle BOLT-specific settings",
                    "Preserve M4 Pro optimization settings"
                ]
            },
            {
                "type": "cli_integration",
                "description": "Integrate BOLT CLI commands into BOB CLI",
                "operations": [
                    "Move bolt_cli.py functionality to bob/cli/",
                    "Update command routing in bob/cli/main.py",
                    "Preserve bolt-specific command options"
                ]
            }
        ]
        
        self.dependency_updates = updates
        return updates
    
    def create_validation_plan(self) -> List[str]:
        """Create comprehensive validation plan."""
        print("✅ Creating validation plan...")
        
        validation_steps = [
            # Performance validation
            "python bob_performance_test.py --validate-throughput",
            "python test_8_agent_coordination.py --full-validation",
            "python validate_work_stealing.py --performance-test",
            
            # Hardware optimization validation
            "python test_m4_pro_optimization.py --benchmark",
            "python validate_gpu_acceleration.py --metal-test", 
            "python test_memory_optimization.py --pressure-test",
            
            # Integration validation
            "python test_einstein_bolt_integration.py --comprehensive",
            "python validate_error_handling.py --recovery-test",
            "python test_thermal_monitoring.py --stress-test",
            
            # System validation
            "python run_bolt_production_validation.py --full-suite",
            "python validate_migration_success.py --complete-check"
        ]
        
        self.validation_steps = validation_steps
        return validation_steps
    
    def create_rollback_script(self) -> str:
        """Create rollback script for emergency recovery."""
        rollback_script = f'''#!/bin/bash
# BOLT to BOB Migration Rollback Script
# Generated: {datetime.now().isoformat()}

set -e

echo "🚨 EMERGENCY ROLLBACK: Restoring BOLT system..."

# Stop all running processes
pkill -f "bob_" || true
pkill -f "bolt_cli" || true

# Restore from backup
if [ -d "{BACKUP_DIR}" ]; then
    echo "Restoring from backup: {BACKUP_DIR}"
    
    # Restore BOLT directory
    if [ -d "{BACKUP_DIR}/bolt_original" ]; then
        rm -rf "{BOLT_DIR}"
        cp -r "{BACKUP_DIR}/bolt_original" "{BOLT_DIR}"
        echo "✅ BOLT directory restored"
    fi
    
    # Restore BOB directory to pre-migration state
    if [ -d "{BACKUP_DIR}/bob_original" ]; then
        rm -rf "{BOB_DIR}"
        cp -r "{BACKUP_DIR}/bob_original" "{BOB_DIR}" 
        echo "✅ BOB directory restored"
    fi
    
    echo "🔄 Rollback complete. System restored to pre-migration state."
else
    echo "❌ Backup directory not found: {BACKUP_DIR}"
    echo "Manual recovery required."
    exit 1
fi
'''
        return rollback_script
    
    def print_migration_summary(self):
        """Print comprehensive migration summary."""
        print("\n" + "="*80)
        print("🚀 BOLT TO BOB MIGRATION PLAN SUMMARY")
        print("="*80)
        
        analysis = self.analyze_current_state()
        
        print(f"\n📊 Current State Analysis:")
        print(f"   BOLT files: {len(analysis['bolt_files'])}")
        print(f"   BOB files: {len(analysis['bob_files'])}")
        print(f"   Already migrated: {len(analysis['already_migrated'])}")
        print(f"   Conflicts detected: {len(analysis['conflicts'])}")
        
        if analysis['conflicts']:
            print(f"\n⚠️  Conflicts requiring attention:")
            for conflict in analysis['conflicts'][:5]:  # Show first 5
                print(f"   - {conflict}")
        
        print(f"\n📋 Migration Phases:")
        for phase, details in self.migration_plan.items():
            print(f"   {phase}: {details['description']}")
        
        print(f"\n🗂️  File Operations:")
        print(f"   Files to migrate: {len(self.file_mappings)}")
        print(f"   Merge operations: {len(self.merge_operations)}")
        print(f"   Config updates: {len(self.config_updates)}")
        
        print(f"\n✅ Validation Steps:")
        for i, step in enumerate(self.validation_steps[:5], 1):
            print(f"   {i}. {step}")
        if len(self.validation_steps) > 5:
            print(f"   ... and {len(self.validation_steps) - 5} more")
        
        print(f"\n🎯 Critical Success Factors:")
        print(f"   ✓ Preserve 1.5 tasks/second throughput")
        print(f"   ✓ Maintain M4 Pro hardware optimizations") 
        print(f"   ✓ Keep work-stealing algorithm functional")
        print(f"   ✓ Ensure 8-agent coordination works")
        print(f"   ✓ Preserve Einstein integration")
        print(f"   ✓ Maintain GPU acceleration")
        print(f"   ✓ Keep error handling systems")
        
        print(f"\n💾 Backup Strategy:")
        print(f"   Backup location: {BACKUP_DIR}")
        print(f"   Rollback script: rollback_bolt_migration.sh")
        print(f"   Recovery: Automatic restoration available")
        
        print(f"\n⚡ Execution Commands:")
        print(f"   Review plan: python bolt_to_bob_migration_plan.py")
        print(f"   Execute migration: python bolt_to_bob_migration_plan.py --execute")
        print(f"   Validate migration: python bolt_to_bob_migration_plan.py --validate")
        print(f"   Emergency rollback: ./rollback_bolt_migration.sh")
        
        print("\n" + "="*80)
        print("🔒 CRITICAL: This is a PLAN ONLY - No files modified yet!")
        print("Review all details above before executing migration.")
        print("="*80 + "\n")

def save_migration_plan(migration: BoltToBobMigration):
    """Save complete migration plan to JSON."""
    plan_file = BASE_DIR / "bolt_migration_plan.json"
    
    plan_data = {
        "timestamp": datetime.now().isoformat(),
        "migration_plan": migration.migration_plan,
        "file_mappings": [(str(src), str(dst)) for src, dst in migration.file_mappings],
        "dependency_updates": migration.dependency_updates,
        "validation_steps": migration.validation_steps,
        "backup_location": str(BACKUP_DIR),
        "rollback_script_location": str(BASE_DIR / "rollback_bolt_migration.sh")
    }
    
    with open(plan_file, 'w') as f:
        json.dump(plan_data, f, indent=2)
    
    print(f"📄 Migration plan saved to: {plan_file}")

def main():
    """Main migration planning function."""
    print("🚀 BOLT to BOB Migration Planning System")
    print("CRITICAL: This creates the plan but does NOT execute operations")
    
    migration = BoltToBobMigration()
    
    # Create comprehensive migration plan
    migration.create_migration_plan()
    migration.generate_file_mappings()
    migration.create_dependency_updates()
    migration.create_validation_plan()
    
    # Print summary
    migration.print_migration_summary()
    
    # Save plan for execution
    save_migration_plan(migration)
    
    # Create rollback script
    rollback_script = migration.create_rollback_script()
    rollback_path = BASE_DIR / "rollback_bolt_migration.sh"
    rollback_path.write_text(rollback_script)
    rollback_path.chmod(0o755)
    print(f"🔄 Rollback script created: {rollback_path}")
    
    print(f"\n✅ Migration planning complete!")
    print(f"📋 Review the plan above and migration files before proceeding.")
    print(f"⚡ Ready for execution review and approval.")

if __name__ == "__main__":
    main()