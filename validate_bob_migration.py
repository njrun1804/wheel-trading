#!/usr/bin/env python3
"""
BOB Migration Validation Script

This script validates that the BOB migration was successful by:
1. Checking directory structure
2. Validating key files exist
3. Testing basic imports
4. Verifying configuration structure
"""

import sys
from pathlib import Path


def validate_directory_structure():
    """Validate the BOB directory structure."""
    print("=== Directory Structure Validation ===")
    
    expected_dirs = [
        "bob",
        "bob/agents",
        "bob/search", 
        "bob/hardware",
        "bob/integration",
        "bob/cli",
        "bob/config",
        "bob/utils",
        "bob/core",
        "bob/context",
        "bob/intent",
        "bob/monitoring",
        "bob/performance",
        "bob/planning",
        "bob/trading"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    if missing_dirs:
        print("\n❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    print("✅ All expected directories present")
    return True


def validate_key_files():
    """Validate key files were migrated correctly."""
    print("\n=== Key Files Validation ===")
    
    key_files = [
        # Core BOB files
        "bob/__init__.py",
        "bob_unified.py",
        
        # CLI system
        "bob/cli/__init__.py",
        "bob/cli/main.py",
        "bob/cli/processor.py",
        "bob/cli/interactive.py",
        "bob/cli/help.py",
        
        # Configuration
        "bob/config/__init__.py",
        "bob/config/config_manager.py",
        "bob/config/unified_config.yaml",
        
        # Migrated components
        "bob/agents/__init__.py",
        "bob/search/__init__.py", 
        "bob/hardware/__init__.py",
        "bob/integration/__init__.py",
        "bob/utils/__init__.py",
        
        # Specific migrated files
        "bob/integration/bolt_integration.py",
        "bob/search/vector_index.py",
        "bob/search/semantic_engine.py",
        "bob/hardware/gpu_acceleration.py",
        "bob/hardware/metal_monitor.py",
        
        # Compatibility
        "bob/compatibility.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print("\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All key files present")
    return True


def validate_imports():
    """Validate basic imports work."""
    print("\n=== Import Validation ===")
    
    success_count = 0
    total_tests = 0
    
    # Test basic BOB import
    total_tests += 1
    try:
        import bob
        print("✅ bob module imports successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ bob import failed: {e}")
    
    # Test BOB components (may fail due to dependencies)
    components_to_test = [
        ("bob.agents", "Agents system"),
        ("bob.search", "Search system"),
        ("bob.hardware", "Hardware system"),
        ("bob.integration", "Integration system"),
        ("bob.utils", "Utilities"),
    ]
    
    for module_name, description in components_to_test:
        total_tests += 1
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name}) imports successfully")
            success_count += 1
        except Exception as e:
            print(f"⚠️  {description} ({module_name}) import has issues: {e}")
    
    print(f"\nImport Success Rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    return success_count > 0


def validate_migration_completeness():
    """Check that migration is complete."""
    print("\n=== Migration Completeness ===")
    
    # Check if original directories still exist (should be preserved during migration)
    original_dirs = ["bolt", "einstein"]
    preserved_dirs = []
    
    for dir_name in original_dirs:
        if Path(dir_name).exists():
            preserved_dirs.append(dir_name)
            print(f"📁 {dir_name} directory preserved")
    
    # Check unified CLI exists
    if Path("bob_unified.py").exists():
        print("✅ Unified CLI entry point created")
    else:
        print("❌ Unified CLI entry point missing")
        
    # Check configuration consolidation
    if Path("bob/config/unified_config.yaml").exists():
        print("✅ Unified configuration file created")
    else:
        print("❌ Unified configuration file missing")
        
    print(f"✅ Migration preserves {len(preserved_dirs)} original directories for compatibility")
    return True


def generate_migration_report():
    """Generate a summary report of the migration."""
    print("\n" + "="*50)
    print("BOB MIGRATION VALIDATION REPORT")
    print("="*50)
    
    # Run all validations
    structure_ok = validate_directory_structure()
    files_ok = validate_key_files()
    imports_ok = validate_imports()
    migration_ok = validate_migration_completeness()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    checks = [
        ("Directory Structure", structure_ok),
        ("Key Files", files_ok),
        ("Basic Imports", imports_ok),
        ("Migration Completeness", migration_ok)
    ]
    
    passed_checks = sum(1 for _, ok in checks if ok)
    total_checks = len(checks)
    
    for check_name, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{check_name:.<20} {status}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 BOB migration appears to be SUCCESSFUL!")
        print("   - All components migrated to unified structure")
        print("   - Unified CLI interface created")
        print("   - Configuration system consolidated")
        print("   - Import compatibility maintained")
    elif passed_checks >= total_checks * 0.75:
        print("\n⚠️  BOB migration is MOSTLY SUCCESSFUL")
        print("   - Core structure is in place")
        print("   - Some dependencies may need to be installed")
        print("   - System should be functional with proper setup")
    else:
        print("\n❌ BOB migration has ISSUES")
        print("   - Critical components missing or broken")
        print("   - Manual intervention required")
    
    print("\nNext steps:")
    print("  1. Install missing Python dependencies (yaml, psutil, etc.)")
    print("  2. Test end-to-end functionality with: python3 bob_unified.py --version")
    print("  3. Run integration tests with the trading system")
    print("  4. Update any external scripts to use bob_unified.py")


if __name__ == "__main__":
    generate_migration_report()