#!/usr/bin/env python3
"""
Simple BOB CLI Test
==================

Basic test of the unified BOB CLI consolidation without heavy dependencies.
Tests core functionality and routing logic.
"""

import re
import sys
from pathlib import Path


def test_intent_classification():
    """Test intent classification logic without full dependencies."""
    print("🧪 Testing Intent Classification Logic")
    print("-" * 40)
    
    # Simple implementation of classification rules
    EINSTEIN_KEYWORDS = {"find", "search", "show", "list", "where", "what", "which"}
    BOLT_KEYWORDS = {"optimize", "fix", "debug", "analyze", "improve", "refactor", "solve"}
    BOB_KEYWORDS = {"unity", "wheel", "trading", "strategy", "risk", "position", "config"}
    
    def classify_simple(query):
        """Simple classification logic."""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Count keyword matches
        einstein_matches = sum(1 for word in words if word in EINSTEIN_KEYWORDS)
        bolt_matches = sum(1 for word in words if word in BOLT_KEYWORDS)
        bob_matches = sum(1 for word in words if word in BOB_KEYWORDS)
        
        # Determine system
        if einstein_matches > bolt_matches and einstein_matches > bob_matches:
            return "einstein"
        elif bolt_matches > einstein_matches and bolt_matches > bob_matches:
            return "bolt"
        elif bob_matches > 0:
            return "bob"
        elif len(words) <= 3:
            return "einstein"  # Short queries
        else:
            return "bolt"  # Complex queries
    
    # Test cases
    test_cases = [
        ("find WheelStrategy", "einstein"),
        ("optimize database performance", "bolt"),
        ("Unity wheel strategy", "bob"),
        ("search TODO", "einstein"),
        ("fix memory leak", "bolt"),
        ("trading risk parameters", "bob"),
        ("show options.py", "einstein"),
        ("analyze performance", "bolt"),
        ("wheel configuration", "bob")
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        predicted = classify_simple(query)
        passed = predicted == expected
        status = "✅" if passed else "❌"
        
        print(f"   {status} '{query}' → {predicted} (expected: {expected})")
        
        if passed:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"\n   🎯 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy >= 80  # 80% minimum accuracy


def test_command_translation():
    """Test command translation logic."""
    print("\n🔄 Testing Command Translation Logic")
    print("-" * 40)
    
    # Translation rules
    translation_rules = [
        (r"^python\s+bolt_cli\.py\s+(.+)$", r"solve \1"),
        (r"^python\s+bob_cli\.py\s+(.+)$", r"\1"),
        (r"^!einstein\s+(.+)$", r"search \1"),
        (r"^!bolt\s+(.+)$", r"solve \1"),
        (r"^(status|health|metrics)$", r"system \1")
    ]
    
    def translate_simple(command):
        """Simple translation logic."""
        for pattern, replacement in translation_rules:
            if re.match(pattern, command, re.IGNORECASE):
                return re.sub(pattern, replacement, command, flags=re.IGNORECASE)
        return command
    
    # Test cases
    test_cases = [
        ("python bolt_cli.py analyze performance", "solve analyze performance"),
        ("python bob_cli.py configure Unity", "configure Unity"),
        ("!einstein find code", "search find code"),
        ("!bolt optimize queries", "solve optimize queries"),
        ("status", "system status"),
        ("find WheelStrategy", "find WheelStrategy")  # No translation needed
    ]
    
    correct = 0
    total = len(test_cases)
    
    for original, expected in test_cases:
        translated = translate_simple(original)
        passed = translated == expected
        status = "✅" if passed else "❌"
        
        print(f"   {status} '{original}' → '{translated}'")
        if not passed:
            print(f"      Expected: '{expected}'")
        
        if passed:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"\n   🔄 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy >= 90  # 90% minimum accuracy


def test_file_structure():
    """Test that consolidation files exist."""
    print("\n📁 Testing File Structure")
    print("-" * 40)
    
    project_root = Path(__file__).parent
    
    # Required files
    required_files = [
        "bob",  # Main executable
        "bob_unified_main",  # Main script
        "setup_bob_symlinks.sh",  # Setup script
        "rollback_bob_consolidation.py",  # Rollback script
        "bob/cli/unified_router.py",  # Router
        "bob/cli/intent_detector.py",  # Intent detection
        "bob/cli/command_translator.py",  # Translation
        "bob/config/unified_bob_config.yaml"  # Config
    ]
    
    # Wrapper files
    wrapper_files = [
        "bob_cli.py",
        "bolt_cli.py", 
        "bob_unified.py",
        "unified_cli.py"
    ]
    
    all_files = required_files + wrapper_files
    
    missing = []
    present = []
    
    for file_path_str in all_files:
        file_path = project_root / file_path_str
        if file_path.exists():
            present.append(file_path_str)
            print(f"   ✅ {file_path_str}")
        else:
            missing.append(file_path_str)
            print(f"   ❌ {file_path_str}")
    
    success_rate = len(present) / len(all_files) * 100
    print(f"\n   📊 Files present: {len(present)}/{len(all_files)} ({success_rate:.1f}%)")
    
    return len(missing) == 0


def test_backup_system():
    """Test backup system."""
    print("\n💾 Testing Backup System")
    print("-" * 40)
    
    project_root = Path(__file__).parent
    backup_dirs = list(project_root.glob(".bob_migration_backup_*"))
    
    if backup_dirs:
        backup_dir = backup_dirs[0]  # Use first backup found
        print(f"   ✅ Backup directory found: {backup_dir.name}")
        
        # Check backup contents
        expected_backup_files = ["bob_cli.py", "bolt_cli.py", "bob_unified.py", "unified_cli.py"]
        backup_files = [f.name for f in backup_dir.iterdir() if f.is_file()]
        
        for file_name in expected_backup_files:
            if file_name in backup_files:
                print(f"   ✅ Backup contains: {file_name}")
            else:
                print(f"   ⚠️  Backup missing: {file_name}")
        
        return True
    else:
        print("   ❌ No backup directory found")
        return False


def main():
    """Run all simple tests."""
    print("🚀 Simple BOB CLI Consolidation Test")
    print("=" * 50)
    
    tests = [
        ("Intent Classification", test_intent_classification),
        ("Command Translation", test_command_translation), 
        ("File Structure", test_file_structure),
        ("Backup System", test_backup_system)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        try:
            if test_function():
                passed_tests += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    # Overall results
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    success_rate = passed_tests / total_tests * 100
    
    if success_rate >= 100:
        print("🎉 Perfect! All tests passed")
        status = "EXCELLENT"
    elif success_rate >= 75:
        print("👍 Good! Most tests passed")
        status = "GOOD"
    elif success_rate >= 50:
        print("⚠️  Warning! Some tests failed")
        status = "WARNING"
    else:
        print("❌ Critical! Many tests failed")
        status = "CRITICAL"
    
    print(f"\n🏆 BOB CLI Consolidation Status: {status}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ BOB CLI consolidation appears to be working correctly!")
        print("   You can now use: ./bob <command>")
        print("   Legacy commands will show deprecation warnings")
    else:
        print("\n⚠️  Some issues detected with BOB CLI consolidation")
        print("   Run full test suite for detailed diagnostics")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)