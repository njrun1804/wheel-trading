#!/usr/bin/env python3
"""
Simple Jarvis2 Test - Test basic functionality without complex imports
"""

import sys

def test_unified_systems():
    """Test unified systems without triggering MetaPrime."""
    
    print("🧪 Testing Unified Systems...")
    
    # Test 1: Unified Configuration
    try:
        from unified_config import get_unified_config
        config = get_unified_config()
        print(f"✅ Unified Config: {config.get_jarvis2_cpu_cores()} cores, {config.hardware.memory_limit_gb}GB")
    except Exception as e:
        print(f"❌ Unified Config: {e}")
        return False
    
    # Test 2: Database Manager
    try:
        from database_manager import get_database_manager
        db_manager = get_database_manager()
        stats = db_manager.get_database_stats()
        print(f"✅ Database Manager: {len(stats)} databases")
        for name, data in stats.items():
            if 'error' not in data:
                print(f"   {name}: {data['size_mb']}MB")
    except Exception as e:
        print(f"❌ Database Manager: {e}")
        return False
    
    # Test 3: Neural Backend Manager
    try:
        from neural_backend_manager import get_neural_backend_manager
        backend_manager = get_neural_backend_manager()
        status = backend_manager.get_backend_status()
        print(f"✅ Neural Backend: {status['current']} ({status['functional_backends']}/{status['total_backends']})")
    except Exception as e:
        print(f"❌ Neural Backend: {e}")
        return False
    
    # Test 4: Memory manager (from jarvis2)
    try:
        from jarvis2.core.memory_manager import get_memory_manager
        mem_manager = get_memory_manager()
        stats = mem_manager.get_stats()
        print(f"✅ Memory Manager: {stats['buffers_total']} buffers, {stats['buffer_utilization']:.1%} used")
    except Exception as e:
        print(f"❌ Memory Manager: {e}")
        return False
    
    print("\n🎉 All unified systems working correctly!")
    return True

def test_database_consolidation():
    """Test that database consolidation worked."""
    
    print("\n🗄️ Testing Database Consolidation...")
    
    import os
    from pathlib import Path
    
    # Check that old databases are gone
    old_dbs = [
        'jarvis2_audit.db',
        'jarvis2_reaudit.db', 
        'jarvis2_strategy.db',
        'meta_daemon_continuous.db'
    ]
    
    found_old = []
    for db in old_dbs:
        if Path(db).exists():
            found_old.append(db)
    
    if found_old:
        print(f"⚠️ Found old databases: {found_old}")
    else:
        print("✅ Old databases properly cleaned up")
    
    # Check that new databases exist and are functional
    new_dbs = [
        'meta_evolution.db',
        'meta_monitoring.db',
        'meta_reality_learning.db'
    ]
    
    existing_new = []
    for db in new_dbs:
        if Path(db).exists():
            existing_new.append(db)
            # Quick integrity check
            import sqlite3
            try:
                conn = sqlite3.connect(db)
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                print(f"✅ {db}: {len(tables)} tables")
            except Exception as e:
                print(f"❌ {db}: {e}")
    
    if len(existing_new) == len(new_dbs):
        print("✅ All consolidated databases present and functional")
        return True
    else:
        print(f"❌ Missing databases: {set(new_dbs) - set(existing_new)}")
        return False

if __name__ == "__main__":
    print("🚀 Comprehensive System Test")
    print("=" * 50)
    
    success = test_unified_systems()
    if success:
        success = test_database_consolidation()
    
    if success:
        print("\n🎯 All tests passed! System is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)
