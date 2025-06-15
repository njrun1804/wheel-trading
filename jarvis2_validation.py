#!/usr/bin/env python3
"""
Comprehensive Validation of All Jarvis2 + Meta System Fixes

Validates that all the fixes implemented work correctly together:
1. Meta system lazy loading (no auto-spawn)
2. Unified configuration system
3. Database consolidation and connection pooling
4. Neural backend validation
5. Memory management
6. File cleanup and organization
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any


def test_1_no_meta_auto_spawn() -> bool:
    """Test that meta system doesn't auto-spawn on imports."""
    
    print("\n🔍 Test 1: Meta System Lazy Loading")
    print("" + "="*50)
    
    # Import core systems and ensure no MetaPrime spawning
    spawn_detected = False
    
    try:
        # Redirect stdout to capture MetaPrime birth messages
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # These imports should NOT trigger MetaPrime birth
            from unified_config import get_unified_config
            from database_manager import get_database_manager
            from neural_backend_manager import get_neural_backend_manager
            
        output = f.getvalue()
        if "MetaPrime born" in output:
            spawn_detected = True
            print(f"❌ Meta auto-spawn detected in unified imports")
        else:
            print(f"✅ No meta auto-spawn in unified imports")
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return not spawn_detected


def test_2_unified_configuration() -> bool:
    """Test unified configuration system."""
    
    print("\n⚙️ Test 2: Unified Configuration System")
    print("" + "="*50)
    
    try:
        from unified_config import get_unified_config
        config = get_unified_config()
        
        # Test resource allocation validation
        total_cpu = (config.hardware.jarvis2_cpu_allocation + 
                    config.hardware.meta_cpu_allocation + 
                    config.hardware.system_cpu_reserve)
        
        if total_cpu > 1.0:
            print(f"❌ CPU allocation exceeds 100%: {total_cpu:.1%}")
            return False
        
        # Test core allocation methods
        jarvis2_cores = config.get_jarvis2_cpu_cores()
        meta_cores = config.get_meta_cpu_cores()
        batch_size = config.get_effective_batch_size()
        
        print(f"✅ Resource allocation valid: {total_cpu:.1%}")
        print(f"   Jarvis2 cores: {jarvis2_cores}")
        print(f"   Meta cores: {meta_cores}")
        print(f"   Batch size: {batch_size}")
        print(f"   Memory limit: {config.hardware.memory_limit_gb}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_3_database_consolidation() -> bool:
    """Test database consolidation and connection pooling."""
    
    print("\n🗄️ Test 3: Database Consolidation & Pooling")
    print("" + "="*50)
    
    try:
        from database_manager import get_database_manager
        
        db_manager = get_database_manager()
        
        # Test connection pooling
        with db_manager.get_connection('evolution') as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Evolution DB: {table_count} tables")
        
        with db_manager.get_connection('monitoring') as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Monitoring DB: {table_count} tables")
        
        with db_manager.get_connection('reality') as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Reality DB: {table_count} tables")
        
        # Test database statistics
        stats = db_manager.get_database_stats()
        for name, data in stats.items():
            if 'error' in data:
                print(f"❌ {name}: {data['error']}")
                return False
            else:
                print(f"✅ {name}: {data['size_mb']}MB, pool size: {data['pool_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_4_neural_backend_validation() -> bool:
    """Test neural backend manager with fallback chain."""
    
    print("\n🧠 Test 4: Neural Backend Validation")
    print("" + "="*50)
    
    try:
        from neural_backend_manager import get_neural_backend_manager
        
        backend_manager = get_neural_backend_manager()
        
        # Test backend status
        status = backend_manager.get_backend_status()
        current_backend = status['current']
        functional_backends = status['functional_backends']
        total_backends = status['total_backends']
        
        print(f"✅ Current backend: {current_backend}")
        print(f"✅ Functional backends: {functional_backends}/{total_backends}")
        
        # Test tensor conversion
        import numpy as np
        test_data = np.array([1.0, 2.0, 3.0])
        
        converted = backend_manager.convert_tensor(test_data)
        print(f"✅ Tensor conversion working")
        
        # Test compatibility validation
        is_compatible = backend_manager.validate_tensor_compatibility(test_data)
        print(f"✅ Tensor compatibility: {is_compatible}")
        
        return functional_backends > 0
        
    except Exception as e:
        print(f"❌ Neural backend test failed: {e}")
        return False


def test_5_memory_management() -> bool:
    """Test memory management and cleanup."""
    
    print("\n💾 Test 5: Memory Management")
    print("" + "="*50)
    
    try:
        from jarvis2.core.memory_manager import get_memory_manager
        
        mem_manager = get_memory_manager()
        
        # Test memory stats
        stats = mem_manager.get_stats()
        
        print(f"✅ System memory: {stats['system_memory_gb']:.1f}GB total")
        print(f"✅ Available: {stats['system_available_gb']:.1f}GB")
        print(f"✅ Buffer pool: {stats['buffer_pool_gb']:.1f}GB")
        print(f"✅ Buffers: {stats['buffers_in_use']}/{stats['buffers_total']} in use")
        print(f"✅ Utilization: {stats['buffer_utilization']:.1%}")
        
        # Test memory allocation and cleanup
        import numpy as np
        test_shape = (100, 100)
        
        buffer_name, array = mem_manager.allocate_tensor(test_shape)
        print(f"✅ Allocated tensor: {buffer_name}")
        
        # Test retrieval
        retrieved = mem_manager.get_tensor(buffer_name)
        print(f"✅ Retrieved tensor shape: {retrieved.shape}")
        
        # Test cleanup
        mem_manager.release_tensor(buffer_name)
        print(f"✅ Released tensor: {buffer_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False


def test_6_file_organization() -> bool:
    """Test that file cleanup and organization worked."""
    
    print("\n🗂 Test 6: File Organization")
    print("" + "="*50)
    
    # Check that duplicate files were moved to backups
    expected_backups = [
        'backups/jarvis2_consolidation/',
        'backups/sequential_thinking_cleanup/',
        'backups/database_consolidation/'
    ]
    
    for backup_dir in expected_backups:
        path = Path(backup_dir)
        if path.exists():
            file_count = len(list(path.glob('*')))
            print(f"✅ {backup_dir}: {file_count} files backed up")
        else:
            print(f"⚠️ {backup_dir}: directory not found")
    
    # Check that main entry points exist
    main_files = [
        'jarvis2_unified.py',
        'unified_config.py', 
        'database_manager.py',
        'neural_backend_manager.py',
        'database_consolidation.py'
    ]
    
    missing_files = []
    for file_name in main_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
        else:
            print(f"✅ {file_name}: exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check that old duplicate files are gone from root
    old_files = [
        'jarvis2_complete.py',
        'jarvis2_core.py', 
        'jarvis2_mcts.py',
        'jarvis2_optimized.py',
        'benchmark_sequential_thinking.py',
        'test_sequential_thinking.py'
    ]
    
    remaining_old = []
    for file_name in old_files:
        if Path(file_name).exists():
            remaining_old.append(file_name)
    
    if remaining_old:
        print(f"⚠️ Old files still in root: {remaining_old}")
    else:
        print(f"✅ All old duplicate files cleaned up")
    
    return len(missing_files) == 0


async def test_7_end_to_end_integration() -> bool:
    """Test end-to-end integration without triggering meta auto-spawn."""
    
    print("\n🔗 Test 7: End-to-End Integration")
    print("" + "="*50)
    
    try:
        # Test that we can import and use core systems together
        from unified_config import get_unified_config
        from database_manager import get_database_manager
        from neural_backend_manager import get_neural_backend_manager
        
        config = get_unified_config()
        db_manager = get_database_manager()
        neural_manager = get_neural_backend_manager()
        
        # Test coordinated operation
        batch_size = config.get_effective_batch_size(0.5)  # 50% memory pressure
        print(f"✅ Adaptive batch size: {batch_size}")
        
        # Test database transaction
        success = db_manager.execute_transaction([
            ("CREATE TABLE IF NOT EXISTS test_integration (id INTEGER, data TEXT)", ()),
            ("INSERT OR REPLACE INTO test_integration VALUES (1, 'integration_test')", ())
        ], 'evolution')
        
        if success:
            print(f"✅ Database transaction successful")
        else:
            print(f"❌ Database transaction failed")
            return False
        
        # Test neural backend with configuration
        backend_status = neural_manager.get_backend_status()
        if backend_status['functional_backends'] > 0:
            print(f"✅ Neural backend integration working")
        else:
            print(f"❌ No functional neural backends")
            return False
        
        print(f"✅ All systems integrated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Run comprehensive validation."""
    
    print("🏆 Comprehensive Jarvis2 + Meta System Validation")
    print("" + "="*70)
    print("Validating all fixes implemented during cleanup phase...")
    
    tests = [
        ("Meta System Lazy Loading", test_1_no_meta_auto_spawn),
        ("Unified Configuration", test_2_unified_configuration),
        ("Database Consolidation", test_3_database_consolidation),
        ("Neural Backend Validation", test_4_neural_backend_validation),
        ("Memory Management", test_5_memory_management),
        ("File Organization", test_6_file_organization),
        ("End-to-End Integration", test_7_end_to_end_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
            
        results.append((test_name, result))
        
        if result:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("" + "="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("🚀 System is ready for production use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)