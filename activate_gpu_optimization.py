#!/usr/bin/env python3
"""
GPU Optimization Activation Script

Quick script to activate GPU initialization optimizations and measure performance.
Run this to apply the <1.0s GPU initialization target optimization.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def activate_optimizations():
    """Activate GPU initialization optimizations."""
    print("🚀 Activating GPU Initialization Optimizations")
    print("Target: Reduce initialization time to <1.0s")
    print("=" * 50)
    
    start_time = time.perf_counter()
    
    try:
        # Step 1: Apply integration optimizations
        print("\n1️⃣ Applying integration optimizations...")
        from src.unity_wheel.gpu.gpu_init_integration import apply_gpu_optimizations, print_optimization_report
        
        replacements = apply_gpu_optimizations()
        print(f"   ✅ Optimized {len(replacements)} components")
        
        # Step 2: Initialize optimized GPU system
        print("\n2️⃣ Initializing optimized GPU system...")
        from src.unity_wheel.gpu.optimized_gpu_init import initialize_gpu_optimized
        
        init_stats = await initialize_gpu_optimized()
        print(f"   ✅ GPU initialized in {init_stats.total_time_ms:.1f}ms")
        
        # Step 3: Warm up components
        print("\n3️⃣ Warming up GPU components...")
        from src.unity_wheel.gpu.lazy_gpu_loader import warmup_gpu_components
        
        warmup_gpu_components()
        print("   ✅ Component warmup started")
        
        # Step 4: Test performance
        print("\n4️⃣ Testing performance...")
        
        # Multiple initialization tests
        test_times = []
        for i in range(3):
            test_start = time.perf_counter()
            test_stats = await initialize_gpu_optimized(force_reinit=(i == 0))
            test_time = (time.perf_counter() - test_start) * 1000
            test_times.append(test_stats.total_time_ms)
            
            cache_status = "cached" if test_stats.was_cached else "fresh"
            print(f"   Test {i+1}: {test_stats.total_time_ms:.1f}ms ({cache_status})")
        
        # Performance summary
        avg_time = sum(test_times) / len(test_times)
        min_time = min(test_times)
        target_achieved = avg_time < 1000
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average: {avg_time:.1f}ms")
        print(f"   Best:    {min_time:.1f}ms")
        print(f"   Target:  {'✅ ACHIEVED' if target_achieved else '❌ MISSED'} (<1000ms)")
        
        # Step 5: Show detailed report
        print("\n5️⃣ Detailed optimization report:")
        print_optimization_report()
        
        # Performance analysis
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"\n⏱️  Total activation time: {total_time:.1f}ms")
        
        if target_achieved:
            print("\n🎉 SUCCESS: GPU optimization target achieved!")
            print("   GPU initialization is now optimized for <1.0s performance")
            return 0
        else:
            print("\n⚠️  PARTIAL SUCCESS: Optimization applied but target not fully achieved")
            print(f"   Current performance: {avg_time:.1f}ms (target: <1000ms)")
            print("   Consider running the full validation test for more details")
            return 1
            
    except Exception as e:
        print(f"\n❌ FAILED: GPU optimization activation failed")
        print(f"   Error: {e}")
        logger.error(f"Activation failed: {e}", exc_info=True)
        return 2


def quick_test():
    """Quick synchronous test of optimizations."""
    print("🔧 Quick GPU Optimization Test")
    print("=" * 30)
    
    try:
        # Test basic imports
        print("Testing imports...")
        from src.unity_wheel.gpu.optimized_gpu_init import get_optimized_gpu_initializer
        from src.unity_wheel.gpu.lazy_gpu_loader import is_gpu_ready, get_gpu_component_stats
        
        print("✅ Core modules imported successfully")
        
        # Test lazy loading
        print("Testing lazy loading...")
        stats = get_gpu_component_stats()
        print(f"✅ Component stats: {len(stats)} components tracked")
        
        # Test initializer
        print("Testing initializer...")
        initializer = get_optimized_gpu_initializer()
        print(f"✅ Initializer created")
        
        print("\n✅ Quick test passed - ready for full optimization")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def show_usage():
    """Show usage information."""
    print("GPU Optimization Activation Script")
    print("=" * 35)
    print()
    print("Usage:")
    print("  python activate_gpu_optimization.py [command]")
    print()
    print("Commands:")
    print("  activate    - Apply GPU optimizations and test performance (default)")
    print("  test        - Quick test of optimization components")
    print("  validate    - Run full validation suite")
    print("  help        - Show this help message")
    print()
    print("Examples:")
    print("  python activate_gpu_optimization.py")
    print("  python activate_gpu_optimization.py test")
    print("  python activate_gpu_optimization.py validate")


def main():
    """Main function."""
    command = sys.argv[1] if len(sys.argv) > 1 else "activate"
    
    if command == "help":
        show_usage()
        return 0
    elif command == "test":
        success = quick_test()
        return 0 if success else 1
    elif command == "validate":
        print("Running full validation suite...")
        import subprocess
        result = subprocess.run([sys.executable, "test_gpu_optimization.py"], 
                              capture_output=False)
        return result.returncode
    elif command == "activate":
        return asyncio.run(activate_optimizations())
    else:
        print(f"Unknown command: {command}")
        show_usage()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)