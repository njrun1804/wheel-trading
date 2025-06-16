#!/usr/bin/env python3
"""
Final Validation Demo - Agent 12
Demonstrates working system capabilities after validation
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_core_functionality():
    """Demonstrate core system functionality that works."""
    print("🚀 FINAL VALIDATION DEMO")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("\n1. Testing Core Imports...")
    try:
        from unity_wheel.math.options import OptionsCalculator
        from unity_wheel.storage.storage import Storage
        from unity_wheel.utils.logging import get_logger
        print("   ✅ Core imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        
    # Test 2: Math calculations
    print("\n2. Testing Math Operations...")
    try:
        calc = OptionsCalculator()
        iv = calc.black_scholes_iv(
            option_price=5.0,
            stock_price=100.0,
            strike=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type='call'
        )
        print(f"   ✅ Options IV calculation: {iv:.4f}")
    except Exception as e:
        print(f"   ❌ Math test failed: {e}")
        
    # Test 3: Accelerated tools
    print("\n3. Testing Accelerated Tools...")
    try:
        from unity_wheel.accelerated_tools.ripgrep_turbo import get_ripgrep_turbo
        rg = get_ripgrep_turbo()
        print("   ✅ Ripgrep turbo available")
    except Exception as e:
        print(f"   ❌ Accelerated tools failed: {e}")
    
    # Test 4: Einstein system
    print("\n4. Testing Einstein Search...")
    try:
        from einstein.unified_index import UnifiedIndex
        print("   ✅ Einstein imports successful")
        # Note: Full initialization would take time, just test imports
    except Exception as e:
        print(f"   ❌ Einstein test failed: {e}")
        
    # Test 5: Bolt system  
    print("\n5. Testing Bolt System...")
    try:
        from bolt.solve import BoltSolver
        print("   ✅ Bolt imports successful")
    except Exception as e:
        print(f"   ❌ Bolt test failed: {e}")

def demo_performance():
    """Demonstrate performance capabilities."""
    print("\n🏎️ PERFORMANCE DEMONSTRATION")
    print("=" * 50)
    
    # Simple performance test
    import numpy as np
    
    # Matrix operations
    start = time.time()
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    duration = time.time() - start
    print(f"   ✅ Matrix multiplication (1000x1000): {duration:.3f}s")
    
    # File operations
    start = time.time()
    test_file = Path("temp_test.txt")
    for i in range(1000):
        with open(test_file, 'w') as f:
            f.write(f"test {i}")
    test_file.unlink()
    duration = time.time() - start
    print(f"   ✅ File operations (1000 writes): {duration:.3f}s")

def demo_hardware_capabilities():
    """Demonstrate hardware acceleration capabilities."""
    print("\n⚡ HARDWARE ACCELERATION STATUS")
    print("=" * 50)
    
    # Check MLX
    try:
        import mlx.core as mx
        print("   ✅ MLX Metal acceleration available")
        
        # Simple MLX test
        x = mx.random.normal((100, 100))
        y = mx.random.normal((100, 100))
        z = mx.matmul(x, y)
        mx.eval(z)
        print("   ✅ MLX computation successful")
    except Exception as e:
        print(f"   ❌ MLX test failed: {e}")
    
    # Check PyTorch MPS
    try:
        import torch
        if torch.backends.mps.is_available():
            print("   ✅ PyTorch MPS acceleration available")
            
            # Simple MPS test
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print("   ✅ PyTorch MPS computation successful")
    except Exception as e:
        print(f"   ❌ PyTorch MPS test failed: {e}")

def show_system_status():
    """Show current system status."""
    print("\n📊 SYSTEM STATUS")
    print("=" * 50)
    
    import psutil
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"   CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"   Memory: {memory.percent:.1f}% ({memory.available // (1024**3):.1f}GB available)")
    
    # Load average
    try:
        load = psutil.getloadavg()
        print(f"   Load: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}")
    except:
        print("   Load: Not available on this platform")

def main():
    """Run complete validation demo."""
    print("🎯 WHEEL TRADING SYSTEM - FINAL VALIDATION")
    print("🤖 Agent 12: Final Validation and System Hardening")
    print("📅 Date: 2025-06-16")
    print()
    
    try:
        demo_core_functionality()
        demo_performance()
        demo_hardware_capabilities()
        show_system_status()
        
        print("\n" + "=" * 50)
        print("✅ VALIDATION DEMO COMPLETE")
        print("📋 See FINAL_PRODUCTION_READINESS_ASSESSMENT.md for full report")
        print("🎯 Overall Status: PARTIAL READINESS - Requires attention to critical issues")
        print("💡 Recommendation: Address auth and database issues before production")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()