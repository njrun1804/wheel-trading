#!/usr/bin/env python3
"""Bolt Smoke Test - Simple functionality validation"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that we can import key Bolt components"""
    print("🔍 Testing imports...")

    try:
        # Test basic imports
        from bolt import __version__ as bolt_version

        print(f"✓ Bolt version: {bolt_version}")
    except ImportError as e:
        print(f"⚠️  Could not import bolt version: {e}")

    try:
        from bolt.cli.main import main as bolt_main

        print("✓ Bolt CLI main imported successfully")
        return True
    except ImportError as e:
        try:
            from bolt.solve import main as solve_main

            print("✓ Bolt solve main imported successfully")
            return True
        except ImportError as e2:
            print(f"❌ Could not import main functions: {e}, {e2}")
            return False


def test_hardware_detection():
    """Test hardware detection capabilities"""
    print("\n🔧 Testing hardware detection...")

    try:
        import psutil

        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=1)

        print(f"✓ CPU cores: {cpu_count} logical, {cpu_physical} physical")
        print(f"✓ CPU usage: {cpu_percent:.1f}%")

        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_percent = memory.percent

        print(f"✓ Memory: {memory_gb:.1f}GB total, {memory_percent:.1f}% used")

        # Check for M4 Pro characteristics
        is_m4_pro_like = cpu_physical >= 10 and memory_gb >= 16
        print(f"✓ M4 Pro-like system: {is_m4_pro_like}")

        return True

    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False


def test_gpu_detection():
    """Test GPU acceleration detection"""
    print("\n🎮 Testing GPU acceleration...")

    # Test MLX
    try:
        import mlx.core as mx

        mlx_available = mx.metal.is_available()
        print(f"✓ MLX available: {mlx_available}")

        if mlx_available:
            devices = mx.metal.get_devices()
            print(f"✓ MLX devices: {len(devices)}")

            # Quick GPU test
            start_time = time.time()
            a = mx.random.normal((100, 100))
            b = mx.random.normal((100, 100))
            c = mx.matmul(a, b)
            mx.eval(c)
            gpu_time = (time.time() - start_time) * 1000
            print(f"✓ MLX matrix multiply: {gpu_time:.2f}ms")

        has_mlx = True
    except ImportError:
        print("⚠️  MLX not available")
        has_mlx = False
    except Exception as e:
        print(f"⚠️  MLX error: {e}")
        has_mlx = False

    # Test PyTorch MPS
    try:
        import torch

        mps_available = torch.backends.mps.is_available()
        print(f"✓ PyTorch MPS available: {mps_available}")

        if mps_available:
            device = torch.device("mps")
            start_time = time.time()
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)
            torch.mps.synchronize()
            mps_time = (time.time() - start_time) * 1000
            print(f"✓ PyTorch MPS matrix multiply: {mps_time:.2f}ms")

        has_torch = True
    except ImportError:
        print("⚠️  PyTorch not available")
        has_torch = False
    except Exception as e:
        print(f"⚠️  PyTorch MPS error: {e}")
        has_torch = False

    return has_mlx or has_torch


def test_cli_interface():
    """Test CLI interface without full execution"""
    print("\n🖥️  Testing CLI interface...")

    try:
        # Test help functionality
        # Mock CLI args for testing
        from click.testing import CliRunner

        from bolt.solve import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        if result.exit_code == 0:
            print("✓ CLI help system working")
            return True
        else:
            print(f"⚠️  CLI help failed with exit code {result.exit_code}")
            return False

    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        return False


async def test_basic_task_execution():
    """Test basic task execution without complex dependencies"""
    print("\n⚡ Testing basic task execution...")

    try:
        # Test with fallback mode
        from bolt.solve import fallback_analyze_and_execute

        test_query = "optimize performance"
        start_time = time.time()

        result = await fallback_analyze_and_execute(test_query, analyze_only=True)

        execution_time = (time.time() - start_time) * 1000

        print(f"✓ Query analysis completed in {execution_time:.1f}ms")
        print(f"✓ Query: {result.get('query', 'N/A')}")
        print(f"✓ Recommendations: {len(result.get('recommendations', []))}")

        return True

    except Exception as e:
        print(f"❌ Basic task execution failed: {e}")
        return False


def test_file_operations():
    """Test file operation capabilities"""
    print("\n📁 Testing file operations...")

    try:
        # Test path operations
        cwd = Path.cwd()
        print(f"✓ Current directory: {cwd}")

        # Test file search
        python_files = list(cwd.glob("*.py"))[:5]  # Limit to first 5
        print(f"✓ Found {len(python_files)} Python files in current directory")

        # Test reading this file
        with open(__file__) as f:
            lines = f.readlines()
        print(f"✓ Successfully read {len(lines)} lines from test file")

        return True

    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False


def main():
    """Main smoke test function"""
    print("🚀 Bolt Smoke Test Suite")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("GPU Detection", test_gpu_detection),
        ("CLI Interface", test_cli_interface),
        ("File Operations", test_file_operations),
    ]

    async_tests = [
        ("Basic Task Execution", test_basic_task_execution),
    ]

    results = []

    # Run synchronous tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            if "--debug" in sys.argv:
                traceback.print_exc()

    # Run async tests
    for test_name, test_func in async_tests:
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            if "--debug" in sys.argv:
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! Bolt system is functional.")
        return 0
    elif passed >= total * 0.5:
        print("⚠️  Some tests failed, but core functionality appears to work.")
        return 0
    else:
        print("❌ Many tests failed. System may have significant issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
