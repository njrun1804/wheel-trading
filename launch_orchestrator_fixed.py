#!/usr/bin/env python3
"""Fixed orchestrator launcher with all imports working."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Configure for optimal performance
os.environ["USE_MCP_SERVERS"] = "0"  # Direct I/O only to avoid MCP issues
os.environ["USE_GPU_ACCELERATION"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    """Launch orchestrator with hardware acceleration."""

    # Import the fixed orchestrator
    from unity_wheel.orchestrator.orchestrator_consolidated import (
        ConsolidatedOrchestrator,
        StrategyType,
    )

    # Get command
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        print("🚀 Unity Wheel Orchestrator - Hardware Accelerated")
        print("=" * 50)
        print("\nUsage: ./launch_orchestrator_fixed.py '<command>'")
        print("\nExamples:")
        print("  ./launch_orchestrator_fixed.py 'analyze trading strategies'")
        print("  ./launch_orchestrator_fixed.py 'optimize wheel strategy code'")
        print("  ./launch_orchestrator_fixed.py 'list all Python files'")
        return

    print("\n🚀 Launching orchestrator...")
    print(f"📋 Command: {command}")
    print("-" * 70)

    # Create orchestrator
    orchestrator = ConsolidatedOrchestrator(".")

    # Initialize
    start_init = time.time()
    await orchestrator.initialize()
    print(f"✅ Initialized in {time.time() - start_init:.2f}s")

    # Auto-select strategy based on command
    if "optimize" in command.lower() or "performance" in command.lower():
        strategy = StrategyType.GPU_ACCELERATED
    elif "trading" in command.lower():
        strategy = StrategyType.TRADING_OPTIMIZED
    elif "quick" in command.lower() or "list" in command.lower():
        strategy = StrategyType.FAST
    else:
        strategy = StrategyType.ENHANCED

    print(f"📊 Using strategy: {strategy.value}")

    # Execute
    start_exec = time.time()
    try:
        result = await orchestrator.execute(command, strategy)

        # Display results
        print(f"\n✅ Execution completed in {time.time() - start_exec:.2f}s")

        # Show phase results
        if "phases" in result:
            print("\n📈 Phase Results:")
            for phase_name, phase_data in result["phases"].items():
                if isinstance(phase_data, dict):
                    print(f"\n{phase_name.upper()}:")
                    for key, value in phase_data.items():
                        if key != "duration_ms":
                            print(f"  • {key}: {value}")
                    if "duration_ms" in phase_data:
                        print(f"  ⏱️  Duration: {phase_data['duration_ms']:.1f}ms")

        # Show performance
        if "performance" in result:
            perf = result["performance"]
            print("\n⚡ Performance:")
            print(f"  • Total time: {perf.get('total_time_ms', 0):.1f}ms")
            print(f"  • Success: {perf.get('success', False)}")
            if "memory_peak_mb" in perf:
                print(f"  • Memory peak: {perf['memory_peak_mb']:.1f}MB")

        # Show any errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback

        traceback.print_exc()

    # Shutdown
    await orchestrator.shutdown()

    # Hardware status
    print("\n" + "-" * 70)
    print("💻 Hardware Status:")

    try:
        import psutil

        print(f"  • CPU cores: {psutil.cpu_count()}")
        print(f"  • CPU usage: {psutil.cpu_percent(interval=0.5):.1f}%")

        mem = psutil.virtual_memory()
        print(f"  • Memory: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB")
    except (ImportError, AttributeError, OSError) as e:
        print(f"  • System info: Unable to get system status ({e})")

    try:
        import torch

        if torch.backends.mps.is_available():
            print("  • GPU: Metal Performance Shaders ✅")
        else:
            print("  • GPU: Not available ❌")
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"  • GPU: PyTorch not available ({e})")

    print("\n✨ Done!")


if __name__ == "__main__":
    asyncio.run(main())
