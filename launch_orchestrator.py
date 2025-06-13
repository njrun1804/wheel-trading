#!/usr/bin/env python3
"""Launch orchestrator with MCP servers and hardware acceleration."""

import os
import sys
import asyncio
from pathlib import Path

# Force hardware acceleration mode
os.environ["USE_MCP_SERVERS"] = "hybrid"  # Use both MCP intelligence + hardware speed
os.environ["USE_GPU_ACCELERATION"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.orchestrator.production_orchestrator import create_production_orchestrator


async def main():
    """Launch orchestrator with full hardware acceleration."""
    
    # Get command from args or prompt
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        print("Enter command (or 'exit' to quit):")
        command = input("> ").strip()
        if command.lower() == 'exit':
            return
    
    print(f"\n🚀 Launching orchestrator with hardware acceleration...")
    print(f"📋 Command: {command}")
    print("-" * 70)
    
    # Create orchestrator with all features enabled
    orchestrator = await create_production_orchestrator(".")
    
    # Execute with hardware acceleration
    result = await orchestrator.execute(command)
    
    # Display results
    print(f"\n✅ Strategy used: {result.get('strategy', 'unknown')}")
    
    if "execution_metrics" in result:
        metrics = result["execution_metrics"]
        print(f"\n⚡ Performance:")
        print(f"  Duration: {metrics.get('duration_ms', 0):.1f}ms")
        print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"  Memory: {metrics.get('memory_mb', 0):.1f}MB")
        if metrics.get('gpu_used'):
            print(f"  GPU: Active")
    
    # Shutdown
    await orchestrator.shutdown()
    print("\n✨ Done!")


if __name__ == "__main__":
    asyncio.run(main())