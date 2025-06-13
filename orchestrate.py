#!/usr/bin/env python3
"""Main entry point for the orchestrator with enhanced M4 Pro optimizations."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.orchestrator.enhanced_session import EnhancedSession


async def main():
    """Run orchestrator with automatic hardware optimization."""
    # Create enhanced session
    session = EnhancedSession(".")
    
    # Interactive mode if no arguments
    if len(sys.argv) < 2:
        print("🚀 Unity Wheel Orchestrator - M4 Pro Optimized")
        print("=" * 50)
        print("\nNo command provided, entering interactive mode...")
        print("(Use './orchestrate <command>' for single execution)")
        
        await session.initialize()
        await session.interactive_mode()
        return
    
    # Single command execution
    command = " ".join(sys.argv[1:])
    
    # Initialize session with optimizations
    await session.initialize()
    
    print(f"\n🎯 Orchestrating: {command}")
    print("-" * 70)
    
    # Execute command
    result = await session.execute(command)
    
    # Display results
    print(f"\n✅ Strategy used: {result.get('strategy', 'unknown')}")
    
    if "phases" in result:
        for phase_name, phase_data in result["phases"].items():
            if isinstance(phase_data, dict) and phase_data:
                print(f"\n{phase_name.upper()} Phase:")
                for key, value in phase_data.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  • {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  • {key}: {len(value)} entries")
                    else:
                        print(f"  • {key}: {value}")
    
    # Show performance metrics
    if "session_metrics" in result:
        metrics = result["session_metrics"]
        print(f"\n⚡ Performance:")
        print(f"  • Duration: {metrics['duration_ms']:.1f}ms")
        print(f"  • Memory: +{metrics['memory_delta_mb']:.1f}MB")
        print(f"  • Backend: {metrics['backend']}")
        
    if "execution_metrics" in result:
        metrics = result["execution_metrics"]
        print(f"\n📊 Detailed Metrics:")
        print(f"  • Total: {metrics.get('duration_ms', 0):.1f}ms")
        print(f"  • CPU: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"  • Memory: {metrics.get('memory_mb', 0):.1f}MB")
    
    # Shutdown
    await session.shutdown()


if __name__ == "__main__":
    asyncio.run(main())