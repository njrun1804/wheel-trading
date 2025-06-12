#!/usr/bin/env python3
"""Demo script for MCP Orchestrator."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.orchestrator import MCPOrchestrator


async def main():
    """Demonstrate orchestrator functionality."""
    print("🚀 MCP Orchestrator Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    workspace = Path(__file__).parent
    orchestrator = MCPOrchestrator(str(workspace))
    
    print("Initializing orchestrator...")
    await orchestrator.initialize()
    
    # Example commands to demonstrate
    commands = [
        "Analyze the trading module for performance improvements",
        "Refactor risk management to use async patterns",
        "Generate comprehensive test suite for options pricing"
    ]
    
    for command in commands:
        print(f"\n📝 Command: {command}")
        print("-" * 50)
        
        # Execute command
        result = await orchestrator.execute_command(command)
        
        # Display results
        print(f"✅ Success: {result['success']}")
        print(f"⏱️  Duration: {result['duration_ms']}ms")
        print(f"🎯 Tokens used: {result['total_tokens']}")
        print(f"💾 Peak memory: {result['memory_peak_mb']:.1f}MB")
        
        # Show phase results
        print("\nPhase Results:")
        for phase in result['phases']:
            status = "✅" if phase['success'] else "❌"
            print(f"  {status} {phase['phase']:12} - {phase['duration_ms']:5}ms - {phase['token_count']} tokens")
            if phase.get('error'):
                print(f"     Error: {phase['error']}")
        
        # Show cache stats
        cache_stats = orchestrator.slice_cache.get_stats()
        print(f"\nCache Stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
              f"{cache_stats['hit_rate']:.1%} hit rate")
        
        # Show memory stats
        mem_stats = orchestrator.memory_monitor.get_stats()
        print(f"Memory Stats: {mem_stats['current_rss_mb']:.1f}MB current, "
              f"{mem_stats['peak_memory_mb']:.1f}MB peak, "
              f"{mem_stats['current_ratio']:.1%} of system")
    
    # Cleanup
    print("\nShutting down orchestrator...")
    await orchestrator.shutdown()
    print("✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())