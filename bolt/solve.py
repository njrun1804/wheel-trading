#!/usr/bin/env python3
"""
Bolt problem solver with full 8-agent system integration.

Uses the comprehensive integration layer with hardware acceleration,
GPU routing, memory safety, and real-time monitoring for M4 Pro.
"""

import asyncio
import sys
from pathlib import Path
import click
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the new integration layer
from bolt.integration import BoltIntegration, bolt_solve_cli

async def solve_problem(instruction: str, analyze_only: bool = False) -> Dict[str, Any]:
    """Solve a problem using the integrated Bolt system.
    
    This now uses the full integration layer with:
    - 8 parallel agents with hardware acceleration
    - Real-time M4 Pro system monitoring
    - GPU/CPU routing optimization
    - Memory safety enforcement
    - Comprehensive task orchestration
    """
    
    # Use the integrated system
    integration = BoltIntegration()
    
    try:
        # Initialize the full system
        await integration.initialize()
        
        # Solve using the integrated approach
        result = await integration.solve(instruction, analyze_only)
        
        return result
        
    finally:
        # Ensure clean shutdown
        await integration.shutdown()

# Legacy functions kept for backwards compatibility
# The new integration layer in bolt.integration.py handles all of this





@click.command()
@click.argument("instruction")
@click.option("--analyze-only", is_flag=True, help="Only analyze, don't make changes")
def main(instruction: str, analyze_only: bool):
    """Solve problems using integrated 8-agent system with M4 Pro acceleration.
    
    Features:
    - 8 parallel Claude Code agents
    - Hardware-accelerated tools (MLX GPU, Metal)
    - Real-time system monitoring
    - Memory safety enforcement
    - Intelligent task orchestration
    
    Examples:
        bolt solve "optimize database queries"
        bolt solve "fix memory leak in trading module" --analyze-only  
        bolt solve "refactor wheel strategy code"
    """
    
    # Use the integrated CLI directly
    return asyncio.run(bolt_solve_cli(instruction, analyze_only))

if __name__ == "__main__":
    sys.exit(main())