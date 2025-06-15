#!/usr/bin/env python3
"""
Standalone meta-coding assistant using hardware-accelerated sequential thinking.
For real coding tasks on this M4 Pro Mac.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.unity_wheel.accelerated_tools.sequential_thinking_metacoding import get_metacoding_thinking


async def optimize_code(file_path: str):
    """Optimize a specific file."""
    thinking = get_metacoding_thinking()
    
    print(f"\n🚀 OPTIMIZING: {file_path}")
    print("=" * 70)
    
    steps = await thinking.think_about_code(
        task=f"optimize {file_path} for maximum performance on M4 Pro",
        target_files=[file_path],
        constraints=[
            "Use all 8 P-cores",
            "Leverage Metal GPU via MLX",
            "Maintain API compatibility",
            "Add performance metrics"
        ]
    )
    
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        print(f"   Code: {step.code_operation[:100]}...")
        print(f"   Why: {step.reasoning}")
        print(f"   Expect: {step.expected_output}")
        
        # Execute if it's a safe read operation
        if step.code_operation.startswith(('grep', 'find', 'cat')):
            try:
                result = subprocess.run(
                    step.code_operation.replace('target_file.py', file_path),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.stdout:
                    print(f"   Found: {result.stdout[:200]}...")
            except:
                pass
    
    thinking.close()


async def implement_feature(feature_description: str):
    """Implement a new feature."""
    thinking = get_metacoding_thinking()
    
    print(f"\n✨ IMPLEMENTING: {feature_description}")
    print("=" * 70)
    
    steps = await thinking.think_about_code(
        task=f"implement {feature_description}",
        constraints=[
            "Follow existing patterns",
            "Use type hints",
            "Add docstrings",
            "Make it fast with M4 Pro optimizations"
        ],
        context={
            "language": "Python",
            "framework": "asyncio",
            "hardware": "M4 Pro"
        }
    )
    
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        if len(step.code_operation) > 200:
            print(f"   Code:\n{step.code_operation}")
        else:
            print(f"   Code: {step.code_operation}")
        print(f"   Reasoning: {step.reasoning}")
    
    thinking.close()


async def accelerate_function(file_path: str, function_name: str):
    """Accelerate a specific function."""
    thinking = get_metacoding_thinking()
    
    print(f"\n⚡ ACCELERATING: {function_name} in {file_path}")
    print("=" * 70)
    
    steps = await thinking.think_about_code(
        task=f"accelerate function {function_name} using Metal GPU and parallel processing",
        target_files=[file_path],
        constraints=[
            "Convert numpy to MLX",
            "Add parallel processing",
            "Pre-allocate buffers",
            "Maintain correctness"
        ]
    )
    
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        print(f"   {step.reasoning}")
        
        # Show the actual code changes
        if "mlx" in step.code_operation.lower() or "parallel" in step.code_operation.lower():
            print(f"\n   Code to add:")
            print("   " + "-" * 50)
            print(step.code_operation)
            print("   " + "-" * 50)
    
    thinking.close()


async def analyze_performance(file_path: str):
    """Analyze performance bottlenecks."""
    thinking = get_metacoding_thinking()
    
    print(f"\n📊 ANALYZING PERFORMANCE: {file_path}")
    print("=" * 70)
    
    # First, profile it
    print("\nProfiling...")
    try:
        subprocess.run(
            f"python -m cProfile -s cumulative {file_path} > profile_output.txt 2>&1",
            shell=True,
            timeout=10
        )
    except:
        print("(Profiling skipped)")
    
    steps = await thinking.think_about_code(
        task=f"identify and fix performance bottlenecks in {file_path}",
        target_files=[file_path],
        constraints=[
            "Find hot loops",
            "Identify memory allocations", 
            "Check for GPU opportunities",
            "Look for parallelization"
        ]
    )
    
    bottlenecks = []
    
    for step in steps:
        print(f"\n{step.step_number}. {step.action}")
        
        # Execute analysis commands
        if step.code_operation.startswith(('grep', 'ast-grep')):
            try:
                cmd = step.code_operation.replace('target_file.py', file_path)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                if result.stdout:
                    print(f"   Found issues:")
                    for line in result.stdout.strip().split('\n')[:5]:
                        print(f"     • {line}")
                        bottlenecks.append(line)
            except:
                pass
    
    if bottlenecks:
        print(f"\n🎯 TOP OPTIMIZATION OPPORTUNITIES:")
        for i, issue in enumerate(bottlenecks[:5], 1):
            print(f"   {i}. {issue}")
    
    thinking.close()


async def interactive_mode():
    """Interactive meta-coding assistant."""
    thinking = get_metacoding_thinking()
    
    print("\n🤖 META-CODING ASSISTANT (M4 Pro Optimized)")
    print("=" * 70)
    print("Commands:")
    print("  optimize <file>     - Optimize file for M4 Pro")
    print("  implement <desc>    - Implement new feature")
    print("  accelerate <file> <func> - GPU accelerate function")
    print("  analyze <file>      - Find performance issues")
    print("  quit               - Exit")
    print("")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command == "quit":
                break
                
            parts = command.split()
            if not parts:
                continue
                
            cmd = parts[0]
            
            if cmd == "optimize" and len(parts) > 1:
                await optimize_code(parts[1])
                
            elif cmd == "implement" and len(parts) > 1:
                feature = " ".join(parts[1:])
                await implement_feature(feature)
                
            elif cmd == "accelerate" and len(parts) > 2:
                await accelerate_function(parts[1], parts[2])
                
            elif cmd == "analyze" and len(parts) > 1:
                await analyze_performance(parts[1])
                
            else:
                print("Invalid command. Try 'help' for usage.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    thinking.close()


async def demo():
    """Demo the meta-coding capabilities."""
    print("META-CODING DEMO: Optimizing Sequential Thinking")
    print("=" * 70)
    
    # Example 1: Optimize our own sequential thinking
    await optimize_code("src/unity_wheel/accelerated_tools/sequential_thinking_turbo.py")
    
    # Example 2: Implement a new feature
    await implement_feature("caching layer for repeated thinking operations")
    
    # Example 3: Accelerate a function
    await accelerate_function(
        "src/unity_wheel/accelerated_tools/sequential_thinking_turbo.py",
        "_score_candidates_gpu"
    )


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            await demo()
        elif command == "interactive":
            await interactive_mode()
        elif command == "optimize" and len(sys.argv) > 2:
            await optimize_code(sys.argv[2])
        elif command == "implement" and len(sys.argv) > 2:
            await implement_feature(" ".join(sys.argv[2:]))
        elif command == "accelerate" and len(sys.argv) > 3:
            await accelerate_function(sys.argv[2], sys.argv[3])
        elif command == "analyze" and len(sys.argv) > 2:
            await analyze_performance(sys.argv[2])
        else:
            print("Usage:")
            print("  ./metacode.py demo                    # Run demo")
            print("  ./metacode.py interactive             # Interactive mode")
            print("  ./metacode.py optimize <file>         # Optimize file")
            print("  ./metacode.py implement <description> # Implement feature")
            print("  ./metacode.py accelerate <file> <func># Accelerate function")
            print("  ./metacode.py analyze <file>          # Analyze performance")
    else:
        # Default to interactive mode
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())