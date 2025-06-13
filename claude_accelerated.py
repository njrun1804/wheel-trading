#!/usr/bin/env python3
"""Claude Code hardware acceleration wrapper - simple and direct."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unity_wheel.direct_accelerator import (
    get_accelerator, fast_search, fast_read, fast_glob, fast_process
)


async def accelerated_command(command: str) -> Dict[str, Any]:
    """Execute Claude's command with hardware acceleration."""
    
    start = time.perf_counter()
    acc = get_accelerator()
    
    # Parse command to determine what to do
    command_lower = command.lower()
    
    # Search operations
    if any(word in command_lower for word in ["search", "find", "grep", "look for"]):
        # Extract search pattern (simple heuristic)
        import re
        quoted = re.findall(r'"([^"]+)"', command) or re.findall(r"'([^']+)'", command)
        pattern = quoted[0] if quoted else command.split()[-1]
        
        print(f"🔍 Searching for '{pattern}' using {acc.cpu_count} cores...")
        results = await fast_search(pattern)
        
        return {
            "action": "search",
            "pattern": pattern,
            "results": results,
            "count": len(results),
            "duration_ms": (time.perf_counter() - start) * 1000,
            "cores_used": acc.cpu_count
        }
    
    # Read operations
    elif any(word in command_lower for word in ["read", "open", "show", "cat"]):
        # Find file paths in command
        words = command.split()
        paths = [w for w in words if ('/' in w or '.' in w) and Path(w).suffix]
        
        if paths:
            print(f"📖 Reading {len(paths)} files in parallel...")
            contents = await fast_read(paths)
            
            return {
                "action": "read",
                "files": paths,
                "contents": contents,
                "duration_ms": (time.perf_counter() - start) * 1000,
                "cores_used": acc.cpu_count
            }
    
    # Glob/list operations
    elif any(word in command_lower for word in ["list", "glob", "files", "*.py", "*.js"]):
        # Extract patterns
        import re
        patterns = re.findall(r'\*\.[a-zA-Z]+', command) or ["**/*.py"]
        
        print(f"📁 Finding files matching {patterns} using {acc.cpu_count} cores...")
        matches = await fast_glob(patterns)
        
        return {
            "action": "glob",
            "patterns": patterns,
            "matches": matches,
            "total_files": sum(len(files) for files in matches.values()),
            "duration_ms": (time.perf_counter() - start) * 1000,
            "cores_used": acc.cpu_count
        }
    
    # Process operations (analyze, optimize, etc)
    elif any(word in command_lower for word in ["analyze", "process", "optimize", "check"]):
        # This would process files in parallel
        print(f"⚡ Processing with {acc.cpu_count} cores...")
        
        # Example: process all Python files
        py_files = await fast_glob(["**/*.py"])
        all_files = [f for files in py_files.values() for f in files][:10]  # Limit for demo
        
        # Simple analysis function
        def analyze_file(path):
            try:
                with open(path) as f:
                    lines = f.readlines()
                return {
                    "file": path,
                    "lines": len(lines),
                    "size": Path(path).stat().st_size
                }
            except:
                return {"file": path, "error": True}
        
        results = await fast_process(analyze_file, all_files, cpu_bound=False)
        
        return {
            "action": "process",
            "files_analyzed": len(results),
            "results": results,
            "duration_ms": (time.perf_counter() - start) * 1000,
            "cores_used": acc.cpu_count
        }
    
    else:
        # Default: just show we're ready
        return {
            "action": "ready",
            "message": f"Hardware accelerator ready with {acc.cpu_count} cores",
            "gpu_available": "mlx" in sys.modules,
            "duration_ms": (time.perf_counter() - start) * 1000
        }


async def main():
    """Main entry point for testing."""
    if len(sys.argv) < 2:
        print("🚀 Claude Hardware Accelerator")
        print("Usage: ./claude_accelerated.py '<command>'")
        print("\nExamples:")
        print("  ./claude_accelerated.py 'search for TODO'")
        print("  ./claude_accelerated.py 'read src/main.py src/utils.py'")
        print("  ./claude_accelerated.py 'list all *.py files'")
        print("  ./claude_accelerated.py 'analyze code'")
        return
    
    command = " ".join(sys.argv[1:])
    result = await accelerated_command(command)
    
    # Display results
    print(f"\n✅ Action: {result['action']}")
    print(f"⚡ Duration: {result.get('duration_ms', 0):.1f}ms")
    print(f"🔧 Cores used: {result.get('cores_used', 'N/A')}")
    
    if result["action"] == "search" and "results" in result:
        print(f"\nFound {result['count']} matches:")
        for r in result["results"][:5]:
            print(f"  {r['file']}:{r['line']} - {r['content'][:60]}...")
    
    elif result["action"] == "read" and "contents" in result:
        for path, content in result["contents"].items():
            print(f"\n{path}:")
            print(content[:200] + "..." if len(content) > 200 else content)
    
    elif result["action"] == "glob" and "matches" in result:
        print(f"\nFound {result['total_files']} files")
        for pattern, files in result["matches"].items():
            print(f"\n{pattern}: {len(files)} files")
            for f in files[:3]:
                print(f"  - {f}")
    
    # Cleanup
    get_accelerator().cleanup()


if __name__ == "__main__":
    asyncio.run(main())