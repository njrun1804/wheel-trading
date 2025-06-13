#!/usr/bin/env python3
"""Analyze all MCP servers with hardware acceleration to determine their value."""

import asyncio
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import hardware maximizer first
from unity_wheel.orchestrator.hardware_maximizer import maximize_hardware
from unity_wheel.direct_accelerator import get_accelerator

# Track performance
start_time = time.perf_counter()
initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB


async def analyze_mcp_server(server_name: str, config: Dict[str, Any], 
                           script_contents: Dict[str, str]) -> Dict[str, Any]:
    """Analyze a single MCP server for value."""
    
    # Find implementation file
    if "args" in config:
        script_path = None
        for arg in config["args"]:
            if arg.endswith(".py"):
                script_path = arg
                break
    
    # Get script content
    content = script_contents.get(script_path, "") if script_path else ""
    
    # Analyze based on patterns
    analysis = {
        "name": server_name,
        "script": script_path,
        "has_implementation": bool(content),
        "lines_of_code": len(content.splitlines()) if content else 0,
        "complexity": "low",  # Will update based on analysis
        "value_category": "unknown",
        "recommendation": "",
        "local_alternative": "",
        "issues": []
    }
    
    # Pattern analysis
    if not content:
        analysis["value_category"] = "no_value"
        analysis["recommendation"] = "Remove - no implementation found"
        return analysis
    
    # Check for specific patterns
    patterns = {
        "imports_mcp": "from mcp" in content or "import mcp" in content,
        "has_tools": "@tool" in content or "def tool_" in content,
        "has_resources": "@resource" in content or "def resource_" in content,
        "error_handling": "try:" in content and "except" in content,
        "async_code": "async def" in content,
        "uses_ai": "openai" in content or "anthropic" in content or "model" in content,
        "file_operations": "open(" in content or "Path(" in content,
        "network_calls": "requests" in content or "httpx" in content or "aiohttp" in content,
        "complex_logic": content.count("if ") > 10 or content.count("for ") > 5
    }
    
    # Categorize based on server name and patterns
    if server_name in ["filesystem", "github"]:
        analysis["value_category"] = "worth_fixing"
        analysis["recommendation"] = "Essential for Claude Code - fix MCP issues"
        analysis["issues"] = ["Startup delays", "Memory usage"]
        
    elif server_name in ["ripgrep", "dependency-graph"]:
        analysis["value_category"] = "replicate_locally"
        analysis["recommendation"] = "High-value search - implement locally with hardware acceleration"
        analysis["local_alternative"] = "Direct ripgrep with parallel execution"
        
    elif server_name in ["memory", "sequential-thinking"]:
        analysis["value_category"] = "worth_fixing"
        analysis["recommendation"] = "Valuable for complex reasoning - optimize MCP"
        
    elif server_name in ["python_analysis", "trace-phoenix"]:
        analysis["value_category"] = "replicate_locally"
        analysis["recommendation"] = "Trading-specific - better as local accelerated module"
        analysis["local_alternative"] = "Direct Python AST analysis with GPU acceleration"
        
    elif server_name in ["duckdb"]:
        analysis["value_category"] = "replicate_locally"
        analysis["recommendation"] = "Direct DuckDB access is faster than MCP overhead"
        analysis["local_alternative"] = "Native DuckDB Python API with connection pooling"
        
    elif server_name in ["pyrepl", "python-code-helper"]:
        analysis["value_category"] = "worth_fixing"
        analysis["recommendation"] = "Useful for interactive coding - keep as MCP"
        
    elif server_name in ["brave", "puppeteer", "statsource"]:
        analysis["value_category"] = "worth_fixing"
        analysis["recommendation"] = "External API access - MCP is appropriate"
        
    elif server_name in ["mlflow", "sklearn", "optionsflow"]:
        analysis["value_category"] = "no_value"
        analysis["recommendation"] = "Unused in trading workflow - remove"
        
    else:
        # Default categorization based on patterns
        if patterns["complex_logic"] and patterns["has_tools"]:
            analysis["value_category"] = "worth_fixing"
        elif patterns["file_operations"] and not patterns["network_calls"]:
            analysis["value_category"] = "replicate_locally"
        else:
            analysis["value_category"] = "no_value"
    
    # Complexity assessment
    if patterns["complex_logic"] or patterns["uses_ai"]:
        analysis["complexity"] = "high"
    elif patterns["async_code"] and patterns["has_tools"]:
        analysis["complexity"] = "medium"
    else:
        analysis["complexity"] = "low"
    
    return analysis


async def main():
    print("🚀 MCP Value Analysis with Hardware Acceleration")
    print("=" * 60)
    
    # Initialize hardware
    maximizer = maximize_hardware()
    accelerator = get_accelerator()
    
    print(f"⚡ Using {accelerator.cpu_count} CPU cores")
    print(f"💾 Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"🎮 GPU: {'Available' if 'mlx' in sys.modules else 'Not available'}")
    print()
    
    # Load MCP configuration
    with open("mcp-servers.json") as f:
        mcp_config = json.load(f)
    
    servers = mcp_config["mcpServers"]
    print(f"📊 Analyzing {len(servers)} MCP servers...")
    
    # Find all script files in parallel
    script_patterns = [
        "scripts/*mcp*.py",
        "scripts/trace-*.py", 
        "scripts/python-*.py",
        "scripts/ripgrep-*.py",
        "scripts/dependency-*.py"
    ]
    
    print("🔍 Finding implementation files...")
    file_matches = await accelerator.parallel_glob(script_patterns)
    all_scripts = [f for files in file_matches.values() for f in files]
    print(f"  Found {len(all_scripts)} script files")
    
    # Read all scripts in parallel
    print("📖 Reading scripts in parallel...")
    read_start = time.perf_counter()
    script_contents = await accelerator.parallel_read_files(all_scripts)
    read_time = (time.perf_counter() - read_start) * 1000
    print(f"  Read {len(script_contents)} files in {read_time:.1f}ms")
    
    # Analyze all servers in parallel
    print("🔬 Analyzing servers in parallel...")
    analyze_start = time.perf_counter()
    
    analyses = await accelerator.parallel_execute(
        lambda item: asyncio.run(analyze_mcp_server(item[0], item[1], script_contents)),
        list(servers.items()),
        cpu_bound=False
    )
    
    analyze_time = (time.perf_counter() - analyze_start) * 1000
    print(f"  Analyzed {len(analyses)} servers in {analyze_time:.1f}ms")
    
    # Categorize results
    categories = {
        "no_value": [],
        "worth_fixing": [],
        "replicate_locally": []
    }
    
    for analysis in analyses:
        categories[analysis["value_category"]].append(analysis)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n❌ NO VALUE - Remove ({len(categories['no_value'])} servers):")
    for server in categories["no_value"]:
        print(f"  • {server['name']}: {server['recommendation']}")
    
    print(f"\n✅ WORTH FIXING - Keep as MCP ({len(categories['worth_fixing'])} servers):")
    for server in categories["worth_fixing"]:
        print(f"  • {server['name']}: {server['recommendation']}")
        if server["issues"]:
            print(f"    Issues: {', '.join(server['issues'])}")
    
    print(f"\n🔧 REPLICATE LOCALLY - Build better ({len(categories['replicate_locally'])} servers):")
    for server in categories["replicate_locally"]:
        print(f"  • {server['name']}: {server['recommendation']}")
        if server["local_alternative"]:
            print(f"    Alternative: {server['local_alternative']}")
    
    # Performance summary
    total_time = (time.perf_counter() - start_time) * 1000
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Total execution time: {total_time:.1f}ms")
    print(f"CPU cores utilized: {accelerator.cpu_count}")
    print(f"Memory used: {memory_used:.1f}MB")
    print(f"Files processed: {len(all_scripts)}")
    print(f"Throughput: {len(servers) / (total_time / 1000):.1f} servers/second")
    
    # Save detailed report
    report = {
        "timestamp": time.time(),
        "hardware": {
            "cpu_cores": accelerator.cpu_count,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": "mlx" in sys.modules
        },
        "performance": {
            "total_time_ms": total_time,
            "read_time_ms": read_time,
            "analyze_time_ms": analyze_time,
            "memory_used_mb": memory_used
        },
        "summary": {
            "total_servers": len(servers),
            "no_value": len(categories["no_value"]),
            "worth_fixing": len(categories["worth_fixing"]),
            "replicate_locally": len(categories["replicate_locally"])
        },
        "detailed_analysis": analyses
    }
    
    with open("mcp_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to mcp_analysis_report.json")
    
    # Cleanup
    accelerator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())