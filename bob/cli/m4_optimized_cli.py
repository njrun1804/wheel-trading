"""M4 Optimized CLI for Bob.

Demonstrates the M4 Pro optimizations with 12-agent coordination and
hardware acceleration for maximum throughput with Claude Code.
"""

import asyncio
import argparse
import json
import sys
import time
from typing import Dict, Any, List

from ..integration.m4_enhanced_integration import (
    get_m4_enhanced_bob,
    process_query_m4_optimized,
    create_optimized_startup_script
)
from ..utils.logging import get_component_logger


class M4OptimizedCLI:
    """M4 optimized CLI for demonstrating enhanced Bob capabilities."""
    
    def __init__(self):
        self.logger = get_component_logger("m4_optimized_cli")
        self.m4_bob = None
    
    async def initialize(self):
        """Initialize the M4 optimized Bob system."""
        self.logger.info("🚀 Initializing M4 Optimized Bob CLI")
        self.m4_bob = await get_m4_enhanced_bob()
        self.logger.info("✅ M4 Optimized Bob ready for maximum throughput")
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark to validate M4 optimizations."""
        self.logger.info("🏃 Running M4 Pro performance benchmark...")
        
        benchmark_queries = [
            "Analyze the wheel trading strategy implementation for performance bottlenecks",
            "Optimize the Einstein search engine for better memory utilization",
            "Design a robust error handling system for the multi-agent coordinator",
            "Implement caching mechanisms for frequently accessed data",
            "Validate the risk management components for edge cases",
            "Generate documentation for the M4 optimization features",
            "Integrate the new features with the existing Bob architecture",
            "Test the 12-agent system under high load conditions"
        ]
        
        start_time = time.time()
        results = []
        
        # Test concurrent processing
        self.logger.info(f"🔄 Processing {len(benchmark_queries)} concurrent queries...")
        concurrent_tasks = []
        for i, query in enumerate(benchmark_queries):
            task = process_query_m4_optimized(
                query, 
                {"benchmark": True, "query_id": i}
            )
            concurrent_tasks.append(task)
        
        # Execute all queries concurrently
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_queries = 0
        total_agents_used = 0
        
        for i, result in enumerate(concurrent_results):
            if isinstance(result, Exception):
                self.logger.error(f"Query {i} failed: {result}")
                results.append({
                    "query_id": i,
                    "success": False,
                    "error": str(result)
                })
            else:
                successful_queries += 1
                total_agents_used += result.get("agents_utilized", 1)
                results.append({
                    "query_id": i,
                    "success": True,
                    "processing_time": result.get("processing_time", 0),
                    "agents_utilized": result.get("agents_utilized", 1)
                })
        
        # Calculate metrics
        throughput = len(benchmark_queries) / total_time
        avg_agents_per_query = total_agents_used / successful_queries if successful_queries > 0 else 0
        
        benchmark_results = {
            "total_queries": len(benchmark_queries),
            "successful_queries": successful_queries,
            "total_time": total_time,
            "throughput_queries_per_second": throughput,
            "avg_agents_per_query": avg_agents_per_query,
            "concurrent_processing": True,
            "m4_optimizations_active": True,
            "results": results
        }
        
        self.logger.info(f"📊 Benchmark complete: {successful_queries}/{len(benchmark_queries)} queries in {total_time:.2f}s")
        self.logger.info(f"🚀 Throughput: {throughput:.2f} queries/second")
        self.logger.info(f"🤖 Average agents per query: {avg_agents_per_query:.1f}")
        
        return benchmark_results
    
    async def interactive_mode(self):
        """Run interactive mode for testing queries."""
        self.logger.info("🎮 Starting M4 Optimized Interactive Mode")
        print("\n" + "="*60)
        print("🚀 M4 Optimized Bob - Interactive Mode")
        print("📊 12-agent system with M4 Pro hardware acceleration")
        print("💡 Type 'help' for commands, 'exit' to quit")
        print("="*60)
        
        while True:
            try:
                query = input("\n🤖 Bob M4> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'benchmark':
                    print("🏃 Running performance benchmark...")
                    benchmark_results = await self.run_performance_benchmark()
                    self._display_benchmark_results(benchmark_results)
                    continue
                
                if query.lower() == 'status':
                    await self._show_status()
                    continue
                
                # Process the query
                print(f"🔄 Processing: {query}")
                start_time = time.time()
                
                result = await process_query_m4_optimized(query)
                
                processing_time = time.time() - start_time
                
                # Display results
                if result.get("success", False):
                    print(f"✅ Success ({processing_time:.2f}s)")
                    print(f"🤖 Agents used: {result.get('agents_utilized', 1)}")
                    if result.get("result"):
                        print(f"📋 Result: {result['result']}")
                else:
                    print(f"❌ Failed ({processing_time:.2f}s)")
                    if result.get("error"):
                        print(f"🚨 Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Type 'exit' to quit gracefully.")
            except Exception as e:
                print(f"🚨 Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🚀 M4 Optimized Bob Commands:

📊 Query Processing:
  - Type any question or request to process with 12-agent system
  - Example: "Analyze the codebase for performance issues"
  
🛠️ Special Commands:
  - help        Show this help message
  - benchmark   Run performance benchmark
  - status      Show system status and metrics
  - exit/quit   Exit interactive mode

🎯 M4 Pro Features:
  - 8 P-core agents for compute-intensive tasks
  - 4 E-core agents for coordination and I/O
  - HTTP/2 session pooling for Claude requests
  - Hardware-aware task scheduling
  - Dynamic workload balancing
"""
        print(help_text)
    
    async def _show_status(self):
        """Show system status and performance metrics."""
        if not self.m4_bob:
            print("❌ M4 Bob not initialized")
            return
        
        print("📊 M4 Optimized Bob Status:")
        
        try:
            metrics = self.m4_bob.get_performance_report()
            
            print(f"⚡ Initialization time: {metrics.get('initialization_time', 0):.3f}s")
            print(f"🔢 Queries processed: {metrics.get('total_queries_processed', 0)}")
            print(f"⏱️  Average latency: {metrics.get('avg_query_latency', 0):.3f}s")
            
            # M4 optimizer stats
            m4_stats = metrics.get('m4_optimizer', {})
            if m4_stats:
                hardware = m4_stats.get('hardware', {})
                print(f"🖥️  P-cores: {hardware.get('p_cores', 'unknown')}")
                print(f"🖥️  E-cores: {hardware.get('e_cores', 'unknown')}")
                print(f"💾 Memory: {hardware.get('unified_memory_gb', 'unknown')}GB")
            
            # Enhanced coordinator stats
            enhanced_stats = metrics.get('enhanced_coordinator', {})
            if enhanced_stats:
                print(f"📈 P-core utilization: {enhanced_stats.get('p_core_utilization', 0):.1f}")
                print(f"📈 E-core utilization: {enhanced_stats.get('e_core_utilization', 0):.1f}")
            
        except Exception as e:
            print(f"🚨 Error getting status: {e}")
    
    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted way."""
        print("\n" + "="*50)
        print("📊 M4 Pro Performance Benchmark Results")
        print("="*50)
        
        print(f"📈 Throughput: {results['throughput_queries_per_second']:.2f} queries/second")
        print(f"✅ Success rate: {results['successful_queries']}/{results['total_queries']}")
        print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        print(f"🤖 Avg agents/query: {results['avg_agents_per_query']:.1f}")
        print(f"🚀 Concurrent processing: {results['concurrent_processing']}")
        print(f"🔧 M4 optimizations: {results['m4_optimizations_active']}")
        
        # Show per-query breakdown
        print(f"\n📋 Query Breakdown:")
        for result in results['results']:
            if result['success']:
                print(f"  ✅ Query {result['query_id']}: {result['processing_time']:.2f}s ({result['agents_utilized']} agents)")
            else:
                print(f"  ❌ Query {result['query_id']}: {result.get('error', 'Unknown error')}")
        
        print("="*50)


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="M4 Optimized Bob CLI - Maximum throughput with Claude Code"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "benchmark", "query"],
        default="interactive",
        help="CLI mode (default: interactive)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (for query mode)"
    )
    parser.add_argument(
        "--create-startup-script",
        action="store_true",
        help="Create optimized startup script and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./start_bob_m4_optimized.sh",
        help="Output path for startup script"
    )
    
    args = parser.parse_args()
    
    # Handle startup script creation
    if args.create_startup_script:
        script_path = create_optimized_startup_script(args.output)
        print(f"✅ Created optimized startup script: {script_path}")
        print("🚀 Run with: ./start_bob_m4_optimized.sh")
        return
    
    # Initialize CLI
    cli = M4OptimizedCLI()
    
    try:
        await cli.initialize()
        
        if args.mode == "interactive":
            await cli.interactive_mode()
        elif args.mode == "benchmark":
            results = await cli.run_performance_benchmark()
            cli._display_benchmark_results(results)
        elif args.mode == "query":
            if not args.query:
                print("❌ Query mode requires --query argument")
                sys.exit(1)
            
            print(f"🔄 Processing query: {args.query}")
            result = await process_query_m4_optimized(args.query)
            
            if result.get("success", False):
                print(f"✅ Success")
                print(f"📋 Result: {result.get('result', 'No result')}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    except Exception as e:
        print(f"🚨 Fatal error: {e}")
        sys.exit(1)
    finally:
        if cli.m4_bob:
            await cli.m4_bob.shutdown()


if __name__ == "__main__":
    asyncio.run(main())