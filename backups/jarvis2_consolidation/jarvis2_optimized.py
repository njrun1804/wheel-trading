#!/usr/bin/env python3
"""Optimized Jarvis 2.0 launcher with all fixes applied."""
import os

# Essential environment fixes for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
os.environ['MTL_DEBUG_LAYER'] = '0'  # Disable Metal validation
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # MPS fallback

import asyncio
import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress noisy loggers
logging.getLogger('faiss').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.WARNING)

from jarvis2 import Jarvis2, Jarvis2Config
from jarvis2.core.solution import CodeSolution


class OptimizedJarvis:
    """Optimized Jarvis wrapper with proper initialization."""
    
    def __init__(self):
        self.jarvis = None
    
    async def initialize(self):
        """Initialize Jarvis with optimized settings."""
        print("🚀 Initializing Jarvis 2.0 (Optimized)...")
        
        # Create optimized config
        config = Jarvis2Config(
            # Reduced for faster startup
            max_parallel_simulations=100,
            gpu_batch_size=32,
            num_diverse_solutions=10,
            
            # Disable background tasks during init
            index_update_interval=3600,  # 1 hour
            background_learning_interval=3600,
            model_update_threshold=1000,
            
            # Use temp directories for testing
            index_path=Path("/tmp/.jarvis/indexes"),
            model_path=Path("/tmp/.jarvis/models"),
            experience_path=Path("/tmp/.jarvis/experience"),
        )
        
        # Create Jarvis instance
        self.jarvis = Jarvis2(config)
        
        # Initialize with timeout
        try:
            await asyncio.wait_for(
                self._initialize_components(),
                timeout=30.0
            )
            print("✅ Jarvis 2.0 initialized successfully!")
        except asyncio.TimeoutError:
            print("❌ Initialization timed out")
            raise
    
    async def _initialize_components(self):
        """Initialize components one by one."""
        components = [
            ("Hardware", self.jarvis.hardware_executor.initialize()),
            ("Indexes", self.jarvis.index_manager.initialize()),
            ("MCTS", self._init_mcts()),
            ("Experience", self.jarvis.experience_buffer.initialize()),
        ]
        
        for name, init_coro in components:
            print(f"  • Initializing {name}...", end='', flush=True)
            start = time.time()
            await init_coro
            print(f" ✅ ({time.time() - start:.1f}s)")
        
        # Mark as initialized
        self.jarvis._initialized = True
    
    async def _init_mcts(self):
        """Initialize MCTS with optimizations."""
        # Skip model loading for faster startup
        self.jarvis.mcts.total_simulations = 0
        logging.info("MCTS initialized (models will load on demand)")
    
    async def query(self, text: str) -> CodeSolution:
        """Run a query with proper error handling."""
        if not self.jarvis._initialized:
            await self.initialize()
        
        print(f"\n🔍 Processing: {text}")
        start = time.time()
        
        try:
            solution = await asyncio.wait_for(
                self.jarvis.assist(text),
                timeout=30.0
            )
            
            elapsed = time.time() - start
            print(f"✅ Completed in {elapsed:.1f}s")
            print(f"   • Confidence: {solution.confidence:.0%}")
            print(f"   • Simulations: {solution.metrics.simulations_performed}")
            print(f"   • Code length: {len(solution.code)} chars")
            
            return solution
            
        except asyncio.TimeoutError:
            print("❌ Query timed out")
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.jarvis:
            await self.jarvis.shutdown()


async def test_jarvis():
    """Test Jarvis with a simple query."""
    jarvis = OptimizedJarvis()
    
    try:
        # Initialize
        await jarvis.initialize()
        
        # Run test queries
        queries = [
            "create a function to add two numbers",
            "implement binary search in Python",
            "optimize this loop for performance: for i in range(1000): result += i",
        ]
        
        for query in queries:
            solution = await jarvis.query(query)
            
            # Show code preview
            if solution.code:
                print("\n📝 Generated code:")
                print("-" * 40)
                preview = solution.code[:200] + "..." if len(solution.code) > 200 else solution.code
                print(preview)
                print("-" * 40)
        
        # Show stats
        stats = jarvis.jarvis.get_stats()
        print(f"\n📊 Final stats:")
        print(f"   • Total assists: {stats['total_assists']}")
        print(f"   • Average time: {stats['average_time_ms']:.0f}ms")
        print(f"   • Memory usage: {stats['memory_usage_mb']:.0f}MB")
        
    finally:
        await jarvis.shutdown()
        print("\n✅ Test complete!")


async def interactive_mode():
    """Run Jarvis in interactive mode."""
    jarvis = OptimizedJarvis()
    
    try:
        await jarvis.initialize()
        
        print("\n🤖 Jarvis 2.0 Interactive Mode")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                query = input("jarvis> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                
                solution = await jarvis.query(query)
                
                # Show solution
                print("\n" + solution.format_display())
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    finally:
        await jarvis.shutdown()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Jarvis 2.0")
    parser.add_argument('-t', '--test', action='store_true', help='Run tests')
    parser.add_argument('-q', '--query', help='Single query')
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_jarvis())
    elif args.query:
        async def single_query():
            jarvis = OptimizedJarvis()
            try:
                await jarvis.initialize()
                solution = await jarvis.query(args.query)
                print("\n" + solution.format_display())
            finally:
                await jarvis.shutdown()
        
        asyncio.run(single_query())
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()