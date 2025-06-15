#!/usr/bin/env python3
"""Jarvis 2.0 - Intelligent Meta-Coder CLI.

An AI-powered code generation system that learns and evolves,
optimized for Apple M4 Pro hardware.
"""
import os
# Fix OpenMP and Metal issues on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MTL_DEBUG_LAYER'] = '0'

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json
import time

from jarvis2 import Jarvis2, Jarvis2Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Jarvis2CLI:
    """Command-line interface for Jarvis 2.0."""
    
    def __init__(self):
        self.jarvis = None
        self.config = None
    
    async def initialize(self, args):
        """Initialize Jarvis with configuration."""
        # Load or create config
        if args.config:
            with open(args.config) as f:
                config_dict = json.load(f)
                self.config = Jarvis2Config(**config_dict)
        else:
            self.config = Jarvis2Config()
        
        # Override with CLI arguments
        if args.simulations:
            self.config.max_parallel_simulations = args.simulations
        if args.variants:
            self.config.num_diverse_solutions = args.variants
        
        # Create Jarvis instance
        self.jarvis = Jarvis2(self.config)
        
        # Initialize
        print("🚀 Initializing Jarvis 2.0...")
        await self.jarvis.initialize()
        print("✅ Jarvis 2.0 ready!")
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🤖 Jarvis 2.0 Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit\n")
        
        while True:
            try:
                query = input("jarvis> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                
                if query.lower() == 'help':
                    self.print_help()
                    continue
                
                if query.lower() == 'stats':
                    self.print_stats()
                    continue
                
                # Process query
                await self.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"❌ Error: {e}")
    
    async def process_query(self, query: str, context: dict = None):
        """Process a single query."""
        print(f"\n🔍 Processing: {query}")
        print("⏳ Generating solution...\n")
        
        start_time = time.time()
        
        # Get solution
        solution = await self.jarvis.assist(query, context)
        
        # Display result
        print(solution.format_display())
        print(f"\n⏱️  Total time: {time.time() - start_time:.2f}s")
    
    def print_help(self):
        """Print help information."""
        print("""
📚 Jarvis 2.0 Commands:

  General:
    help          - Show this help message
    stats         - Show performance statistics
    exit          - Exit Jarvis
  
  Code Generation:
    Just type your request naturally, for example:
    - "optimize this function for performance"
    - "add error handling to this code"
    - "refactor this class to use dependency injection"
    - "create unit tests for this module"
  
  Advanced:
    You can provide context by prefixing with @context:
    @context file:path/to/file.py "optimize this function"
        """)
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.jarvis.get_stats()
        
        print("\n📊 Jarvis 2.0 Statistics:")
        print(f"  Total assists: {stats['total_assists']}")
        print(f"  Average time: {stats['average_time_ms']:.0f}ms")
        print(f"  GPU utilization: {stats['gpu_utilization']:.1f}%")
        print(f"  Memory usage: {stats['memory_usage_mb']:.0f}MB")
        print(f"  Index size: {stats.get('index_size', 'N/A')}")
        print(f"  Models updated: {stats.get('models_updated', 0)} times")
    
    async def batch_mode(self, queries: list):
        """Process multiple queries."""
        print(f"\n🔄 Processing {len(queries)} queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            await self.process_query(query)
            print("\n" + "="*60 + "\n")
    
    async def run(self, args):
        """Main run method."""
        try:
            # Initialize
            await self.initialize(args)
            
            if args.query:
                # Single query mode
                await self.process_query(args.query)
            elif args.batch:
                # Batch mode
                with open(args.batch) as f:
                    queries = [line.strip() for line in f if line.strip()]
                await self.batch_mode(queries)
            else:
                # Interactive mode
                await self.interactive_mode()
            
        finally:
            # Cleanup
            if self.jarvis:
                await self.jarvis.shutdown()


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Jarvis 2.0 - Intelligent Meta-Coder for M4 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python jarvis2.py
  
  # Single query
  python jarvis2.py -q "optimize this sorting algorithm"
  
  # Batch processing
  python jarvis2.py -b queries.txt
  
  # With custom config
  python jarvis2.py -c jarvis_config.json
  
  # High-quality mode
  python jarvis2.py -s 5000 -v 200 -q "complex refactoring task"
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        help='Single query to process'
    )
    
    parser.add_argument(
        '-b', '--batch',
        help='Batch file containing queries (one per line)'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Configuration file (JSON)'
    )
    
    parser.add_argument(
        '-s', '--simulations',
        type=int,
        help='Number of MCTS simulations (default: auto)'
    )
    
    parser.add_argument(
        '-v', '--variants',
        type=int,
        help='Number of diverse variants to generate (default: 100)'
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Disable GPU if requested
    if args.no_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Print banner
    print("""
    ╦╔═╗╦═╗╦  ╦╦╔═╗  ╔═╗ ╔═╗
    ║╠═╣╠╦╝╚╗╔╝║╚═╗  ╔═╝ ║ ║
   ╚╝╩ ╩╩╚═ ╚╝ ╩╚═╝  ╚═╝o╚═╝
   
   Intelligent Meta-Coder for M4 Pro
   Neural-Guided MCTS • Diversity Engine • Continuous Learning
    """)
    
    # Run CLI
    cli = Jarvis2CLI()
    await cli.run(args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)