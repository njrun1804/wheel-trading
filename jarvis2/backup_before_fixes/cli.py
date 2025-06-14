#!/usr/bin/env python3
"""Command-line interface for Jarvis2 meta-coding system."""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis2.core.orchestrator import Jarvis2Orchestrator, CodeRequest, Jarvis2Config


async def main():
    parser = argparse.ArgumentParser(
        description="Jarvis2 - AI-powered meta-coding system for M4 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jarvis2 "Create a binary search function"
  jarvis2 "Optimize this sorting algorithm" --simulations 5000
  jarvis2 "Design a REST API for user management" --confidence 0.9
  jarvis2 --stats  # Show system statistics
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Code generation query'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=2000,
        help='Number of MCTS simulations (default: 2000)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.8,
        help='Minimum confidence threshold (default: 0.8)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Configure based on environment
    config = Jarvis2Config()
    
    # Override with environment variables if set
    if os.environ.get('JARVIS2_SEARCH_WORKERS'):
        config.num_search_workers = int(os.environ['JARVIS2_SEARCH_WORKERS'])
    if os.environ.get('JARVIS2_NEURAL_WORKERS'):
        config.num_neural_workers = int(os.environ['JARVIS2_NEURAL_WORKERS'])
    if os.environ.get('JARVIS2_MEMORY_LIMIT_GB'):
        config.max_memory_gb = float(os.environ['JARVIS2_MEMORY_LIMIT_GB'])
    
    config.default_simulations = args.simulations
    
    # Initialize Jarvis2
    print("🤖 Initializing Jarvis2 on M4 Pro...")
    jarvis = Jarvis2Orchestrator(config=config)
    
    try:
        await jarvis.initialize()
        
        if args.stats:
            # Show statistics
            stats = jarvis.get_stats()
            print("\n📊 Jarvis2 System Statistics:")
            print("=" * 50)
            print(f"Requests processed: {stats['request_count']}")
            print(f"Average time: {stats['average_time_ms']:.0f}ms")
            print(f"Memory usage: {stats['memory_stats']['system_used_gb']:.1f}GB / {stats['memory_stats']['system_memory_gb']:.1f}GB")
            print(f"Backends: {stats['backends']}")
            return
        
        if args.interactive or not args.query:
            # Interactive mode
            print("\n✨ Jarvis2 Interactive Mode")
            print("Type 'exit' to quit, 'help' for commands\n")
            
            while True:
                try:
                    query = input("jarvis2> ").strip()
                    
                    if query.lower() in ['exit', 'quit']:
                        break
                    elif query.lower() == 'help':
                        print("\nCommands:")
                        print("  help     - Show this help")
                        print("  stats    - Show system statistics")
                        print("  clear    - Clear screen")
                        print("  exit     - Exit Jarvis2")
                        print("\nOtherwise, enter your code generation query.")
                        continue
                    elif query.lower() == 'stats':
                        stats = jarvis.get_stats()
                        print(f"\nRequests: {stats['request_count']}, Avg time: {stats['average_time_ms']:.0f}ms")
                        continue
                    elif query.lower() == 'clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        continue
                    elif not query:
                        continue
                    
                    # Generate code
                    request = CodeRequest(query)
                    solution = await jarvis.generate_code(request)
                    
                    print(f"\n{'='*60}")
                    print(solution.code)
                    print(f"{'='*60}")
                    print(f"\n📊 Confidence: {solution.confidence:.0%}")
                    print(f"⏱️  Time: {solution.metrics['generation_time_ms']:.0f}ms")
                    print(f"🔧 Backend: {solution.metrics.get('backend_used', 'unknown')}")
                    
                    if solution.alternatives and args.verbose:
                        print(f"\n📝 Alternative solutions: {len(solution.alternatives)}")
                    
                    print()
                    
                except KeyboardInterrupt:
                    print("\n")
                    continue
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
        else:
            # Single query mode
            request = CodeRequest(args.query)
            solution = await jarvis.generate_code(request)
            
            # Check confidence threshold
            if solution.confidence < args.confidence:
                print(f"\n⚠️  Warning: Confidence ({solution.confidence:.0%}) below threshold ({args.confidence:.0%})")
            
            print(f"\n{'='*60}")
            print(solution.code)
            print(f"{'='*60}")
            print(f"\n📊 Confidence: {solution.confidence:.0%}")
            print(f"⏱️  Time: {solution.metrics['generation_time_ms']:.0f}ms")
            
            if args.verbose:
                print(f"🔧 Backend: {solution.metrics.get('backend_used', 'unknown')}")
                print(f"🌳 Simulations: {solution.metrics.get('simulations', 0)}")
                if solution.alternatives:
                    print(f"\n📝 Alternative solutions:")
                    for i, alt in enumerate(solution.alternatives[:3]):
                        print(f"\n{i+1}. (confidence: {alt['confidence']:.0%}):")
                        print(alt['code'][:200] + "..." if len(alt['code']) > 200 else alt['code'])
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n🔌 Shutting down Jarvis2...")
        await jarvis.shutdown()


if __name__ == "__main__":
    asyncio.run(main())