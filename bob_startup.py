#!/usr/bin/env python3
"""
BOB Unified System Startup

Single-command startup for the complete BOB system with <1s initialization.
Provides hardware detection, component loading, Einstein integration, and system validation.
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bob.startup import (
    rapid_startup_bob,
    StartupConfig,
    get_startup_manager,
    detect_hardware_profile
)
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for startup"""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Reduce noise from some modules
    if not verbose:
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("concurrent.futures").setLevel(logging.WARNING)


async def main_startup(
    verbose: bool = False,
    benchmark: bool = False,
    validate: bool = True,
    hardware_optimization: bool = True,
    parallel_loading: bool = True,
    agent_count: int = 8,
    max_startup_time_ms: float = 1000.0
) -> int:
    """Main startup function"""
    
    setup_logging(verbose)
    
    print("🤖 BOB Unified System Startup")
    print("=" * 50)
    
    # Create startup configuration
    config = StartupConfig(
        max_startup_time_ms=max_startup_time_ms,
        enable_hardware_optimization=hardware_optimization,
        parallel_component_loading=parallel_loading,
        default_agent_count=agent_count,
        skip_non_critical_validation=not validate
    )
    
    try:
        # Phase 1: Hardware Detection (if benchmarking)
        if benchmark:
            print("\n🔍 Phase 1: Hardware Detection")
            hw_start = time.perf_counter()
            hardware_profile = await detect_hardware_profile()
            hw_time = (time.perf_counter() - hw_start) * 1000
            
            print(f"   Hardware: {hardware_profile.cpu_brand}")
            print(f"   CPU: {hardware_profile.cpu_cores} cores ({hardware_profile.performance_cores}P + {hardware_profile.efficiency_cores}E)")
            print(f"   Memory: {hardware_profile.memory_gb:.1f}GB {hardware_profile.memory_type}")
            print(f"   GPU: {hardware_profile.gpu_type} ({hardware_profile.gpu_cores} cores)")
            print(f"   Detection time: {hw_time:.1f}ms")
        
        # Phase 2: Rapid Startup
        print(f"\n🚀 Phase 2: BOB System Initialization (target: {max_startup_time_ms:.0f}ms)")
        
        startup_profile = await rapid_startup_bob(config)
        
        # Phase 3: Results
        print(f"\n📊 Phase 3: Startup Results")
        print(f"   Total time: {startup_profile.total_time_ms:.1f}ms")
        print(f"   Hardware detection: {startup_profile.hardware_detection_ms:.1f}ms")
        print(f"   Component loading: {startup_profile.component_loading_ms:.1f}ms")
        print(f"   Einstein init: {startup_profile.einstein_initialization_ms:.1f}ms")
        print(f"   Agent pool: {startup_profile.agent_pool_creation_ms:.1f}ms")
        print(f"   Validation: {startup_profile.validation_ms:.1f}ms")
        print(f"   Memory usage: {startup_profile.memory_usage_mb:.1f}MB")
        print(f"   Components loaded: {startup_profile.components_loaded}")
        print(f"   Agents started: {startup_profile.agents_started}")
        print(f"   Hardware optimization: {'✅' if startup_profile.hardware_optimization_enabled else '❌'}")
        print(f"   Validation passed: {'✅' if startup_profile.validation_passed else '❌'}")
        
        # Performance assessment
        if startup_profile.total_time_ms <= max_startup_time_ms:
            if startup_profile.validation_passed:
                print(f"\n🏆 EXCELLENT: Startup completed successfully under target time!")
                status_code = 0
            else:
                print(f"\n✅ GOOD: Startup completed under target time with warnings")
                status_code = 0
        else:
            print(f"\n⚠️ SLOW: Startup time exceeded target ({startup_profile.total_time_ms:.1f}ms > {max_startup_time_ms:.0f}ms)")
            status_code = 1
        
        # Additional diagnostics if requested
        if benchmark:
            print(f"\n🔧 System Diagnostics:")
            startup_manager = get_startup_manager()
            diagnostics = startup_manager.get_startup_diagnostics()
            
            print(f"   Initialization state: {'✅' if diagnostics['is_initialized'] else '❌'}")
            print(f"   Components loaded: {diagnostics['startup_profile']['components_loaded']}")
            print(f"   Critical errors: {len(diagnostics['critical_errors'])}")
            
            if diagnostics['critical_errors']:
                print("   Errors:")
                for error in diagnostics['critical_errors']:
                    print(f"     - {error}")
        
        return status_code
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        print(f"\n❌ STARTUP FAILED: {str(e)}")
        return 1


def create_startup_script() -> None:
    """Create a simple startup script for easy launching"""
    script_content = '''#!/bin/bash
# BOB Quick Startup Script
# Generated automatically

set -e

echo "🤖 Starting BOB Unified System..."

# Run Python startup with optimizations
python3 bob_startup.py --optimize --parallel

echo "✅ BOB startup complete!"
'''
    
    script_path = Path("start_bob.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    print(f"✅ Created startup script: {script_path}")


async def interactive_startup() -> None:
    """Interactive startup with user prompts"""
    print("🤖 BOB Interactive Startup Configuration")
    print("-" * 40)
    
    # Hardware optimization
    hw_opt = input("Enable hardware optimization? [Y/n]: ").strip().lower()
    hardware_optimization = hw_opt != 'n'
    
    # Parallel loading
    parallel = input("Enable parallel component loading? [Y/n]: ").strip().lower()
    parallel_loading = parallel != 'n'
    
    # Agent count
    try:
        agent_input = input("Number of agents [8]: ").strip()
        agent_count = int(agent_input) if agent_input else 8
    except ValueError:
        agent_count = 8
    
    # Validation
    validate_input = input("Run system validation? [Y/n]: ").strip().lower()
    validate = validate_input != 'n'
    
    # Verbose
    verbose_input = input("Verbose output? [y/N]: ").strip().lower()
    verbose = verbose_input == 'y'
    
    print(f"\n🚀 Starting BOB with configuration:")
    print(f"   Hardware optimization: {hardware_optimization}")
    print(f"   Parallel loading: {parallel_loading}")
    print(f"   Agent count: {agent_count}")
    print(f"   Validation: {validate}")
    print(f"   Verbose: {verbose}")
    print()
    
    return await main_startup(
        verbose=verbose,
        benchmark=True,
        validate=validate,
        hardware_optimization=hardware_optimization,
        parallel_loading=parallel_loading,
        agent_count=agent_count
    )


def main() -> int:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="BOB Unified System Startup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bob_startup.py                    # Standard startup
  python3 bob_startup.py --fast             # Fast startup (minimal validation)
  python3 bob_startup.py --benchmark        # Detailed performance metrics
  python3 bob_startup.py --interactive      # Interactive configuration
  python3 bob_startup.py --no-optimize      # Disable hardware optimization
  python3 bob_startup.py --agents 12        # Use 12 agents
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Enable detailed benchmarking"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast startup (skip non-critical validation)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip system validation"
    )
    
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable hardware optimization"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel component loading"
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        default=8,
        help="Number of agents to start (default: 8)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=1000.0,
        help="Maximum startup time in milliseconds (default: 1000)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive startup configuration"
    )
    
    parser.add_argument(
        "--create-script",
        action="store_true",
        help="Create a startup script and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.create_script:
        create_startup_script()
        return 0
    
    if args.interactive:
        return asyncio.run(interactive_startup())
    
    # Configure startup parameters
    validate = not args.no_validate and not args.fast
    hardware_optimization = not args.no_optimize
    parallel_loading = not args.no_parallel
    
    # Run main startup
    try:
        return asyncio.run(main_startup(
            verbose=args.verbose,
            benchmark=args.benchmark,
            validate=validate,
            hardware_optimization=hardware_optimization,
            parallel_loading=parallel_loading,
            agent_count=args.agents,
            max_startup_time_ms=args.timeout
        ))
    except KeyboardInterrupt:
        print("\n⚠️ Startup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())