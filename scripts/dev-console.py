#!/usr/bin/env python3
"""Interactive development console for Unity Wheel Trading Bot.

Provides an enhanced REPL environment with pre-loaded modules,
debugging utilities, and interactive analysis tools.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import IPython
    from IPython.terminal.embed import InteractiveShellEmbed
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Core imports
from src.config.loader import get_config
from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.data_providers.unified_provider import get_unified_provider
from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.strategy.wheel import WheelStrategy
from src.unity_wheel.math.options import black_scholes_price_validated
from src.unity_wheel.risk.analytics import RiskAnalytics
from src.unity_wheel.utils.performance_cache import get_cache_stats

import logging
logger = StructuredLogger(logging.getLogger(__name__))


class DevConsole:
    """Enhanced development console with trading bot utilities."""
    
    def __init__(self):
        self.config = None
        self.provider = None
        self.advisor = None
        self.strategy = None
        self.risk_analytics = None
        self._initialized = False
        
        # Console utilities
        self.utils = DevUtils()
        
        logger.info("dev_console_created")
    
    async def initialize(self):
        """Initialize all components asynchronously."""
        if self._initialized:
            return
            
        print("🚀 Initializing Unity Wheel Development Console...")
        
        try:
            # Load configuration
            print("📋 Loading configuration...")
            self.config = get_config()
            
            # Initialize data provider
            print("📊 Setting up data providers...")
            self.provider = get_unified_provider()
            
            # Initialize strategy components
            print("⚙️ Initializing strategy components...")
            self.advisor = WheelAdvisor()
            self.strategy = WheelStrategy(self.config)
            self.risk_analytics = RiskAnalytics(self.config)
            
            self._initialized = True
            print("✅ Console initialization complete!")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            logger.error("console_initialization_failed", extra={"error": str(e)})
    
    def show_status(self):
        """Show current system status."""
        print("\n🎯 Unity Wheel Trading Bot - Development Status")
        print("=" * 50)
        
        # Configuration status
        if self.config:
            print(f"📋 Config: ✅ Loaded ({self.config.metadata.environment})")
            print(f"   Unity ticker: {self.config.unity.ticker}")
            print(f"   Delta target: {self.config.strategy.greeks.delta_target}")
        else:
            print("📋 Config: ❌ Not loaded")
        
        # Provider status
        if self.provider:
            stats = self.provider.get_performance_stats()
            print(f"📊 Data Provider: ✅ {len(stats['registered_providers'])} providers")
            print(f"   Providers: {', '.join(stats['registered_providers'])}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
        else:
            print("📊 Data Provider: ❌ Not initialized")
        
        # Performance stats
        cache_stats = get_cache_stats()
        print(f"⚡ Performance: {cache_stats['total_hits']} cache hits")
        
        print()
    
    def quick_recommendation(self, portfolio_value: float = 100000):
        """Get a quick trading recommendation."""
        if not self._initialized:
            print("❌ Console not initialized. Run 'await console.initialize()' first.")
            return
        
        print(f"🎯 Generating recommendation for ${portfolio_value:,.0f} portfolio...")
        
        try:
            # This would normally be async, but for console convenience
            # we'll use a simple sync wrapper
            import asyncio
            loop = asyncio.get_event_loop()
            
            recommendation = loop.run_until_complete(
                self.advisor.advise_position(portfolio_value=portfolio_value)
            )
            
            print(f"✅ Recommendation: {recommendation.action}")
            if hasattr(recommendation, 'strike_price'):
                print(f"   Strike: ${recommendation.strike_price}")
            if hasattr(recommendation, 'position_size'):
                print(f"   Position size: ${recommendation.position_size:,.0f}")
            if hasattr(recommendation, 'confidence'):
                print(f"   Confidence: {recommendation.confidence:.1%}")
                
        except Exception as e:
            print(f"❌ Recommendation failed: {e}")
    
    def test_black_scholes(self, spot=35.0, strike=35.0, time_to_expiry=0.123, 
                          risk_free_rate=0.05, volatility=0.25):
        """Test Black-Scholes pricing with current parameters."""
        print(f"🧮 Testing Black-Scholes with:")
        print(f"   Spot: ${spot}, Strike: ${strike}")
        print(f"   Time: {time_to_expiry:.3f} years, Rate: {risk_free_rate:.2%}")
        print(f"   Volatility: {volatility:.1%}")
        
        for option_type in ['call', 'put']:
            result = black_scholes_price_validated(
                S=spot, K=strike, T=time_to_expiry, 
                r=risk_free_rate, sigma=volatility, 
                option_type=option_type
            )
            
            print(f"   {option_type.title()}: ${result.value:.3f} (confidence: {result.confidence:.1%})")
    
    def analyze_cache(self):
        """Analyze cache performance."""
        stats = get_cache_stats()
        
        print("\n📊 Cache Performance Analysis")
        print("=" * 30)
        print(f"Total hits: {stats['total_hits']}")
        print(f"Total misses: {stats['total_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
        
        if 'cache_details' in stats:
            print("\nCache breakdown:")
            for cache_name, details in stats['cache_details'].items():
                print(f"  {cache_name}: {details['hits']} hits, {details['size']} entries")
    
    def help(self):
        """Show available console commands."""
        print("\n🛠️ Unity Wheel Development Console Commands")
        print("=" * 45)
        print("📋 Configuration & Status:")
        print("  console.show_status()                    - Show system status")
        print("  config                                   - Access configuration")
        print("  provider                                 - Access data provider")
        print()
        print("🎯 Trading & Analysis:")
        print("  console.quick_recommendation(100000)     - Get trading recommendation")
        print("  console.test_black_scholes()             - Test options pricing")
        print("  advisor                                  - Access WheelAdvisor")
        print("  strategy                                 - Access WheelStrategy")
        print()
        print("📊 Data & Performance:")
        print("  console.analyze_cache()                  - Analyze cache performance")
        print("  utils.profile_function(func)             - Profile function performance")
        print("  utils.check_memory()                     - Check memory usage")
        print()
        print("🔧 Utilities:")
        print("  utils.validate_environment()             - Check environment setup")
        print("  utils.test_data_sources()               - Test data provider connections")
        print("  console.help()                          - Show this help")
        print()


class DevUtils:
    """Development utilities for debugging and analysis."""
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a function's performance."""
        import time
        import tracemalloc
        
        print(f"🔍 Profiling: {func.__name__}")
        
        # Memory profiling
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = (end_time - start_time) * 1000
            
            print(f"✅ Execution time: {execution_time:.2f}ms")
            print(f"📈 Memory usage: {current / 1024 / 1024:.2f} MB (peak: {peak / 1024 / 1024:.2f} MB)")
            
            return result
            
        except Exception as e:
            tracemalloc.stop()
            print(f"❌ Function failed: {e}")
            raise
    
    def check_memory(self):
        """Check current memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"💾 Memory Usage:")
        print(f"   RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"   VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        print(f"   CPU: {process.cpu_percent():.1f}%")
    
    def validate_environment(self):
        """Validate development environment setup."""
        print("🔍 Validating development environment...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        else:
            print("✅ Python version OK")
        
        # Check required environment variables
        required_vars = ['DATABENTO_API_KEY', 'FRED_API_KEY']
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")
            else:
                print(f"✅ {var} set")
        
        # Check file structure
        required_paths = [
            'src/unity_wheel',
            'config.yaml',
            'tests'
        ]
        
        base_path = Path(__file__).parent.parent
        for path in required_paths:
            if not (base_path / path).exists():
                issues.append(f"Missing path: {path}")
            else:
                print(f"✅ {path} exists")
        
        if issues:
            print(f"\n❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ Environment validation passed!")
    
    async def test_data_sources(self):
        """Test data source connections."""
        print("🔌 Testing data source connections...")
        
        try:
            provider = get_unified_provider()
            health_results = await provider.health_check_all()
            
            for provider_name, health in health_results.items():
                status = health.get('status', 'unknown')
                if status == 'healthy':
                    print(f"✅ {provider_name}: {status}")
                else:
                    print(f"❌ {provider_name}: {status}")
                    if 'error' in health:
                        print(f"   Error: {health['error']}")
        
        except Exception as e:
            print(f"❌ Data source test failed: {e}")


def create_console_namespace(console):
    """Create namespace for console with pre-loaded utilities."""
    return {
        # Console and utilities
        'console': console,
        'utils': console.utils,
        
        # Configuration
        'config': console.config,
        
        # Core components (will be None until initialized)
        'provider': console.provider,
        'advisor': console.advisor,
        'strategy': console.strategy,
        'risk_analytics': console.risk_analytics,
        
        # Common imports for convenience
        'datetime': datetime,
        'timedelta': timedelta,
        'asyncio': asyncio,
        
        # Mathematical functions
        'black_scholes': black_scholes_price_validated,
        
        # Helpful shortcuts
        'help': console.help,
        'status': console.show_status,
    }


async def main():
    """Start the interactive development console."""
    print("🌐 Unity Wheel Trading Bot - Development Console")
    print("=" * 50)
    
    # Create console
    console = DevConsole()
    await console.initialize()
    
    # Show initial status
    console.show_status()
    
    # Create namespace
    namespace = create_console_namespace(console)
    
    # Show help
    print("💡 Quick start:")
    print("   - Type 'help()' or 'console.help()' for commands")
    print("   - Type 'status()' for system status")
    print("   - All components are pre-loaded and ready!")
    print("   - Use 'await' for async operations")
    print()
    
    # Start interactive shell
    if HAS_IPYTHON:
        print("🚀 Starting IPython console...")
        shell = InteractiveShellEmbed(
            user_ns=namespace,
            banner1="Unity Wheel Development Console (IPython)",
            banner2="Type 'help()' for available commands."
        )
        shell()
    else:
        print("🚀 Starting Python console (install IPython for enhanced experience)...")
        import code
        code.interact(
            banner="Unity Wheel Development Console\nType 'help()' for available commands.",
            local=namespace
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Console session ended.")
    except Exception as e:
        print(f"\n❌ Console error: {e}")
        logger.error("dev_console_error", extra={"error": str(e)})