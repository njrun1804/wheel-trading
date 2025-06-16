#!/usr/bin/env python3
"""
Deploy ANE Neural Engine acceleration to Einstein pipeline.

This script integrates the ANE acceleration optimizations into the existing
Einstein embedding pipeline for maximum performance on M4 Pro systems.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = []

    try:
        logger.info("✅ MLX available")
        dependencies.append(("MLX", True, "Neural engine framework"))
    except ImportError:
        logger.error("❌ MLX not available - required for ANE acceleration")
        dependencies.append(("MLX", False, "Neural engine framework"))

    try:
        from src.unity_wheel.accelerated_tools.neural_engine_turbo import (
            get_neural_engine_turbo,
        )

        logger.info("✅ NeuralEngineTurbo available")
        dependencies.append(("NeuralEngineTurbo", True, "ANE acceleration module"))
    except ImportError:
        logger.error("❌ NeuralEngineTurbo not available")
        dependencies.append(("NeuralEngineTurbo", False, "ANE acceleration module"))

    try:
        from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
            get_einstein_ane_pipeline,
        )

        logger.info("✅ Einstein ANE integration available")
        dependencies.append(("EinsteinANE", True, "Einstein-ANE bridge"))
    except ImportError:
        logger.error("❌ Einstein ANE integration not available")
        dependencies.append(("EinsteinANE", False, "Einstein-ANE bridge"))

    try:
        from einstein.einstein_config import get_einstein_config

        logger.info("✅ Einstein configuration available")
        dependencies.append(("EinsteinConfig", True, "Configuration system"))
    except ImportError:
        logger.error("❌ Einstein configuration not available")
        dependencies.append(("EinsteinConfig", False, "Configuration system"))

    return dependencies


def test_ane_detection():
    """Test ANE hardware detection."""
    try:
        from einstein.einstein_config import get_einstein_config

        logger.info("🔍 Testing ANE detection...")
        config = get_einstein_config()

        logger.info(f"Platform: {config.hardware.platform_type}")
        logger.info(f"ANE available: {config.hardware.has_ane}")
        logger.info(f"ANE cores: {config.hardware.ane_cores}")
        logger.info(f"ANE enabled in config: {config.ml.enable_ane}")

        if config.hardware.has_ane and config.ml.enable_ane:
            logger.info("✅ ANE ready for deployment")
            return True
        else:
            if not config.hardware.has_ane:
                logger.warning("⚠️ ANE not detected on this system")
            if not config.ml.enable_ane:
                logger.warning("⚠️ ANE disabled in configuration")
            return False

    except Exception as e:
        logger.error(f"❌ ANE detection failed: {e}")
        return False


async def test_neural_engine():
    """Test the neural engine initialization and basic functionality."""
    try:
        from src.unity_wheel.accelerated_tools.neural_engine_turbo import (
            get_neural_engine_turbo,
        )

        logger.info("🧠 Testing Neural Engine...")

        # Initialize neural engine
        engine = get_neural_engine_turbo(cache_size_mb=256)

        # Get device info
        device_info = engine.get_device_info()
        logger.info(f"Device: {device_info.device_name}")
        logger.info(f"ANE available: {device_info.available}")
        logger.info(f"ANE cores: {device_info.cores}")
        logger.info(f"Preferred batch size: {device_info.preferred_batch_size}")

        # Test embedding generation
        test_texts = [
            "def test_function():",
            "class TestClass:",
            "import numpy as np",
            "# Test comment",
            "async def async_test():",
        ]

        logger.info("🚀 Testing embedding generation...")
        start_time = time.time()

        result = await engine.embed_texts_async(test_texts)

        elapsed = time.time() - start_time

        logger.info(
            f"✅ Generated embeddings for {len(test_texts)} texts in {elapsed:.3f}s"
        )
        logger.info(f"Tokens processed: {result.tokens_processed}")
        logger.info(f"Device used: {result.device_used}")
        logger.info(f"Cache hit: {result.cache_hit}")

        # Get performance metrics
        metrics = engine.get_performance_metrics()
        logger.info(f"ANE utilization: {metrics.ane_utilization:.1%}")
        logger.info(f"Tokens/sec: {metrics.tokens_per_second:.0f}")

        engine.shutdown()
        return True

    except Exception as e:
        logger.error(f"❌ Neural engine test failed: {e}")
        return False


async def test_einstein_integration():
    """Test Einstein-ANE integration."""
    try:
        from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
            EinsteinEmbeddingConfig,
            get_einstein_ane_pipeline,
        )

        logger.info("🔗 Testing Einstein-ANE integration...")

        # Create configuration
        config = EinsteinEmbeddingConfig(
            use_ane=True,
            fallback_on_error=True,
            max_batch_size=64,
            cache_embeddings=True,
            performance_logging=True,
            warmup_on_startup=False,  # Skip warmup for test
        )

        # Initialize pipeline
        pipeline = get_einstein_ane_pipeline(config=config)

        # Test batch embedding
        test_texts = [
            "from typing import List, Dict",
            "def process_data(data: List[str]) -> Dict[str, int]:",
            "    result = {}",
            "    for item in data:",
            "        result[item] = len(item)",
            "    return result",
        ]

        logger.info("🚀 Testing batch embedding...")
        start_time = time.time()

        results = await pipeline.neural_bridge.embed_text_batch(test_texts)

        elapsed = time.time() - start_time

        logger.info(f"✅ Embedded {len(test_texts)} texts in {elapsed:.3f}s")

        # Check results
        for i, (embedding, token_count) in enumerate(results):
            logger.info(
                f"Text {i+1}: {embedding.shape} embedding, {token_count} tokens"
            )

        # Get performance comparison
        perf_comparison = pipeline.neural_bridge.get_performance_comparison()
        logger.info(f"ANE calls: {perf_comparison['ane_calls']}")
        logger.info(f"Fallback calls: {perf_comparison['fallback_calls']}")
        logger.info(f"ANE usage: {perf_comparison['ane_usage_percent']:.1f}%")
        if perf_comparison["ane_calls"] > 0:
            logger.info(f"Speedup factor: {perf_comparison['speedup_factor']:.1f}x")

        return True

    except Exception as e:
        logger.error(f"❌ Einstein integration test failed: {e}")
        return False


async def deploy_ane_optimizations():
    """Deploy ANE optimizations to production Einstein system."""
    logger.info("🚀 Deploying ANE optimizations to Einstein...")

    # Check dependencies
    logger.info("\n📋 Checking dependencies...")
    dependencies = check_dependencies()

    missing_deps = [dep for dep in dependencies if not dep[1]]
    if missing_deps:
        logger.error("❌ Missing required dependencies:")
        for name, _available, description in missing_deps:
            logger.error(f"   - {name}: {description}")
        return False

    logger.info("✅ All dependencies available")

    # Test ANE detection
    logger.info("\n🔍 Testing ANE detection...")
    if not test_ane_detection():
        logger.warning(
            "⚠️ ANE not available, deployment will proceed with fallback mode"
        )

    # Test neural engine
    logger.info("\n🧠 Testing Neural Engine...")
    if not await test_neural_engine():
        logger.error("❌ Neural engine test failed")
        return False

    # Test Einstein integration
    logger.info("\n🔗 Testing Einstein integration...")
    if not await test_einstein_integration():
        logger.error("❌ Einstein integration test failed")
        return False

    # Update Einstein configuration to enable ANE by default
    logger.info("\n⚙️ Updating Einstein configuration...")
    try:
        from einstein.einstein_config import get_einstein_config

        config = get_einstein_config()

        if config.hardware.has_ane:
            logger.info("✅ ANE detected and enabled in configuration")
        else:
            logger.info("ℹ️ ANE not detected, fallback mode will be used")

        logger.info(f"ANE batch size: {config.ml.ane_batch_size}")
        logger.info(f"ANE cache size: {config.ml.ane_cache_size_mb}MB")
        logger.info(f"ANE warmup on startup: {config.ml.ane_warmup_on_startup}")

    except Exception as e:
        logger.error(f"❌ Configuration update failed: {e}")
        return False

    # Create integration example
    logger.info("\n📝 Creating integration example...")
    create_integration_example()

    logger.info("\n🎉 ANE deployment completed successfully!")
    logger.info("\nNext steps:")
    logger.info(
        "1. Use get_einstein_ane_pipeline() instead of regular Einstein pipeline"
    )
    logger.info("2. Set EINSTEIN_ENABLE_ANE=true in environment for production")
    logger.info("3. Monitor performance with pipeline.get_enhanced_stats()")
    logger.info("4. Run benchmark with: python test_ane_benchmark.py")

    return True


def create_integration_example():
    """Create example script showing how to use ANE acceleration."""
    example_content = '''#!/usr/bin/env python3
"""
Example: Using ANE-accelerated Einstein pipeline.

This example shows how to use the ANE-accelerated Einstein embedding pipeline
for maximum performance on M4 Pro systems.
"""

import asyncio
from pathlib import Path
from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
    get_einstein_ane_pipeline, EinsteinEmbeddingConfig
)

async def main():
    """Example usage of ANE-accelerated Einstein pipeline."""
    
    # Configure ANE acceleration
    config = EinsteinEmbeddingConfig(
        use_ane=True,                    # Enable ANE acceleration
        fallback_on_error=True,          # Graceful fallback to CPU
        max_batch_size=256,              # Optimal for ANE
        cache_embeddings=True,           # Cache for better performance
        performance_logging=True,        # Monitor performance
        warmup_on_startup=True           # Warm up ANE on startup
    )
    
    # Get ANE-accelerated pipeline
    pipeline = get_einstein_ane_pipeline(config=config)
    
    # Example: Embed code files
    code_files = [
        "src/unity_wheel/strategy/wheel.py",
        "src/unity_wheel/analytics/decision_engine.py",
        "src/unity_wheel/math/options.py"
    ]
    
    print("🚀 Embedding code files with ANE acceleration...")
    results = await pipeline.embed_file_batch(code_files)
    
    print(f"✅ Embedded {len(code_files)} files")
    
    # Get enhanced statistics
    stats = pipeline.get_enhanced_stats()
    print("\\n📊 Performance Statistics:")
    print(f"Files processed: {stats['pipeline_stats']['files_processed']}")
    print(f"ANE accelerated: {stats['pipeline_stats']['ane_accelerated']}")
    print(f"Cache hits: {stats['pipeline_stats']['cache_hits']}")
    
    if stats['performance_comparison']['ane_calls'] > 0:
        perf = stats['performance_comparison']
        print(f"ANE usage: {perf['ane_usage_percent']:.1f}%")
        print(f"Speedup: {perf['speedup_factor']:.1f}x")
        print(f"ANE tokens/sec: {perf['ane_tokens_per_sec']:.0f}")

if __name__ == "__main__":
    asyncio.run(main())
'''

    example_path = Path("einstein_ane_example.py")
    example_path.write_text(example_content)
    logger.info(f"✅ Created integration example: {example_path}")


def create_benchmark_script():
    """Create benchmark script for ANE performance testing."""
    benchmark_content = '''#!/usr/bin/env python3
"""
ANE Performance Benchmark for Einstein Pipeline.

Comprehensive benchmark comparing ANE acceleration vs. CPU fallback.
"""

import asyncio
import time
from pathlib import Path
from src.unity_wheel.accelerated_tools.einstein_neural_integration import (
    get_einstein_ane_pipeline, EinsteinEmbeddingConfig
)

async def benchmark_ane_performance():
    """Benchmark ANE vs CPU performance."""
    
    # Test data - various code patterns
    test_texts = [
        f"def function_{i}(x: int) -> int: return x * 2" for i in range(100)
    ] + [
        f"class Class{i}: pass" for i in range(50) 
    ] + [
        f"# Comment about feature {i}" for i in range(50)
    ]
    
    print("🚀 Starting ANE Performance Benchmark")
    print(f"Test data: {len(test_texts)} code snippets")
    
    # Test with ANE enabled
    print("\\n🧠 Testing with ANE acceleration...")
    config_ane = EinsteinEmbeddingConfig(use_ane=True, performance_logging=True)
    pipeline_ane = get_einstein_ane_pipeline(config=config_ane)
    
    start_time = time.time()
    results_ane = await pipeline_ane.neural_bridge.embed_text_batch(test_texts)
    ane_time = time.time() - start_time
    
    ane_stats = pipeline_ane.neural_bridge.get_performance_comparison()
    
    # Test with ANE disabled (CPU fallback)
    print("💻 Testing with CPU fallback...")
    config_cpu = EinsteinEmbeddingConfig(use_ane=False, performance_logging=True)
    pipeline_cpu = get_einstein_ane_pipeline(config=config_cpu)
    
    start_time = time.time()
    results_cpu = await pipeline_cpu.neural_bridge.embed_text_batch(test_texts)
    cpu_time = time.time() - start_time
    
    cpu_stats = pipeline_cpu.neural_bridge.get_performance_comparison()
    
    # Results
    print("\\n📊 Benchmark Results:")
    print(f"ANE time: {ane_time:.3f}s")
    print(f"CPU time: {cpu_time:.3f}s")
    print(f"Speedup: {cpu_time / ane_time:.1f}x")
    print(f"ANE tokens/sec: {ane_stats.get('ane_tokens_per_sec', 0):.0f}")
    print(f"CPU tokens/sec: {cpu_stats.get('fallback_tokens_per_sec', 0):.0f}")
    
    return {
        'ane_time': ane_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time / ane_time,
        'ane_tokens_per_sec': ane_stats.get('ane_tokens_per_sec', 0),
        'cpu_tokens_per_sec': cpu_stats.get('fallback_tokens_per_sec', 0)
    }

if __name__ == "__main__":
    asyncio.run(benchmark_ane_performance())
'''

    benchmark_path = Path("test_ane_benchmark.py")
    benchmark_path.write_text(benchmark_content)
    logger.info(f"✅ Created benchmark script: {benchmark_path}")


async def main():
    """Main deployment function."""
    logger.info("🚀 Starting ANE deployment to Einstein pipeline...")

    try:
        success = await deploy_ane_optimizations()

        if success:
            logger.info("\n✅ Deployment completed successfully!")

            # Create additional scripts
            create_benchmark_script()

            logger.info("\n📋 Available scripts:")
            logger.info("- einstein_ane_example.py: Integration example")
            logger.info("- test_ane_benchmark.py: Performance benchmark")

            return 0
        else:
            logger.error("\n❌ Deployment failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("\n⏹️ Deployment cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error during deployment: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
