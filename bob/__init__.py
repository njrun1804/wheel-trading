
"""
BOB - Unified System
===================

The unified BOB system consolidates:
- Einstein semantic search (bob.search)
- BOLT 8-agent orchestration (bob.integration.bolt)
- Hardware acceleration (bob.hardware)
- Wheel trading integration

This package provides backward compatibility for legacy imports.
"""

__version__ = "2.0.0"
__all__ = []

# Load compatibility layer first
from .compatibility import ensure_compatibility
ensure_compatibility()

# Import core components with error handling
try:
    # Einstein Search Components (migrated from einstein/)
    from .search.engine import UnifiedIndex
    from .search.query_processor import QueryRouter
    __all__.extend(["UnifiedIndex", "QueryRouter"])
except ImportError:
    pass

try:
    # BOLT Integration Components (migrated from bolt/)
    from .integration.bolt.core_integration import BoltIntegration
    from .integration.bolt.optimized_integration import OptimizedBoltIntegration
    __all__.extend(["BoltIntegration", "OptimizedBoltIntegration"])
except ImportError:
    pass

try:
    # Hardware Components
    from .hardware.gpu.bolt_gpu_acceleration import BoltGPUAcceleration
    __all__.extend(["BoltGPUAcceleration"])
except ImportError:
    pass

try:
    # Performance Components  
    from .performance.bolt.benchmarks import BoltBenchmarks
    __all__.extend(["BoltBenchmarks"])
except ImportError:
    pass

# Configuration
try:
    from .config.search_config import EinsteinConfig
    __all__.extend(["EinsteinConfig"])
except ImportError:
    pass
