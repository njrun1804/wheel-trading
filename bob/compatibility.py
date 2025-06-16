"""
BOB Compatibility Layer
======================

Provides backward compatibility for import paths during migration from:
- Einstein standalone → bob.search
- BOLT standalone → bob.integration.bolt
- Legacy wheel trading imports
"""

import warnings
import sys
from pathlib import Path

# Add the project root to path for compatibility
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def deprecation_warning(old_path, new_path):
    """Issue a deprecation warning for old import paths."""
    warnings.warn(
        f"Import '{old_path}' is deprecated. Use '{new_path}' instead. "
        f"Support for old paths will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Einstein compatibility aliases
class EinsteinCompatibility:
    """Compatibility layer for Einstein imports."""
    
    def __getattr__(self, name):
        if name == 'UnifiedIndex':
            deprecation_warning('einstein.unified_index', 'bob.search.engine')
            try:
                from bob.search.engine import UnifiedIndex
                return UnifiedIndex
            except ImportError:
                # Fallback to mock
                class MockUnifiedIndex:
                    def search(self, query, **kwargs):
                        return []
                return MockUnifiedIndex
        
        elif name == 'QueryRouter':
            deprecation_warning('einstein.query_router', 'bob.search.query_processor')
            try:
                from bob.search.query_processor import QueryRouter
                return QueryRouter
            except ImportError:
                class MockQueryRouter:
                    def route(self, query, **kwargs):
                        return {"type": "fallback", "query": query}
                return MockQueryRouter
        
        elif name == 'EinsteinConfig':
            deprecation_warning('einstein.einstein_config', 'bob.config.search_config')
            try:
                from bob.config.search_config import EinsteinConfig
                return EinsteinConfig
            except ImportError:
                class MockEinsteinConfig:
                    def __init__(self):
                        self.max_search_ms = 100
                return MockEinsteinConfig
        
        raise AttributeError(f"module 'einstein' has no attribute '{name}'")

# BOLT compatibility aliases
class BoltCompatibility:
    """Compatibility layer for BOLT imports."""
    
    def __getattr__(self, name):
        if name == 'BoltIntegration':
            deprecation_warning('bolt.core.integration', 'bob.integration.bolt.core_integration')
            try:
                from bob.integration.bolt.core_integration import BoltIntegration
                return BoltIntegration
            except ImportError:
                class MockBoltIntegration:
                    def solve(self, problem):
                        return f"Mock solution for: {problem}"
                    def analyze(self, problem):
                        return f"Mock analysis for: {problem}"
                return MockBoltIntegration
        
        elif name == 'AgentOrchestrator':
            deprecation_warning('bolt.agents.orchestrator', 'bob.agents.orchestrator')
            try:
                from bob.agents.orchestrator import AgentOrchestrator
                return AgentOrchestrator
            except ImportError:
                class MockAgentOrchestrator:
                    def __init__(self, num_agents=8):
                        self.num_agents = num_agents
                    def solve(self, problem):
                        return f"Mock solution using {self.num_agents} agents: {problem}"
                return MockAgentOrchestrator
        
        raise AttributeError(f"module 'bolt' has no attribute '{name}'")

# Install compatibility modules
sys.modules['einstein'] = EinsteinCompatibility()
sys.modules['bolt'] = BoltCompatibility()

# Additional compatibility for specific submodules
sys.modules['einstein.unified_index'] = EinsteinCompatibility()
sys.modules['einstein.query_router'] = EinsteinCompatibility()
sys.modules['bolt.core'] = BoltCompatibility()
sys.modules['bolt.agents'] = BoltCompatibility()

# Unity wheel imports should work as-is since they haven't moved
# Just ensure the path is available
unity_wheel_path = project_root / "src"
if str(unity_wheel_path) not in sys.path:
    sys.path.insert(0, str(unity_wheel_path))

def ensure_compatibility():
    """Ensure all compatibility layers are loaded."""
    # This function can be called to ensure compatibility is initialized
    pass

if __name__ == "__main__":
    print("BOB Compatibility Layer loaded")
    print("- Einstein imports → bob.search.*")
    print("- BOLT imports → bob.integration.bolt.*")
    print("- Deprecation warnings enabled")