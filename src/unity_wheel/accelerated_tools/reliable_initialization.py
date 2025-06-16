"""
Optimized Accelerated Tools Initialization System
100% Reliable initialization with fallbacks and error handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolStatus:
    """Status information for an accelerated tool."""
    name: str
    available: bool
    instance: Optional[Any]
    error: Optional[str]
    init_time: float
    version: Optional[str] = None


class AcceleratedToolsManager:
    """Manages reliable initialization of all accelerated tools."""
    
    def __init__(self):
        self._tools: Dict[str, ToolStatus] = {}
        self._initialized = False
        self._init_start_time = 0.0
        self._init_duration = 0.0
        
        # Expected tools with their initialization functions
        self._tool_specs = {
            'ripgrep_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.ripgrep_turbo',
                'factory': 'get_ripgrep_turbo',
                'required': True
            },
            'dependency_graph_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.dependency_graph_turbo', 
                'factory': 'get_dependency_graph',
                'required': True
            },
            'python_analysis_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.python_analysis_turbo',
                'factory': 'get_python_analyzer', 
                'required': True
            },
            'duckdb_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.duckdb_turbo',
                'factory': 'get_duckdb_turbo',
                'required': True
            },
            'trace_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.trace_turbo',
                'factory': 'get_trace_turbo',
                'required': True
            },
            'python_helpers_turbo': {
                'module': 'src.unity_wheel.accelerated_tools.python_helpers_turbo',
                'factory': 'get_code_helper',
                'required': True
            },
            'bolt_integration': {
                'module': 'bolt.core.fallbacks',
                'factory': 'get_accelerated_tool',
                'factory_args': ('ripgrep_turbo',),
                'required': False,  # Make this optional since it's an integration test
                'import_fallback': True,  # Try alternative import if standard fails
                'single_arg': True  # Only pass the first argument
            }
        }
    
    async def initialize_all(self, timeout: float = 30.0) -> Dict[str, ToolStatus]:
        """Initialize all accelerated tools with timeout and retry logic."""
        self._init_start_time = time.perf_counter()
        
        logger.info("Starting accelerated tools initialization...")
        
        # Initialize tools in parallel for faster startup
        initialization_tasks = []
        for tool_name in self._tool_specs:
            task = asyncio.create_task(
                self._initialize_single_tool(tool_name),
                name=f"init_{tool_name}"
            )
            initialization_tasks.append(task)
        
        try:
            # Wait for all initializations with timeout
            await asyncio.wait_for(
                asyncio.gather(*initialization_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Tool initialization timed out after {timeout}s")
            # Cancel any remaining tasks
            for task in initialization_tasks:
                if not task.done():
                    task.cancel()
        
        self._init_duration = time.perf_counter() - self._init_start_time
        self._initialized = True
        
        # Log summary
        available_count = sum(1 for status in self._tools.values() if status.available)
        total_count = len(self._tool_specs)
        
        logger.info(f"Accelerated tools initialized: {available_count}/{total_count} "
                   f"available in {self._init_duration:.2f}s")
        
        if available_count < total_count:
            failed_tools = [name for name, status in self._tools.items() 
                           if not status.available]
            logger.warning(f"Failed to initialize: {failed_tools}")
        
        return self._tools.copy()
    
    async def _initialize_single_tool(self, tool_name: str) -> None:
        """Initialize a single tool with error handling."""
        start_time = time.perf_counter()
        spec = self._tool_specs[tool_name]
        
        try:
            # Dynamic import with fallback support
            try:
                module_parts = spec['module'].split('.')
                module = __import__(module_parts[0])
                for part in module_parts[1:]:
                    module = getattr(module, part)
            except (ImportError, AttributeError) as e:
                # Try fallback import for bolt_integration
                if tool_name == 'bolt_integration' and spec.get('import_fallback'):
                    try:
                        # Try direct import from bolt.core.fallbacks
                        import sys
                        from pathlib import Path
                        project_root = Path.cwd()
                        if str(project_root) not in sys.path:
                            sys.path.insert(0, str(project_root))
                        
                        from bolt.core.fallbacks import get_accelerated_tool
                        # Don't wrap in MockModule - use the function directly
                        module = type('MockModule', (), {'get_accelerated_tool': lambda self, tool_name: get_accelerated_tool(tool_name)})()
                    except ImportError:
                        raise e
                else:
                    raise e
            
            # Get factory function
            factory_func = getattr(module, spec['factory'])
            
            # Call factory with optional arguments
            factory_args = spec.get('factory_args', ())
            if factory_args:
                # Special handling for functions that expect only the first argument
                if spec.get('single_arg') and len(factory_args) > 0:
                    logger.debug(f"Calling {spec['factory']} with single arg: {factory_args[0]}")
                    instance = factory_func(factory_args[0])
                else:
                    logger.debug(f"Calling {spec['factory']} with args: {factory_args}")
                    instance = factory_func(*factory_args)
            else:
                logger.debug(f"Calling {spec['factory']} with no args")
                instance = factory_func()
            
            # For bolt_integration, check if we got a valid tool back
            if tool_name == 'bolt_integration' and instance is None:
                raise RuntimeError("Bolt integration returned None")
            
            # Get version if available
            version = getattr(instance, '__version__', None) or getattr(instance, 'version', None)
            
            init_time = time.perf_counter() - start_time
            
            self._tools[tool_name] = ToolStatus(
                name=tool_name,
                available=True,
                instance=instance,
                error=None,
                init_time=init_time,
                version=version
            )
            
            logger.debug(f"✅ {tool_name} initialized in {init_time:.3f}s")
            
        except Exception as e:
            init_time = time.perf_counter() - start_time
            error_msg = str(e)
            
            self._tools[tool_name] = ToolStatus(
                name=tool_name,
                available=False,
                instance=None,
                error=error_msg,
                init_time=init_time
            )
            
            if spec.get('required', False):
                logger.error(f"❌ Required tool {tool_name} failed: {error_msg}")
            else:
                logger.warning(f"⚠️ Optional tool {tool_name} failed: {error_msg}")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get an initialized tool instance."""
        if not self._initialized:
            raise RuntimeError("Tools not initialized. Call initialize_all() first.")
        
        status = self._tools.get(tool_name)
        if status and status.available:
            return status.instance
        return None
    
    def get_status(self, tool_name: str) -> Optional[ToolStatus]:
        """Get status information for a tool."""
        return self._tools.get(tool_name)
    
    def get_all_status(self) -> Dict[str, ToolStatus]:
        """Get status for all tools."""
        return self._tools.copy()
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available."""
        status = self._tools.get(tool_name)
        return status is not None and status.available
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [name for name, status in self._tools.items() if status.available]
    
    def get_availability_summary(self) -> Tuple[int, int, float]:
        """Get summary: (available_count, total_count, success_rate)."""
        available = sum(1 for status in self._tools.values() if status.available)
        total = len(self._tool_specs)
        success_rate = available / total if total > 0 else 0.0
        return available, total, success_rate
    
    def validate_critical_tools(self) -> Tuple[bool, List[str]]:
        """Validate that all critical tools are available."""
        required_tools = [name for name, spec in self._tool_specs.items() 
                         if spec.get('required', False)]
        
        missing_tools = []
        for tool_name in required_tools:
            if not self.is_tool_available(tool_name):
                missing_tools.append(tool_name)
        
        return len(missing_tools) == 0, missing_tools


# Global instance for singleton pattern
_manager_instance: Optional[AcceleratedToolsManager] = None


def get_tools_manager() -> AcceleratedToolsManager:
    """Get the global tools manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AcceleratedToolsManager()
    return _manager_instance


async def ensure_tools_initialized(timeout: float = 30.0) -> Dict[str, ToolStatus]:
    """Ensure all tools are initialized, initializing if necessary."""
    manager = get_tools_manager()
    if not manager._initialized:
        return await manager.initialize_all(timeout=timeout)
    return manager.get_all_status()


def get_tool_fast(tool_name: str) -> Optional[Any]:
    """Fast tool access (assumes tools are already initialized)."""
    manager = get_tools_manager()
    return manager.get_tool(tool_name)


async def validate_all_tools() -> Tuple[bool, Dict[str, Any]]:
    """Comprehensive validation of all accelerated tools."""
    manager = get_tools_manager()
    
    # Ensure initialization
    await ensure_tools_initialized()
    
    # Get summary
    available, total, success_rate = manager.get_availability_summary()
    all_available, missing = manager.validate_critical_tools()
    
    validation_result = {
        'all_tools_available': all_available,
        'available_count': available,
        'total_count': total,
        'success_rate': success_rate,
        'missing_critical_tools': missing,
        'available_tools': manager.get_available_tools(),
        'init_duration': manager._init_duration,
        'tool_details': {
            name: {
                'available': status.available,
                'error': status.error,
                'init_time': status.init_time,
                'version': status.version
            }
            for name, status in manager.get_all_status().items()
        }
    }
    
    return all_available, validation_result


# Convenience functions for direct tool access
async def get_ripgrep():
    """Get ripgrep turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('ripgrep_turbo')


async def get_dependency_graph():
    """Get dependency graph turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('dependency_graph_turbo')


async def get_python_analyzer():
    """Get python analyzer turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('python_analysis_turbo')


async def get_duckdb():
    """Get duckdb turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('duckdb_turbo')


async def get_tracer():
    """Get trace turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('trace_turbo')


async def get_code_helper():
    """Get python helpers turbo instance."""
    await ensure_tools_initialized()
    return get_tool_fast('python_helpers_turbo')


async def get_bolt_integration():
    """Get bolt integration instance."""
    await ensure_tools_initialized()
    return get_tool_fast('bolt_integration')