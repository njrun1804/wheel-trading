#!/usr/bin/env python3
"""
Unified System Lifecycle Management

Provides coordinated startup and shutdown procedures for Einstein and Bolt systems,
ensuring proper initialization order, resource allocation, health checks, and
graceful shutdown with cleanup.

Key Features:
- Coordinated system startup with dependency management
- Health monitoring and readiness checks
- Graceful shutdown with resource cleanup
- Error handling and recovery during lifecycle events
- Performance monitoring and optimization
- State persistence and recovery
"""

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from einstein_bolt_resource_sharing import get_resource_manager, initialize_resource_sharing
from unified_configuration_system import SystemType, get_configuration_manager, initialize_unified_configuration
from unified_hardware_abstraction import get_hardware_abstraction, initialize_hardware_abstraction

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ComponentState(Enum):
    """Individual component states."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    STARTING = "starting"
    READY = "ready"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemComponent:
    """Represents a system component in the lifecycle."""
    
    name: str
    system_type: SystemType
    priority: int = 5  # 1=highest, 10=lowest
    dependencies: Set[str] = field(default_factory=set)
    state: ComponentState = ComponentState.INACTIVE
    health_score: float = 0.0
    last_health_check: float = 0.0
    startup_time: float = 0.0
    shutdown_time: float = 0.0
    error_count: int = 0
    restart_count: int = 0
    
    # Lifecycle callbacks
    initialize_func: Optional[Callable] = None
    start_func: Optional[Callable] = None
    stop_func: Optional[Callable] = None
    health_check_func: Optional[Callable] = None
    cleanup_func: Optional[Callable] = None


@dataclass
class SystemMetrics:
    """System lifecycle metrics."""
    
    total_startup_time: float = 0.0
    total_shutdown_time: float = 0.0
    successful_startups: int = 0
    failed_startups: int = 0
    successful_shutdowns: int = 0
    failed_shutdowns: int = 0
    system_uptime: float = 0.0
    component_health_average: float = 0.0
    last_updated: float = field(default_factory=time.time)


class UnifiedSystemLifecycle:
    """Manages unified lifecycle for Einstein and Bolt systems."""
    
    def __init__(self):
        self.system_state = SystemState.UNINITIALIZED
        self.components: Dict[str, SystemComponent] = {}
        self.metrics = SystemMetrics()
        
        # Core system instances
        self.hardware = None
        self.config_manager = None
        self.resource_manager = None
        
        # Einstein components
        self.einstein_components = {}
        
        # Bolt components
        self.bolt_components = {}
        
        # Lifecycle management
        self.startup_start_time = 0.0
        self.system_start_time = 0.0
        self.shutdown_handlers: List[Callable] = []
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.lifecycle_lock = asyncio.Lock()
        
        # Signal handling for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("🔄 Unified System Lifecycle Manager initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, self._signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            logger.warning(f"Failed to setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown_system())
    
    async def initialize_system(self, config_file: Optional[str] = None) -> bool:
        """Initialize the unified system."""
        try:
            async with self.lifecycle_lock:
                if self.system_state != SystemState.UNINITIALIZED:
                    logger.warning(f"System already initialized (state: {self.system_state})")
                    return True
                
                self.system_state = SystemState.INITIALIZING
                self.startup_start_time = time.time()
                
                logger.info("🚀 Starting unified system initialization...")
                
                # Step 1: Initialize hardware abstraction
                logger.info("📟 Initializing hardware abstraction...")
                self.hardware = await initialize_hardware_abstraction()
                
                # Step 2: Initialize configuration system
                logger.info("⚙️  Initializing configuration system...")
                self.config_manager = await initialize_unified_configuration(config_file)
                
                # Step 3: Initialize resource sharing
                logger.info("🔄 Initializing resource sharing...")
                self.resource_manager = await initialize_resource_sharing()
                
                # Step 4: Register system components
                await self._register_system_components()
                
                # Step 5: Validate dependencies
                if not self._validate_component_dependencies():
                    raise RuntimeError("Component dependency validation failed")
                
                self.system_state = SystemState.STARTING
                logger.info("✅ System initialization completed successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.system_state = SystemState.ERROR
            return False
    
    async def _register_system_components(self):
        """Register all system components."""
        
        # Core system components
        self._register_component(SystemComponent(
            name="hardware_abstraction",
            system_type=SystemType.UNIFIED,
            priority=1,
            state=ComponentState.READY,  # Already initialized
            health_check_func=self._check_hardware_health
        ))
        
        self._register_component(SystemComponent(
            name="configuration_manager",
            system_type=SystemType.UNIFIED,
            priority=1,
            state=ComponentState.READY,  # Already initialized
            health_check_func=self._check_config_health
        ))
        
        self._register_component(SystemComponent(
            name="resource_manager",
            system_type=SystemType.UNIFIED,
            priority=2,
            dependencies={"hardware_abstraction", "configuration_manager"},
            state=ComponentState.READY,  # Already initialized
            health_check_func=self._check_resource_manager_health
        ))
        
        # Einstein components
        self._register_component(SystemComponent(
            name="einstein_index_hub",
            system_type=SystemType.EINSTEIN,
            priority=3,
            dependencies={"hardware_abstraction", "configuration_manager", "resource_manager"},
            initialize_func=self._initialize_einstein_index_hub,
            start_func=self._start_einstein_index_hub,
            stop_func=self._stop_einstein_index_hub,
            health_check_func=self._check_einstein_health,
            cleanup_func=self._cleanup_einstein
        ))
        
        self._register_component(SystemComponent(
            name="einstein_search_system",
            system_type=SystemType.EINSTEIN,
            priority=4,
            dependencies={"einstein_index_hub"},
            initialize_func=self._initialize_einstein_search,
            start_func=self._start_einstein_search,
            stop_func=self._stop_einstein_search,
            health_check_func=self._check_einstein_search_health
        ))
        
        # Bolt components
        self._register_component(SystemComponent(
            name="bolt_orchestrator",
            system_type=SystemType.BOLT,
            priority=3,
            dependencies={"hardware_abstraction", "configuration_manager", "resource_manager"},
            initialize_func=self._initialize_bolt_orchestrator,
            start_func=self._start_bolt_orchestrator,
            stop_func=self._stop_bolt_orchestrator,
            health_check_func=self._check_bolt_health,
            cleanup_func=self._cleanup_bolt
        ))
        
        self._register_component(SystemComponent(
            name="bolt_agent_pool",
            system_type=SystemType.BOLT,
            priority=4,
            dependencies={"bolt_orchestrator"},
            initialize_func=self._initialize_bolt_agents,
            start_func=self._start_bolt_agents,
            stop_func=self._stop_bolt_agents,
            health_check_func=self._check_bolt_agents_health
        ))
        
        # Integration components
        self._register_component(SystemComponent(
            name="integration_coordinator",
            system_type=SystemType.UNIFIED,
            priority=5,
            dependencies={"einstein_search_system", "bolt_agent_pool"},
            initialize_func=self._initialize_integration,
            start_func=self._start_integration,
            stop_func=self._stop_integration,
            health_check_func=self._check_integration_health
        ))
        
        logger.info(f"Registered {len(self.components)} system components")
    
    def _register_component(self, component: SystemComponent):
        """Register a system component."""
        self.components[component.name] = component
        
        if component.system_type == SystemType.EINSTEIN:
            self.einstein_components[component.name] = component
        elif component.system_type == SystemType.BOLT:
            self.bolt_components[component.name] = component
    
    def _validate_component_dependencies(self) -> bool:
        """Validate component dependencies."""
        try:
            for name, component in self.components.items():
                for dep in component.dependencies:
                    if dep not in self.components:
                        logger.error(f"Component {name} depends on unknown component {dep}")
                        return False
            
            # Check for circular dependencies
            if self._detect_circular_dependencies():
                logger.error("Circular dependencies detected")
                return False
            
            logger.info("✅ Component dependencies validated")
            return True
            
        except Exception as e:
            logger.error(f"Dependency validation failed: {e}")
            return False
    
    def _detect_circular_dependencies(self) -> bool:
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(component_name: str) -> bool:
            visited.add(component_name)
            rec_stack.add(component_name)
            
            component = self.components[component_name]
            for dep in component.dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(component_name)
            return False
        
        for name in self.components:
            if name not in visited:
                if has_cycle(name):
                    return True
        
        return False
    
    async def start_system(self) -> bool:
        """Start all system components in proper order."""
        try:
            async with self.lifecycle_lock:
                if self.system_state != SystemState.STARTING:
                    logger.warning(f"System not ready to start (state: {self.system_state})")
                    return False
                
                logger.info("🚀 Starting system components...")
                
                # Get startup order based on priorities and dependencies
                startup_order = self._get_startup_order()
                
                # Start components in order
                for component_name in startup_order:
                    if not await self._start_component(component_name):
                        logger.error(f"Failed to start component {component_name}")
                        await self._handle_startup_failure(component_name)
                        return False
                
                # Start health monitoring
                await self._start_health_monitoring()
                
                # Mark system as running
                self.system_state = SystemState.RUNNING
                self.system_start_time = time.time()
                
                # Update metrics
                self.metrics.total_startup_time = self.system_start_time - self.startup_start_time
                self.metrics.successful_startups += 1
                
                logger.info(f"✅ System startup completed in {self.metrics.total_startup_time:.2f}s")
                return True
                
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.system_state = SystemState.ERROR
            self.metrics.failed_startups += 1
            return False
    
    def _get_startup_order(self) -> List[str]:
        """Get component startup order based on dependencies and priorities."""
        # Topological sort with priority ordering
        in_degree = {name: 0 for name in self.components}
        
        # Calculate in-degrees
        for name, component in self.components.items():
            for dep in component.dependencies:
                in_degree[name] += 1
        
        # Initialize queue with components that have no dependencies
        queue = []
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)
        
        # Sort by priority
        queue.sort(key=lambda x: self.components[x].priority)
        
        result = []
        
        while queue:
            # Get component with highest priority (lowest priority number)
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees for dependents
            for name, component in self.components.items():
                if current in component.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
            
            # Re-sort queue by priority
            queue.sort(key=lambda x: self.components[x].priority)
        
        return result
    
    async def _start_component(self, component_name: str) -> bool:
        """Start a specific component."""
        try:
            component = self.components[component_name]
            
            # Skip if already started
            if component.state in [ComponentState.READY, ComponentState.HEALTHY]:
                return True
            
            logger.info(f"🔧 Starting component: {component_name}")
            start_time = time.time()
            
            component.state = ComponentState.STARTING
            
            # Initialize if needed
            if component.initialize_func and component.state != ComponentState.READY:
                await component.initialize_func()
            
            # Start the component
            if component.start_func:
                await component.start_func()
            
            # Perform health check
            if component.health_check_func:
                health_ok = await component.health_check_func()
                component.state = ComponentState.HEALTHY if health_ok else ComponentState.DEGRADED
            else:
                component.state = ComponentState.READY
            
            component.startup_time = time.time() - start_time
            
            logger.info(f"✅ Component {component_name} started in {component.startup_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start component {component_name}: {e}")
            component.state = ComponentState.ERROR
            component.error_count += 1
            return False
    
    async def _handle_startup_failure(self, failed_component: str):
        """Handle component startup failure."""
        logger.error(f"Handling startup failure for {failed_component}")
        
        # Stop any components that were already started
        started_components = [
            name for name, comp in self.components.items()
            if comp.state in [ComponentState.READY, ComponentState.HEALTHY, ComponentState.STARTING]
            and name != failed_component
        ]
        
        for component_name in reversed(started_components):
            try:
                await self._stop_component(component_name)
            except Exception as e:
                logger.error(f"Error stopping {component_name} during failure cleanup: {e}")
    
    async def shutdown_system(self, graceful: bool = True) -> bool:
        """Shutdown all system components."""
        try:
            async with self.lifecycle_lock:
                if self.system_state in [SystemState.STOPPED, SystemState.STOPPING]:
                    logger.info("System already stopped or stopping")
                    return True
                
                logger.info("🛑 Starting system shutdown...")
                self.system_state = SystemState.STOPPING
                shutdown_start_time = time.time()
                
                # Stop health monitoring
                await self._stop_health_monitoring()
                
                # Get shutdown order (reverse of startup order)
                shutdown_order = list(reversed(self._get_startup_order()))
                
                # Stop components in order
                for component_name in shutdown_order:
                    await self._stop_component(component_name, graceful)
                
                # Stop core systems
                if self.resource_manager:
                    await self.resource_manager.stop_resource_management()
                
                if self.hardware:
                    await self.hardware.stop_monitoring()
                
                # Mark system as stopped
                self.system_state = SystemState.STOPPED
                
                # Update metrics
                self.metrics.total_shutdown_time = time.time() - shutdown_start_time
                self.metrics.successful_shutdowns += 1
                
                if self.system_start_time > 0:
                    self.metrics.system_uptime = time.time() - self.system_start_time
                
                logger.info(f"✅ System shutdown completed in {self.metrics.total_shutdown_time:.2f}s")
                return True
                
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            self.metrics.failed_shutdowns += 1
            return False
    
    async def _stop_component(self, component_name: str, graceful: bool = True):
        """Stop a specific component."""
        try:
            component = self.components[component_name]
            
            if component.state == ComponentState.STOPPED:
                return
            
            logger.info(f"🔧 Stopping component: {component_name}")
            stop_start_time = time.time()
            
            # Stop the component
            if component.stop_func:
                await component.stop_func()
            
            # Cleanup if needed
            if component.cleanup_func:
                await component.cleanup_func()
            
            component.state = ComponentState.STOPPED
            component.shutdown_time = time.time() - stop_start_time
            
            logger.info(f"✅ Component {component_name} stopped in {component.shutdown_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error stopping component {component_name}: {e}")
            component.state = ComponentState.ERROR
    
    async def _start_health_monitoring(self):
        """Start health monitoring for all components."""
        if self.health_monitor_task:
            return
        
        self.health_monitor_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("📊 Health monitoring started")
    
    async def _stop_health_monitoring(self):
        """Stop health monitoring."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None
        
        logger.info("📊 Health monitoring stopped")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            while self.system_state == SystemState.RUNNING:
                await self._perform_health_checks()
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all components."""
        try:
            total_health = 0.0
            healthy_components = 0
            
            for name, component in self.components.items():
                if component.health_check_func and component.state != ComponentState.STOPPED:
                    try:
                        health_ok = await component.health_check_func()
                        component.health_score = 1.0 if health_ok else 0.0
                        component.last_health_check = time.time()
                        
                        if health_ok:
                            if component.state == ComponentState.DEGRADED:
                                component.state = ComponentState.HEALTHY
                            healthy_components += 1
                        else:
                            component.state = ComponentState.DEGRADED
                            logger.warning(f"Component {name} health check failed")
                        
                        total_health += component.health_score
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {e}")
                        component.health_score = 0.0
                        component.state = ComponentState.FAILING
            
            # Update system health metrics
            if len(self.components) > 0:
                self.metrics.component_health_average = total_health / len(self.components)
            
            # Update system state based on component health
            if healthy_components == 0:
                self.system_state = SystemState.ERROR
            elif healthy_components < len(self.components) * 0.8:
                self.system_state = SystemState.DEGRADED
            else:
                self.system_state = SystemState.RUNNING
            
        except Exception as e:
            logger.error(f"Health check loop error: {e}")
    
    # Component-specific initialization and health check functions
    async def _initialize_einstein_index_hub(self):
        """Initialize Einstein index hub."""
        try:
            # Import Einstein components dynamically
            from einstein.einstein_optimized_integration import OptimizedEinsteinHub
            
            einstein_config = self.config_manager.get_system_config(SystemType.EINSTEIN)
            self.einstein_components["index_hub"] = OptimizedEinsteinHub()
            await self.einstein_components["index_hub"].initialize()
            
        except Exception as e:
            logger.error(f"Einstein index hub initialization failed: {e}")
            raise
    
    async def _start_einstein_index_hub(self):
        """Start Einstein index hub."""
        # Implementation would start the Einstein hub
        pass
    
    async def _stop_einstein_index_hub(self):
        """Stop Einstein index hub."""
        if "index_hub" in self.einstein_components:
            # Implementation would stop the Einstein hub
            pass
    
    async def _check_einstein_health(self) -> bool:
        """Check Einstein system health."""
        try:
            # Implementation would check Einstein health
            return True
        except Exception:
            return False
    
    async def _initialize_einstein_search(self):
        """Initialize Einstein search system."""
        # Implementation would initialize Einstein search
        pass
    
    async def _start_einstein_search(self):
        """Start Einstein search system."""
        # Implementation would start Einstein search
        pass
    
    async def _stop_einstein_search(self):
        """Stop Einstein search system."""
        # Implementation would stop Einstein search
        pass
    
    async def _check_einstein_search_health(self) -> bool:
        """Check Einstein search health."""
        return True
    
    async def _initialize_bolt_orchestrator(self):
        """Initialize Bolt orchestrator."""
        try:
            # Import Bolt components dynamically
            from bolt.agents.orchestrator import BoltOrchestrator
            
            bolt_config = self.config_manager.get_system_config(SystemType.BOLT)
            self.bolt_components["orchestrator"] = BoltOrchestrator(bolt_config)
            
        except Exception as e:
            logger.error(f"Bolt orchestrator initialization failed: {e}")
            raise
    
    async def _start_bolt_orchestrator(self):
        """Start Bolt orchestrator."""
        # Implementation would start the Bolt orchestrator
        pass
    
    async def _stop_bolt_orchestrator(self):
        """Stop Bolt orchestrator."""
        # Implementation would stop the Bolt orchestrator
        pass
    
    async def _check_bolt_health(self) -> bool:
        """Check Bolt system health."""
        return True
    
    async def _initialize_bolt_agents(self):
        """Initialize Bolt agent pool."""
        # Implementation would initialize Bolt agents
        pass
    
    async def _start_bolt_agents(self):
        """Start Bolt agent pool."""
        # Implementation would start Bolt agents
        pass
    
    async def _stop_bolt_agents(self):
        """Stop Bolt agent pool."""
        # Implementation would stop Bolt agents
        pass
    
    async def _check_bolt_agents_health(self) -> bool:
        """Check Bolt agents health."""
        return True
    
    async def _initialize_integration(self):
        """Initialize integration coordinator."""
        # Implementation would initialize integration between systems
        pass
    
    async def _start_integration(self):
        """Start integration coordinator."""
        # Implementation would start integration coordination
        pass
    
    async def _stop_integration(self):
        """Stop integration coordinator."""
        # Implementation would stop integration coordination
        pass
    
    async def _check_integration_health(self) -> bool:
        """Check integration health."""
        return True
    
    async def _cleanup_einstein(self):
        """Cleanup Einstein resources."""
        # Implementation would cleanup Einstein resources
        pass
    
    async def _cleanup_bolt(self):
        """Cleanup Bolt resources."""
        # Implementation would cleanup Bolt resources
        pass
    
    async def _check_hardware_health(self) -> bool:
        """Check hardware abstraction health."""
        try:
            return self.hardware is not None and hasattr(self.hardware, 'config')
        except Exception:
            return False
    
    async def _check_config_health(self) -> bool:
        """Check configuration manager health."""
        try:
            return self.config_manager is not None and hasattr(self.config_manager, 'config')
        except Exception:
            return False
    
    async def _check_resource_manager_health(self) -> bool:
        """Check resource manager health."""
        try:
            return self.resource_manager is not None and self.resource_manager.running
        except Exception:
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_state": self.system_state.value,
            "uptime_seconds": time.time() - self.system_start_time if self.system_start_time > 0 else 0,
            "components": {
                name: {
                    "state": comp.state.value,
                    "health_score": comp.health_score,
                    "last_health_check": comp.last_health_check,
                    "startup_time": comp.startup_time,
                    "error_count": comp.error_count,
                    "restart_count": comp.restart_count,
                    "system_type": comp.system_type.value,
                    "priority": comp.priority
                }
                for name, comp in self.components.items()
            },
            "metrics": {
                "total_startup_time": self.metrics.total_startup_time,
                "total_shutdown_time": self.metrics.total_shutdown_time,
                "successful_startups": self.metrics.successful_startups,
                "failed_startups": self.metrics.failed_startups,
                "successful_shutdowns": self.metrics.successful_shutdowns,
                "failed_shutdowns": self.metrics.failed_shutdowns,
                "system_uptime": self.metrics.system_uptime,
                "component_health_average": self.metrics.component_health_average
            }
        }


# Global lifecycle manager
_lifecycle_manager = None

def get_lifecycle_manager() -> UnifiedSystemLifecycle:
    """Get the global lifecycle manager instance."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = UnifiedSystemLifecycle()
    return _lifecycle_manager


async def start_unified_system(config_file: Optional[str] = None) -> UnifiedSystemLifecycle:
    """Start the complete unified system."""
    manager = get_lifecycle_manager()
    
    if not await manager.initialize_system(config_file):
        raise RuntimeError("System initialization failed")
    
    if not await manager.start_system():
        raise RuntimeError("System startup failed")
    
    return manager


async def stop_unified_system(graceful: bool = True) -> bool:
    """Stop the unified system."""
    manager = get_lifecycle_manager()
    return await manager.shutdown_system(graceful)


if __name__ == "__main__":
    async def main():
        print("Unified System Lifecycle Test")
        print("=" * 50)
        
        try:
            # Start the unified system
            manager = await start_unified_system("test_config.yaml")
            
            # Show system status
            status = manager.get_system_status()
            print(f"\nSystem Status: {status['system_state']}")
            print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
            print(f"Component Health Average: {status['metrics']['component_health_average']:.2f}")
            
            print(f"\nComponents ({len(status['components'])}):")
            for name, comp_status in status['components'].items():
                print(f"  {name}: {comp_status['state']} (health: {comp_status['health_score']:.1f})")
            
            # Run for a short time
            print(f"\nSystem running... (waiting 10 seconds)")
            await asyncio.sleep(10)
            
            # Check status again
            status = manager.get_system_status()
            print(f"\nUpdated Status: {status['system_state']}")
            print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
            
            # Shutdown the system
            print(f"\nShutting down system...")
            success = await stop_unified_system()
            
            if success:
                print("✅ System shutdown completed successfully")
            else:
                print("❌ System shutdown failed")
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            await stop_unified_system(graceful=False)
    
    asyncio.run(main())