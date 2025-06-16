#!/usr/bin/env python3
"""
Einstein-Bolt Resource Sharing System

Implements advanced resource sharing mechanisms between Einstein and Bolt systems,
ensuring optimal resource utilization, preventing conflicts, and enabling
seamless cooperation between both systems.

Key Features:
- Dynamic resource allocation and rebalancing
- Priority-based resource scheduling
- Cross-system communication and coordination
- Resource pool management
- Performance optimization
- Conflict resolution
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from unified_hardware_abstraction import get_hardware_abstraction

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be shared."""
    CPU_CORES = "cpu_cores"
    GPU_CORES = "gpu_cores"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceRequestStatus(Enum):
    """Status of resource requests."""
    PENDING = "pending"
    ALLOCATED = "allocated"
    DENIED = "denied"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequest:
    """Resource request from a system component."""
    
    request_id: str = field(default_factory=lambda: str(uuid4()))
    component_name: str = ""
    system_type: str = ""  # "einstein" or "bolt"
    resource_type: ResourceType = ResourceType.CPU_CORES
    amount_requested: int = 0
    duration_seconds: float = 0.0
    priority: TaskPriority = TaskPriority.NORMAL
    status: ResourceRequestStatus = ResourceRequestStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    allocated_amount: int = 0
    allocation_timestamp: Optional[float] = None
    completion_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePool:
    """Resource pool for a specific resource type."""
    
    resource_type: ResourceType
    total_capacity: int
    available_capacity: int
    reserved_capacity: int = 0
    allocated_capacity: int = 0
    active_allocations: Dict[str, int] = field(default_factory=dict)
    allocation_history: List[Tuple[str, int, float]] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """Performance metrics for resource sharing."""
    
    total_requests: int = 0
    successful_allocations: int = 0
    denied_requests: int = 0
    average_allocation_time: float = 0.0
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    system_performance: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class EinsteinBoltResourceManager:
    """Advanced resource sharing manager for Einstein and Bolt systems."""
    
    def __init__(self):
        self.hardware = get_hardware_abstraction()
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        self.pending_requests: List[ResourceRequest] = []
        self.active_requests: Dict[str, ResourceRequest] = {}
        self.completed_requests: List[ResourceRequest] = []
        self.metrics = SystemMetrics()
        
        # System configurations
        self.einstein_config = {}
        self.bolt_config = {}
        
        # Resource sharing policies
        self.max_einstein_cpu_share = 0.6  # Einstein can use up to 60% of CPU
        self.max_bolt_cpu_share = 0.8      # Bolt can use up to 80% of CPU
        self.priority_weights = {
            TaskPriority.CRITICAL: 10,
            TaskPriority.HIGH: 5,
            TaskPriority.NORMAL: 3,
            TaskPriority.LOW: 2,
            TaskPriority.BACKGROUND: 1
        }
        
        # Coordination
        self.coordination_lock = asyncio.Lock()
        self.allocation_scheduler: Optional[asyncio.Task] = None
        self.metrics_collector: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("🔄 Einstein-Bolt Resource Manager initialized")
    
    async def initialize(self):
        """Initialize the resource sharing system."""
        try:
            # Get hardware information
            hardware_info = self.hardware.get_hardware_info()
            
            # Initialize resource pools
            self._initialize_resource_pools(hardware_info)
            
            # Get optimal configurations for both systems
            self.einstein_config = self.hardware.get_optimal_einstein_config()
            self.bolt_config = self.hardware.get_optimal_bolt_config()
            
            # Start background tasks
            await self.start_resource_management()
            
            logger.info("✅ Resource sharing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize resource sharing: {e}")
            raise
    
    def _initialize_resource_pools(self, hardware_info: Dict[str, Any]):
        """Initialize resource pools based on hardware capabilities."""
        # CPU cores pool
        cpu_cores = hardware_info["cpu"]["physical_cores"]
        self.resource_pools[ResourceType.CPU_CORES] = ResourcePool(
            resource_type=ResourceType.CPU_CORES,
            total_capacity=cpu_cores,
            available_capacity=cpu_cores,
            reserved_capacity=2  # Reserve 2 cores for system
        )
        
        # GPU cores pool
        gpu_cores = hardware_info["gpu"]["cores"]
        self.resource_pools[ResourceType.GPU_CORES] = ResourcePool(
            resource_type=ResourceType.GPU_CORES,
            total_capacity=gpu_cores,
            available_capacity=gpu_cores
        )
        
        # Memory pool (in MB)
        memory_mb = int(hardware_info["memory"]["total_gb"] * 1024 * 0.8)  # 80% of total
        self.resource_pools[ResourceType.MEMORY] = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=memory_mb,
            available_capacity=memory_mb,
            reserved_capacity=2048  # Reserve 2GB for system
        )
        
        logger.info(f"Resource pools initialized: {cpu_cores} CPU cores, "
                   f"{gpu_cores} GPU cores, {memory_mb}MB memory")
    
    async def request_resources(
        self,
        component_name: str,
        system_type: str,
        resources: Dict[ResourceType, int],
        duration_seconds: float = 300.0,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """Request resources for a component."""
        request_ids = []
        
        try:
            async with self.coordination_lock:
                for resource_type, amount in resources.items():
                    request = ResourceRequest(
                        component_name=component_name,
                        system_type=system_type,
                        resource_type=resource_type,
                        amount_requested=amount,
                        duration_seconds=duration_seconds,
                        priority=priority,
                        metadata=metadata or {}
                    )
                    
                    self.pending_requests.append(request)
                    request_ids.append(request.request_id)
                    
                    logger.info(f"📝 Resource request submitted: {component_name} "
                               f"({system_type}) requests {amount} {resource_type.value}")
                
                # Trigger immediate allocation attempt
                await self._process_pending_requests()
                
        except Exception as e:
            logger.error(f"Failed to submit resource request: {e}")
            
        return request_ids
    
    async def release_resources(self, request_ids: List[str]) -> bool:
        """Release allocated resources."""
        try:
            async with self.coordination_lock:
                released_count = 0
                
                for request_id in request_ids:
                    if request_id in self.active_requests:
                        request = self.active_requests[request_id]
                        
                        # Release from resource pool
                        pool = self.resource_pools[request.resource_type]
                        if request_id in pool.active_allocations:
                            allocated_amount = pool.active_allocations[request_id]
                            pool.available_capacity += allocated_amount
                            pool.allocated_capacity -= allocated_amount
                            del pool.active_allocations[request_id]
                            
                            # Update request status
                            request.status = ResourceRequestStatus.COMPLETED
                            request.completion_timestamp = time.time()
                            
                            # Move to completed requests
                            self.completed_requests.append(request)
                            del self.active_requests[request_id]
                            
                            released_count += 1
                            
                            logger.info(f"🔄 Released {allocated_amount} {request.resource_type.value} "
                                       f"from {request.component_name}")
                
                # Try to allocate pending requests with newly available resources
                if released_count > 0:
                    await self._process_pending_requests()
                
                return released_count > 0
            
        except Exception as e:
            logger.error(f"Failed to release resources: {e}")
            return False
    
    async def _process_pending_requests(self):
        """Process pending resource requests."""
        if not self.pending_requests:
            return
        
        # Sort requests by priority and timestamp
        self.pending_requests.sort(
            key=lambda r: (self.priority_weights[r.priority], r.timestamp),
            reverse=True
        )
        
        allocated_requests = []
        
        for request in self.pending_requests[:]:
            if await self._try_allocate_request(request):
                allocated_requests.append(request)
                self.pending_requests.remove(request)
        
        # Update metrics
        self.metrics.total_requests += len(allocated_requests)
        self.metrics.successful_allocations += len(allocated_requests)
    
    async def _try_allocate_request(self, request: ResourceRequest) -> bool:
        """Try to allocate resources for a request."""
        try:
            pool = self.resource_pools[request.resource_type]
            
            # Check if we have enough available capacity
            effective_available = pool.available_capacity - pool.reserved_capacity
            if effective_available < request.amount_requested:
                return False
            
            # Check system-specific allocation limits
            if not self._check_system_allocation_limits(request):
                return False
            
            # Allocate resources
            pool.available_capacity -= request.amount_requested
            pool.allocated_capacity += request.amount_requested
            pool.active_allocations[request.request_id] = request.amount_requested
            
            # Update request status
            request.status = ResourceRequestStatus.ALLOCATED
            request.allocated_amount = request.amount_requested
            request.allocation_timestamp = time.time()
            
            # Move to active requests
            self.active_requests[request.request_id] = request
            
            # Schedule automatic release if duration is specified
            if request.duration_seconds > 0:
                asyncio.create_task(
                    self._auto_release_after_duration(request.request_id, request.duration_seconds)
                )
            
            logger.info(f"✅ Allocated {request.amount_requested} {request.resource_type.value} "
                       f"to {request.component_name} ({request.system_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate request {request.request_id}: {e}")
            return False
    
    def _check_system_allocation_limits(self, request: ResourceRequest) -> bool:
        """Check if allocation respects system-specific limits."""
        if request.resource_type != ResourceType.CPU_CORES:
            return True  # Only CPU cores have system-specific limits for now
        
        pool = self.resource_pools[ResourceType.CPU_CORES]
        current_system_allocation = sum(
            allocated_amount for req_id, allocated_amount in pool.active_allocations.items()
            if self.active_requests.get(req_id, {}).system_type == request.system_type
        )
        
        total_capacity = pool.total_capacity - pool.reserved_capacity
        
        if request.system_type == "einstein":
            max_allowed = int(total_capacity * self.max_einstein_cpu_share)
        elif request.system_type == "bolt":
            max_allowed = int(total_capacity * self.max_bolt_cpu_share)
        else:
            max_allowed = total_capacity  # Unknown system gets no limit
        
        return (current_system_allocation + request.amount_requested) <= max_allowed
    
    async def _auto_release_after_duration(self, request_id: str, duration: float):
        """Automatically release resources after specified duration."""
        try:
            await asyncio.sleep(duration)
            await self.release_resources([request_id])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Auto-release failed for request {request_id}: {e}")
    
    async def start_resource_management(self):
        """Start background resource management tasks."""
        if self.running:
            return
        
        self.running = True
        
        # Start allocation scheduler
        self.allocation_scheduler = asyncio.create_task(self._allocation_scheduler_loop())
        
        # Start metrics collector
        self.metrics_collector = asyncio.create_task(self._metrics_collector_loop())
        
        logger.info("🚀 Resource management tasks started")
    
    async def stop_resource_management(self):
        """Stop background resource management tasks."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel tasks
        if self.allocation_scheduler:
            self.allocation_scheduler.cancel()
            try:
                await self.allocation_scheduler
            except asyncio.CancelledError:
                pass
        
        if self.metrics_collector:
            self.metrics_collector.cancel()
            try:
                await self.metrics_collector
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Resource management tasks stopped")
    
    async def _allocation_scheduler_loop(self):
        """Background loop for processing resource allocations."""
        try:
            while self.running:
                if self.pending_requests:
                    async with self.coordination_lock:
                        await self._process_pending_requests()
                
                await asyncio.sleep(1.0)  # Check every second
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Allocation scheduler error: {e}")
    
    async def _metrics_collector_loop(self):
        """Background loop for collecting resource metrics."""
        try:
            while self.running:
                await self._update_metrics()
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Metrics collector error: {e}")
    
    async def _update_metrics(self):
        """Update resource utilization metrics."""
        try:
            # Update resource utilization
            for resource_type, pool in self.resource_pools.items():
                if pool.total_capacity > 0:
                    utilization = pool.allocated_capacity / pool.total_capacity
                    self.metrics.resource_utilization[resource_type] = utilization
            
            # Update system performance metrics
            einstein_requests = [r for r in self.active_requests.values() if r.system_type == "einstein"]
            bolt_requests = [r for r in self.active_requests.values() if r.system_type == "bolt"]
            
            self.metrics.system_performance["einstein_active_requests"] = len(einstein_requests)
            self.metrics.system_performance["bolt_active_requests"] = len(bolt_requests)
            self.metrics.system_performance["total_active_requests"] = len(self.active_requests)
            self.metrics.system_performance["pending_requests"] = len(self.pending_requests)
            
            self.metrics.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and metrics."""
        return {
            "resource_pools": {
                resource_type.value: {
                    "total_capacity": pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "allocated_capacity": pool.allocated_capacity,
                    "reserved_capacity": pool.reserved_capacity,
                    "utilization": pool.allocated_capacity / pool.total_capacity if pool.total_capacity > 0 else 0,
                    "active_allocations": len(pool.active_allocations)
                }
                for resource_type, pool in self.resource_pools.items()
            },
            "request_status": {
                "pending": len(self.pending_requests),
                "active": len(self.active_requests),
                "completed": len(self.completed_requests)
            },
            "system_breakdown": {
                "einstein": {
                    "active_requests": len([r for r in self.active_requests.values() if r.system_type == "einstein"]),
                    "cpu_allocated": sum(
                        pool.active_allocations.get(req_id, 0)
                        for req_id, req in self.active_requests.items()
                        if req.system_type == "einstein" and req.resource_type == ResourceType.CPU_CORES
                        for pool in [self.resource_pools[ResourceType.CPU_CORES]]
                    )
                },
                "bolt": {
                    "active_requests": len([r for r in self.active_requests.values() if r.system_type == "bolt"]),
                    "cpu_allocated": sum(
                        pool.active_allocations.get(req_id, 0)
                        for req_id, req in self.active_requests.items()
                        if req.system_type == "bolt" and req.resource_type == ResourceType.CPU_CORES
                        for pool in [self.resource_pools[ResourceType.CPU_CORES]]
                    )
                }
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_allocations": self.metrics.successful_allocations,
                "success_rate": self.metrics.successful_allocations / self.metrics.total_requests if self.metrics.total_requests > 0 else 0,
                "resource_utilization": {rt.value: util for rt, util in self.metrics.resource_utilization.items()},
                "last_updated": self.metrics.last_updated
            }
        }
    
    async def optimize_allocations(self) -> bool:
        """Optimize current resource allocations."""
        try:
            async with self.coordination_lock:
                # Get current allocations
                current_allocations = list(self.active_requests.values())
                
                # Sort by priority and performance impact
                optimization_candidates = []
                
                for request in current_allocations:
                    if request.priority in [TaskPriority.LOW, TaskPriority.BACKGROUND]:
                        optimization_candidates.append(request)
                
                # If we have high-priority pending requests, consider rebalancing
                high_priority_pending = [
                    r for r in self.pending_requests 
                    if r.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
                ]
                
                if high_priority_pending and optimization_candidates:
                    # Release some low-priority resources
                    release_ids = [r.request_id for r in optimization_candidates[:2]]
                    await self.release_resources(release_ids)
                    
                    logger.info(f"🔄 Optimized allocations: released {len(release_ids)} low-priority requests")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Allocation optimization failed: {e}")
            return False
    
    async def get_system_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for system optimization."""
        recommendations = {
            "einstein": [],
            "bolt": [],
            "general": []
        }
        
        try:
            status = self.get_resource_status()
            
            # CPU utilization recommendations
            cpu_util = status["resource_pools"]["cpu_cores"]["utilization"]
            if cpu_util > 0.9:
                recommendations["general"].append("High CPU utilization detected. Consider scaling down non-critical tasks.")
            elif cpu_util < 0.3:
                recommendations["general"].append("Low CPU utilization. Consider increasing parallelism or batch sizes.")
            
            # Memory utilization recommendations
            memory_util = status["resource_pools"]["memory"]["utilization"]
            if memory_util > 0.8:
                recommendations["general"].append("High memory utilization. Consider optimizing memory usage or reducing batch sizes.")
            
            # System-specific recommendations
            einstein_cpu = status["system_breakdown"]["einstein"]["cpu_allocated"]
            bolt_cpu = status["system_breakdown"]["bolt"]["cpu_allocated"]
            
            if einstein_cpu > bolt_cpu * 2:
                recommendations["einstein"].append("Consider reducing Einstein CPU allocation to allow more resources for Bolt.")
            elif bolt_cpu > einstein_cpu * 2:
                recommendations["bolt"].append("Consider reducing Bolt CPU allocation to allow more resources for Einstein.")
            
            # Pending requests recommendations
            if status["request_status"]["pending"] > 5:
                recommendations["general"].append("High number of pending requests. Consider optimizing resource allocation policies.")
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations


# Global resource manager instance
_resource_manager = None

def get_resource_manager() -> EinsteinBoltResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = EinsteinBoltResourceManager()
    return _resource_manager


async def initialize_resource_sharing() -> EinsteinBoltResourceManager:
    """Initialize the resource sharing system."""
    manager = get_resource_manager()
    await manager.initialize()
    return manager


if __name__ == "__main__":
    async def main():
        # Test the resource sharing system
        manager = await initialize_resource_sharing()
        
        print("Resource Sharing System Test")
        print("=" * 50)
        
        # Test Einstein resource request
        einstein_requests = await manager.request_resources(
            component_name="EinsteinSearch",
            system_type="einstein",
            resources={
                ResourceType.CPU_CORES: 4,
                ResourceType.GPU_CORES: 8,
                ResourceType.MEMORY: 4096
            },
            duration_seconds=10.0,
            priority=TaskPriority.HIGH
        )
        
        # Test Bolt resource request
        bolt_requests = await manager.request_resources(
            component_name="BoltAgent",
            system_type="bolt",
            resources={
                ResourceType.CPU_CORES: 6,
                ResourceType.GPU_CORES: 12,
                ResourceType.MEMORY: 6144
            },
            duration_seconds=10.0,
            priority=TaskPriority.NORMAL
        )
        
        # Wait a bit and check status
        await asyncio.sleep(2)
        
        status = manager.get_resource_status()
        print(f"\nResource Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Get recommendations
        recommendations = await manager.get_system_recommendations()
        print(f"\nRecommendations:")
        for system, recs in recommendations.items():
            if recs:
                print(f"  {system}: {recs}")
        
        # Clean up
        await manager.release_resources(einstein_requests + bolt_requests)
        await manager.stop_resource_management()
        
        print("\n✅ Resource sharing test completed successfully")
    
    asyncio.run(main())