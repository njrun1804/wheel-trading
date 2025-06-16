#!/usr/bin/env python3
"""
Production Deployment and Validation Framework for Optimized Bolt System

Comprehensive deployment, validation, and monitoring for production-ready
Bolt system with all M4 Pro optimizations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .adaptive_concurrency import get_adaptive_concurrency_manager
from .agents.agent_pool import TaskPriority, WorkStealingAgentPool, WorkStealingTask
from .core.task_subdivision import get_subdivision_system
from .gpu_pipeline_optimization import get_gpu_pipeline_optimizer
from .metal_accelerated_search import get_metal_search
from .production_error_recovery import production_error_handling
from .unified_memory import get_unified_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""

    num_agents: int = 8
    enable_work_stealing: bool = True
    enable_task_subdivision: bool = True
    enable_gpu_pipeline: bool = True
    enable_adaptive_concurrency: bool = True
    pipeline_depth: int = 4
    validation_threshold: float = 0.8  # 80% success rate
    performance_target_tasks_per_sec: float = 100.0
    memory_limit_gb: float = 18.0


@dataclass
class ValidationResult:
    """Result of deployment validation."""

    component: str
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class ProductionBoltSystem:
    """Production-ready Bolt system with all optimizations."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.agent_pool: WorkStealingAgentPool | None = None
        self.subdivision_system = None
        self.gpu_pipeline = None
        self.concurrency_manager = None
        self.memory_manager = None
        self.metal_search = None

        self.deployment_start_time = 0.0
        self.validation_results: list[ValidationResult] = []
        self.performance_metrics = {}

        logger.info("Initialized Production Bolt System")

    async def deploy(self) -> dict[str, Any]:
        """Deploy the complete optimized Bolt system."""
        async with production_error_handling("ProductionBoltSystem", "deploy"):
            self.deployment_start_time = time.time()
            logger.info("🚀 Starting production Bolt system deployment...")

            try:
                # Phase 1: Initialize core components
                await self._initialize_core_components()

                # Phase 2: Initialize optimizations
                await self._initialize_optimizations()

                # Phase 3: Validate all components
                validation_success = await self._validate_deployment()

                # Phase 4: Performance benchmarking
                performance_results = await self._run_performance_benchmarks()

                deployment_time = time.time() - self.deployment_start_time

                # Generate deployment report
                report = {
                    "deployment_success": validation_success,
                    "deployment_time_ms": deployment_time * 1000,
                    "components_status": self._get_components_status(),
                    "validation_results": [
                        {
                            "component": r.component,
                            "success": r.success,
                            "metrics": r.metrics,
                            "errors": r.errors,
                            "duration_ms": r.duration_ms,
                        }
                        for r in self.validation_results
                    ],
                    "performance_benchmarks": performance_results,
                    "configuration": {
                        "num_agents": self.config.num_agents,
                        "work_stealing_enabled": self.config.enable_work_stealing,
                        "task_subdivision_enabled": self.config.enable_task_subdivision,
                        "gpu_pipeline_enabled": self.config.enable_gpu_pipeline,
                        "adaptive_concurrency_enabled": self.config.enable_adaptive_concurrency,
                    },
                }

                if validation_success:
                    logger.info("✅ Production Bolt system deployment successful")
                else:
                    logger.error(
                        "❌ Production Bolt system deployment failed validation"
                    )

                return report

            except Exception as e:
                logger.error(f"❌ Production deployment failed: {e}")
                raise

    async def _initialize_core_components(self):
        """Initialize core Bolt components."""
        logger.info("Initializing core components...")

        # Initialize memory manager
        self.memory_manager = get_unified_memory_manager()
        logger.info("✅ Unified memory manager initialized")

        # Initialize work stealing agent pool
        self.agent_pool = WorkStealingAgentPool(
            num_agents=self.config.num_agents,
            enable_work_stealing=self.config.enable_work_stealing,
        )
        await self.agent_pool.initialize()
        logger.info(
            f"✅ Work stealing agent pool initialized ({self.config.num_agents} agents)"
        )

    async def _initialize_optimizations(self):
        """Initialize optimization components."""
        logger.info("Initializing optimizations...")

        # Task subdivision system
        if self.config.enable_task_subdivision:
            self.subdivision_system = get_subdivision_system()
            logger.info("✅ Task subdivision system initialized")

        # GPU pipeline optimizer
        if self.config.enable_gpu_pipeline:
            self.gpu_pipeline = get_gpu_pipeline_optimizer()
            await self.gpu_pipeline.start_pipeline()
            logger.info("✅ GPU pipeline optimizer initialized")

        # Adaptive concurrency manager
        if self.config.enable_adaptive_concurrency:
            self.concurrency_manager = get_adaptive_concurrency_manager()
            logger.info("✅ Adaptive concurrency manager initialized")

        # Metal accelerated search
        try:
            self.metal_search = await get_metal_search(embedding_dim=768)
            logger.info("✅ Metal accelerated search initialized")
        except Exception as e:
            logger.warning(f"⚠️  Metal search initialization failed: {e}")

    async def _validate_deployment(self) -> bool:
        """Validate all deployed components."""
        logger.info("Running deployment validation...")

        validation_tasks = [
            self._validate_agent_pool(),
            self._validate_work_stealing(),
            self._validate_task_subdivision(),
            self._validate_gpu_pipeline(),
            self._validate_memory_management(),
            self._validate_metal_search(),
        ]

        validation_results = await asyncio.gather(
            *validation_tasks, return_exceptions=True
        )

        success_count = sum(
            1
            for result in validation_results
            if isinstance(result, ValidationResult) and result.success
        )
        total_validations = len(validation_results)

        success_rate = success_count / total_validations
        validation_success = success_rate >= self.config.validation_threshold

        logger.info(
            f"Validation complete: {success_count}/{total_validations} passed ({success_rate:.1%})"
        )

        return validation_success

    async def _validate_agent_pool(self) -> ValidationResult:
        """Validate agent pool functionality."""
        start_time = time.time()

        try:
            # Test basic agent functionality
            test_tasks = [
                WorkStealingTask(
                    id=f"validation_task_{i}",
                    description=f"Validation test {i}",
                    priority=TaskPriority.NORMAL,
                )
                for i in range(5)
            ]

            # Submit tasks
            for task in test_tasks:
                await self.agent_pool.submit_task(task)

            # Wait for completion
            await asyncio.sleep(1.0)

            # Check status
            status = self.agent_pool.get_pool_status()

            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="agent_pool",
                success=status["total_agents"] == self.config.num_agents,
                metrics=status,
                duration_ms=duration_ms,
            )

            if result.success:
                logger.info("✅ Agent pool validation passed")
            else:
                logger.error("❌ Agent pool validation failed")
                result.errors.append("Agent count mismatch")

            self.validation_results.append(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="agent_pool",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _validate_work_stealing(self) -> ValidationResult:
        """Validate work stealing functionality."""
        start_time = time.time()

        try:
            if not self.config.enable_work_stealing:
                return ValidationResult(
                    component="work_stealing",
                    success=True,
                    metrics={"enabled": False},
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Create unbalanced workload to test stealing
            heavy_task = WorkStealingTask(
                id="heavy_validation_task",
                description="Heavy task for work stealing test",
                priority=TaskPriority.HIGH,
                estimated_duration=5.0,
            )

            await self.agent_pool.submit_task(heavy_task)

            # Submit light tasks that should trigger stealing
            light_tasks = [
                WorkStealingTask(
                    id=f"light_task_{i}",
                    description=f"Light task {i}",
                    priority=TaskPriority.NORMAL,
                    estimated_duration=0.1,
                )
                for i in range(10)
            ]

            for task in light_tasks:
                await self.agent_pool.submit_task(task)

            await asyncio.sleep(2.0)

            status = self.agent_pool.get_pool_status()
            steals_attempted = status["performance_metrics"].get(
                "total_steals_attempted", 0
            )

            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="work_stealing",
                success=steals_attempted > 0,
                metrics={"steals_attempted": steals_attempted},
                duration_ms=duration_ms,
            )

            if result.success:
                logger.info(
                    f"✅ Work stealing validation passed ({steals_attempted} steals attempted)"
                )
            else:
                logger.warning("⚠️  Work stealing validation: no steals detected")

            self.validation_results.append(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="work_stealing",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _validate_task_subdivision(self) -> ValidationResult:
        """Validate task subdivision functionality."""
        start_time = time.time()

        try:
            if not self.config.enable_task_subdivision:
                return ValidationResult(
                    component="task_subdivision",
                    success=True,
                    metrics={"enabled": False},
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Test subdivision with large task
            large_task = WorkStealingTask(
                id="subdivision_test_task",
                description="Large task for subdivision test",
                estimated_duration=10.0,
                subdividable=True,
            )

            (
                subdivided,
                subtasks,
                metrics,
            ) = await self.subdivision_system.analyze_and_subdivide(
                large_task, available_agents=4, current_system_load=0.5
            )

            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="task_subdivision",
                success=subdivided and len(subtasks) > 1,
                metrics={
                    "subdivided": subdivided,
                    "num_subtasks": len(subtasks),
                    "predicted_speedup": metrics.predicted_speedup,
                },
                duration_ms=duration_ms,
            )

            if result.success:
                logger.info(
                    f"✅ Task subdivision validation passed ({len(subtasks)} subtasks)"
                )
            else:
                logger.warning(
                    "⚠️  Task subdivision validation: no subdivision occurred"
                )

            self.validation_results.append(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="task_subdivision",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _validate_gpu_pipeline(self) -> ValidationResult:
        """Validate GPU pipeline functionality."""
        start_time = time.time()

        try:
            if not self.config.enable_gpu_pipeline:
                return ValidationResult(
                    component="gpu_pipeline",
                    success=True,
                    metrics={"enabled": False},
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Test GPU pipeline with sample data
            import numpy as np

            test_data = np.random.randn(100, 128).astype(np.float32)

            result_future = await self.gpu_pipeline.submit_task(
                test_data, "gpu_validation_task"
            )
            result = await asyncio.wait_for(result_future, timeout=5.0)

            metrics = self.gpu_pipeline.get_performance_metrics()

            duration_ms = (time.time() - start_time) * 1000
            validation_result = ValidationResult(
                component="gpu_pipeline",
                success=result is not None and metrics["tasks_processed"] > 0,
                metrics=metrics,
                duration_ms=duration_ms,
            )

            if validation_result.success:
                logger.info("✅ GPU pipeline validation passed")
            else:
                logger.error("❌ GPU pipeline validation failed")

            self.validation_results.append(validation_result)
            return validation_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="gpu_pipeline",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _validate_memory_management(self) -> ValidationResult:
        """Validate unified memory management."""
        start_time = time.time()

        try:
            # Test memory allocation and operations
            test_buffer = await self.memory_manager.allocate_buffer(
                1024 * 1024,  # 1MB
                self.memory_manager.BufferType.TEMPORARY,
                "validation_buffer",
            )

            import numpy as np

            test_data = np.random.randn(1000, 256).astype(np.float32)
            await test_buffer.copy_from_numpy(test_data)

            retrieved_data = await test_buffer.as_numpy(np.float32, test_data.shape)

            # Cleanup
            self.memory_manager.release_buffer("validation_buffer")

            memory_stats = self.memory_manager.get_memory_stats()

            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="memory_management",
                success=retrieved_data.shape == test_data.shape,
                metrics=memory_stats,
                duration_ms=duration_ms,
            )

            if result.success:
                logger.info("✅ Memory management validation passed")
            else:
                logger.error("❌ Memory management validation failed")

            self.validation_results.append(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="memory_management",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _validate_metal_search(self) -> ValidationResult:
        """Validate Metal accelerated search."""
        start_time = time.time()

        try:
            if not self.metal_search:
                return ValidationResult(
                    component="metal_search",
                    success=True,
                    metrics={"enabled": False},
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Test search functionality
            import numpy as np

            test_embeddings = np.random.randn(100, 768).astype(np.float32)
            test_metadata = [{"content": f"test_doc_{i}"} for i in range(100)]

            await self.metal_search.load_corpus(test_embeddings, test_metadata)

            query = np.random.randn(1, 768).astype(np.float32)
            search_results = await self.metal_search.search(query, k=5)

            performance_stats = self.metal_search.get_performance_stats()

            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="metal_search",
                success=len(search_results) > 0 and len(search_results[0]) > 0,
                metrics=performance_stats,
                duration_ms=duration_ms,
            )

            if result.success:
                logger.info("✅ Metal search validation passed")
            else:
                logger.error("❌ Metal search validation failed")

            self.validation_results.append(result)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                component="metal_search",
                success=False,
                errors=[str(e)],
                duration_ms=duration_ms,
            )
            self.validation_results.append(result)
            return result

    async def _run_performance_benchmarks(self) -> dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        logger.info("Running performance benchmarks...")

        benchmarks = {}

        # Agent pool throughput benchmark
        if self.agent_pool:
            start_time = time.time()

            # Submit many tasks
            test_tasks = [
                WorkStealingTask(
                    id=f"benchmark_task_{i}",
                    description=f"Benchmark task {i}",
                    priority=TaskPriority.NORMAL,
                )
                for i in range(100)
            ]

            for task in test_tasks:
                await self.agent_pool.submit_task(task)

            # Wait for completion
            await asyncio.sleep(3.0)

            duration = time.time() - start_time
            throughput = len(test_tasks) / duration

            benchmarks["agent_pool_throughput"] = {
                "tasks_per_second": throughput,
                "total_tasks": len(test_tasks),
                "duration_seconds": duration,
                "meets_target": throughput
                >= self.config.performance_target_tasks_per_sec,
            }

        # Memory performance benchmark
        if self.memory_manager:
            start_time = time.time()

            # Allocate and release many buffers
            from .unified_memory import BufferType

            for i in range(10):
                await self.memory_manager.allocate_buffer(
                    1024 * 1024, BufferType.TEMPORARY, f"benchmark_buffer_{i}"  # 1MB
                )
                self.memory_manager.release_buffer(f"benchmark_buffer_{i}")

            duration = time.time() - start_time
            memory_stats = self.memory_manager.get_memory_stats()

            benchmarks["memory_performance"] = {
                "allocation_latency_ms": (duration / 20)
                * 1000,  # Per allocation+release
                "memory_stats": memory_stats,
            }

        logger.info("Performance benchmarks completed")
        return benchmarks

    def _get_components_status(self) -> dict[str, Any]:
        """Get status of all components."""
        return {
            "agent_pool": {
                "initialized": self.agent_pool is not None,
                "status": self.agent_pool.get_pool_status()
                if self.agent_pool
                else None,
            },
            "subdivision_system": {
                "initialized": self.subdivision_system is not None,
                "performance": self.subdivision_system.get_performance_report()
                if self.subdivision_system
                else None,
            },
            "gpu_pipeline": {
                "initialized": self.gpu_pipeline is not None,
                "metrics": self.gpu_pipeline.get_performance_metrics()
                if self.gpu_pipeline
                else None,
            },
            "memory_manager": {
                "initialized": self.memory_manager is not None,
                "stats": self.memory_manager.get_memory_stats()
                if self.memory_manager
                else None,
            },
            "metal_search": {
                "initialized": self.metal_search is not None,
                "stats": self.metal_search.get_performance_stats()
                if self.metal_search
                else None,
            },
        }

    async def shutdown(self):
        """Gracefully shutdown the production system."""
        async with production_error_handling("ProductionBoltSystem", "shutdown"):
            logger.info("Shutting down production Bolt system...")

            if self.agent_pool:
                await self.agent_pool.shutdown()

            if self.gpu_pipeline:
                await self.gpu_pipeline.shutdown()

            # Additional cleanup
            logger.info("Production Bolt system shutdown complete")


async def deploy_production_bolt_system(
    config: DeploymentConfig | None = None,
) -> dict[str, Any]:
    """Deploy production-ready Bolt system with all optimizations."""
    if config is None:
        config = DeploymentConfig()

    system = ProductionBoltSystem(config)
    return await system.deploy()


# Quick deployment function
async def quick_deploy() -> dict[str, Any]:
    """Quick deployment with default configuration."""
    return await deploy_production_bolt_system()
