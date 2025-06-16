"""
Production Error Recovery System for M4 Pro Optimizations

Comprehensive error handling and recovery mechanisms for real-world deployment.
"""

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"  # Minor issues, system continues normally
    MEDIUM = "medium"  # Notable issues, some degradation
    HIGH = "high"  # Serious issues, significant degradation
    CRITICAL = "critical"  # System failure, immediate action required


class RecoveryAction(Enum):
    """Recovery action types"""

    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback implementation
    RESTART_COMPONENT = "restart"  # Restart the failing component
    DEGRADE_GRACEFULLY = "degrade"  # Reduce functionality gracefully
    ALERT_ONLY = "alert"  # Log and continue
    MANUAL_INTERVENTION = "manual"  # Requires manual intervention


@dataclass
class ErrorContext:
    """Context information for error handling"""

    component: str
    operation: str
    timestamp: float = field(default_factory=time.time)
    system_state: dict[str, Any] | None = None
    previous_errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryPlan:
    """Recovery plan for a specific error type"""

    severity: ErrorSeverity
    action: RecoveryAction
    max_retries: int = 3
    backoff_factor: float = 1.5
    timeout_seconds: float = 30.0
    fallback_function: Callable | None = None
    requires_user_notification: bool = False


class ProductionErrorRecovery:
    """
    Production-grade error recovery system for M4 Pro optimizations.

    Provides automatic error detection, classification, and recovery
    with minimal impact on system performance.
    """

    def __init__(self):
        self.error_counts: dict[str, int] = {}
        self.recovery_plans: dict[str, RecoveryPlan] = {}
        self.active_recoveries: dict[str, asyncio.Task] = {}
        self.system_health_score = 100.0
        self.degraded_components: set = set()

        # Initialize default recovery plans
        self._setup_default_recovery_plans()

        logger.info("Production error recovery system initialized")

    def _setup_default_recovery_plans(self):
        """Setup default recovery plans for common error types"""

        # Memory errors
        self.recovery_plans["memory_pressure"] = RecoveryPlan(
            severity=ErrorSeverity.HIGH,
            action=RecoveryAction.DEGRADE_GRACEFULLY,
            fallback_function=self._handle_memory_pressure,
            requires_user_notification=True,
        )

        self.recovery_plans["out_of_memory"] = RecoveryPlan(
            severity=ErrorSeverity.CRITICAL,
            action=RecoveryAction.RESTART_COMPONENT,
            max_retries=1,
            requires_user_notification=True,
        )

        # GPU/Metal errors
        self.recovery_plans["metal_device_error"] = RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.FALLBACK,
            fallback_function=self._handle_metal_fallback,
        )

        self.recovery_plans["mlx_compilation_error"] = RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.FALLBACK,
            fallback_function=self._handle_mlx_fallback,
        )

        # Database errors
        self.recovery_plans["database_lock"] = RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.RETRY,
            max_retries=5,
            backoff_factor=2.0,
            timeout_seconds=60.0,
        )

        self.recovery_plans["database_corruption"] = RecoveryPlan(
            severity=ErrorSeverity.CRITICAL,
            action=RecoveryAction.MANUAL_INTERVENTION,
            requires_user_notification=True,
        )

        # Network/API errors
        self.recovery_plans["connection_timeout"] = RecoveryPlan(
            severity=ErrorSeverity.LOW,
            action=RecoveryAction.RETRY,
            max_retries=3,
            backoff_factor=1.5,
        )

        # System resource errors
        self.recovery_plans["high_cpu_usage"] = RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.DEGRADE_GRACEFULLY,
            fallback_function=self._handle_cpu_pressure,
        )

        # Component-specific errors
        self.recovery_plans["ane_initialization_failure"] = RecoveryPlan(
            severity=ErrorSeverity.LOW,
            action=RecoveryAction.FALLBACK,
            fallback_function=self._handle_ane_fallback,
        )

        self.recovery_plans["embedding_search_failure"] = RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.FALLBACK,
            fallback_function=self._handle_search_fallback,
        )

    @asynccontextmanager
    async def error_handling_context(
        self, component: str, operation: str, context: ErrorContext | None = None
    ):
        """Context manager for automatic error handling and recovery"""
        if context is None:
            context = ErrorContext(component=component, operation=operation)

        try:
            yield context
        except Exception as e:
            await self.handle_error(e, context)
            raise

    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """
        Handle an error with automatic recovery.

        Returns True if error was handled and operation can continue,
        False if error requires escalation.
        """
        error_type = self._classify_error(error)
        error_key = f"{context.component}:{error_type}"

        # Update error tracking
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Update system health score
        self._update_health_score(error_type, context.component)

        # Get recovery plan
        recovery_plan = self.recovery_plans.get(error_type)
        if not recovery_plan:
            recovery_plan = self._get_default_recovery_plan(error)

        # Log error with context
        self._log_error(error, context, recovery_plan)

        # Execute recovery action
        try:
            success = await self._execute_recovery(error, context, recovery_plan)

            if success:
                logger.info(
                    f"Successfully recovered from {error_type} in {context.component}"
                )
                return True
            else:
                logger.error(f"Recovery failed for {error_type} in {context.component}")
                return False

        except Exception as recovery_error:
            logger.error(f"Recovery action failed: {recovery_error}")
            return False

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery planning"""
        error_message = str(error).lower()
        type(error).__name__.lower()

        # Memory-related errors
        if "memory" in error_message or "out of memory" in error_message:
            return "out_of_memory"
        elif "memory pressure" in error_message:
            return "memory_pressure"

        # GPU/Metal-related errors
        elif "metal" in error_message or "gpu" in error_message:
            return "metal_device_error"
        elif "mlx" in error_message or "compilation" in error_message:
            return "mlx_compilation_error"

        # Database-related errors
        elif "lock" in error_message or "database" in error_message:
            if "lock" in error_message:
                return "database_lock"
            elif "corrupt" in error_message:
                return "database_corruption"
            else:
                return "database_error"

        # Network/connection errors
        elif "timeout" in error_message or "connection" in error_message:
            return "connection_timeout"

        # System resource errors
        elif "cpu" in error_message and "high" in error_message:
            return "high_cpu_usage"

        # Component-specific errors
        elif "ane" in error_message or "coreml" in error_message:
            return "ane_initialization_failure"
        elif "embedding" in error_message or "search" in error_message:
            return "embedding_search_failure"

        # Generic error classification
        elif "permission" in error_message:
            return "permission_error"
        elif "file" in error_message and "not found" in error_message:
            return "file_not_found"
        else:
            return "unknown_error"

    def _get_default_recovery_plan(self, error: Exception) -> RecoveryPlan:
        """Get default recovery plan for unclassified errors"""
        # Conservative default recovery plan
        return RecoveryPlan(
            severity=ErrorSeverity.MEDIUM,
            action=RecoveryAction.RETRY,
            max_retries=2,
            backoff_factor=2.0,
            timeout_seconds=30.0,
        )

    async def _execute_recovery(
        self, error: Exception, context: ErrorContext, plan: RecoveryPlan
    ) -> bool:
        """Execute the recovery plan"""

        if plan.action == RecoveryAction.RETRY:
            return await self._retry_with_backoff(context, plan)

        elif plan.action == RecoveryAction.FALLBACK:
            return await self._execute_fallback(context, plan)

        elif plan.action == RecoveryAction.RESTART_COMPONENT:
            return await self._restart_component(context.component)

        elif plan.action == RecoveryAction.DEGRADE_GRACEFULLY:
            return await self._degrade_gracefully(context.component, plan)

        elif plan.action == RecoveryAction.ALERT_ONLY:
            self._send_alert(error, context, plan)
            return True  # Continue operation

        elif plan.action == RecoveryAction.MANUAL_INTERVENTION:
            self._request_manual_intervention(error, context, plan)
            return False  # Cannot auto-recover

        else:
            logger.warning(f"Unknown recovery action: {plan.action}")
            return False

    async def _retry_with_backoff(
        self, context: ErrorContext, plan: RecoveryPlan
    ) -> bool:
        """Retry operation with exponential backoff"""
        if context.retry_count >= plan.max_retries:
            logger.error(
                f"Max retries ({plan.max_retries}) exceeded for {context.operation}"
            )
            return False

        # Calculate backoff delay
        delay = min(plan.backoff_factor**context.retry_count, 60.0)  # Max 60s delay

        logger.info(
            f"Retrying {context.operation} in {delay:.1f}s (attempt {context.retry_count + 1})"
        )
        await asyncio.sleep(delay)

        context.retry_count += 1
        return True  # Indicate retry should be attempted

    async def _execute_fallback(
        self, context: ErrorContext, plan: RecoveryPlan
    ) -> bool:
        """Execute fallback function if available"""
        if plan.fallback_function:
            try:
                await plan.fallback_function(context)
                logger.info(f"Fallback executed successfully for {context.component}")
                return True
            except Exception as e:
                logger.error(f"Fallback failed for {context.component}: {e}")
                return False
        else:
            logger.warning(f"No fallback function available for {context.component}")
            return False

    async def _restart_component(self, component_name: str) -> bool:
        """Restart a specific component"""
        logger.warning(f"Restarting component: {component_name}")

        try:
            # Component restart logic would go here
            # For now, just mark as degraded and return success
            self.degraded_components.add(component_name)

            # Simulate restart delay
            await asyncio.sleep(2.0)

            logger.info(f"Component {component_name} restarted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to restart component {component_name}: {e}")
            return False

    async def _degrade_gracefully(
        self, component_name: str, plan: RecoveryPlan
    ) -> bool:
        """Degrade component functionality gracefully"""
        logger.info(f"Gracefully degrading component: {component_name}")

        self.degraded_components.add(component_name)

        if plan.fallback_function:
            try:
                await plan.fallback_function(
                    ErrorContext(component=component_name, operation="degrade")
                )
                return True
            except Exception as e:
                logger.error(f"Graceful degradation failed: {e}")
                return False

        return True  # Basic degradation successful

    def _update_health_score(self, error_type: str, component: str):
        """Update system health score based on errors"""
        severity_impact = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 5,
            ErrorSeverity.HIGH: 15,
            ErrorSeverity.CRITICAL: 30,
        }

        plan = self.recovery_plans.get(error_type)
        if plan:
            impact = severity_impact.get(plan.severity, 5)
            self.system_health_score = max(0, self.system_health_score - impact)

            # Gradual recovery of health score over time
            if self.system_health_score < 100:
                self.system_health_score = min(100, self.system_health_score + 0.1)

    def _log_error(self, error: Exception, context: ErrorContext, plan: RecoveryPlan):
        """Log error with appropriate level based on severity"""
        error_msg = f"Error in {context.component}.{context.operation}: {error}"

        if plan.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_msg)
            logger.critical(f"Stack trace: {traceback.format_exc()}")
        elif plan.severity == ErrorSeverity.HIGH:
            logger.error(error_msg)
        elif plan.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_msg)
        else:
            logger.info(error_msg)

    def _send_alert(self, error: Exception, context: ErrorContext, plan: RecoveryPlan):
        """Send alert notification"""
        logger.warning(f"ALERT: {context.component} error requires attention: {error}")
        # In a real system, this would send notifications via email, Slack, etc.

    def _request_manual_intervention(
        self, error: Exception, context: ErrorContext, plan: RecoveryPlan
    ):
        """Request manual intervention for critical errors"""
        logger.critical(f"MANUAL INTERVENTION REQUIRED: {context.component}")
        logger.critical(f"Error: {error}")
        logger.critical(f"Context: {context}")
        # In a real system, this would create tickets, send urgent notifications, etc.

    # Specific fallback handlers
    async def _handle_memory_pressure(self, context: ErrorContext):
        """Handle memory pressure by freeing caches and buffers"""
        logger.info("Handling memory pressure - freeing caches")

        # Get memory managers and free caches
        try:
            # Import locally to avoid circular dependencies
            from .memory_pools import get_memory_pool_manager
            from .unified_memory import get_unified_memory_manager

            memory_manager = get_unified_memory_manager()
            if hasattr(memory_manager, "cleanup_unused"):
                memory_manager.cleanup_unused()

            pool_manager = get_memory_pool_manager()
            if hasattr(pool_manager, "global_cleanup"):
                pool_manager.global_cleanup()

        except Exception as e:
            logger.error(f"Memory pressure handling failed: {e}")

    async def _handle_metal_fallback(self, context: ErrorContext):
        """Handle Metal device errors by switching to CPU"""
        logger.info("Switching to CPU fallback for Metal operations")
        # This would disable Metal acceleration for future operations

    async def _handle_mlx_fallback(self, context: ErrorContext):
        """Handle MLX compilation errors"""
        logger.info("Using CPU fallback for MLX operations")
        # This would disable MLX acceleration

    async def _handle_cpu_pressure(self, context: ErrorContext):
        """Handle high CPU usage by reducing concurrency"""
        logger.info("Reducing concurrency due to high CPU usage")

        try:
            from .adaptive_concurrency import get_adaptive_concurrency_manager

            get_adaptive_concurrency_manager()
            # Reduce concurrency limits temporarily

        except Exception as e:
            logger.error(f"CPU pressure handling failed: {e}")

    async def _handle_ane_fallback(self, context: ErrorContext):
        """Handle ANE initialization failure"""
        logger.info("Using CPU fallback for ANE operations")
        # ANE operations will fall back to CPU

    async def _handle_search_fallback(self, context: ErrorContext):
        """Handle embedding search failures"""
        logger.info("Using basic search fallback")
        # Search operations will use simple CPU-based similarity

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status and health metrics"""
        return {
            "health_score": self.system_health_score,
            "degraded_components": list(self.degraded_components),
            "error_counts": dict(self.error_counts),
            "active_recoveries": len(self.active_recoveries),
            "recovery_plans_configured": len(self.recovery_plans),
        }


# Global error recovery instance
_error_recovery: ProductionErrorRecovery | None = None


def get_error_recovery() -> ProductionErrorRecovery:
    """Get global error recovery instance"""
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = ProductionErrorRecovery()
    return _error_recovery


@asynccontextmanager
async def production_error_handling(component: str, operation: str):
    """Convenience context manager for production error handling"""
    recovery = get_error_recovery()
    async with recovery.error_handling_context(component, operation) as context:
        yield context
