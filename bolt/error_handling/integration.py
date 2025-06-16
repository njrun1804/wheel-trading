"""
Bolt Error Handling Integration

Provides integration between Einstein and Bolt error handling systems,
enabling coordinated error recovery and system-wide graceful degradation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .exceptions import BoltException, ErrorCategory, RecoveryStrategy
from .recovery import ErrorRecoveryManager
from .graceful_degradation import GracefulDegradationManager, DegradationLevel

# Import Einstein error handling if available
try:
    from einstein.error_handling import (
        EinsteinException,
        EinsteinRecoveryManager,
        get_einstein_recovery_manager
    )
    EINSTEIN_AVAILABLE = True
except ImportError:
    EINSTEIN_AVAILABLE = False


class SystemErrorLevel(Enum):
    """System-wide error levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


@dataclass
class SystemErrorEvent:
    """System-wide error event."""
    timestamp: float
    source_system: str  # "bolt" or "einstein"
    error_level: SystemErrorLevel
    component: str
    message: str
    recovery_action: str
    success: bool
    metadata: Dict[str, Any]


class IntegratedErrorManager:
    """Manages error handling across both Bolt and Einstein systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IntegratedErrorManager")
        
        # Initialize Bolt components
        self.bolt_recovery = ErrorRecoveryManager()
        self.degradation_manager = GracefulDegradationManager()
        
        # Initialize Einstein components if available
        self.einstein_recovery = None
        if EINSTEIN_AVAILABLE:
            try:
                self.einstein_recovery = get_einstein_recovery_manager()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Einstein recovery: {e}")
        
        # System state tracking
        self.system_errors: List[SystemErrorEvent] = []
        self.current_degradation_level = DegradationLevel.NORMAL
        self.error_cascade_threshold = 3  # Number of errors that trigger cascade response
        self.cascade_window = 300.0  # 5 minutes
        
        # Integration callbacks
        self.error_callbacks: List[callable] = []
        
        # Statistics
        self.stats = {
            'total_system_errors': 0,
            'bolt_errors': 0,
            'einstein_errors': 0,
            'cascade_events': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'degradation_triggers': 0
        }
    
    def register_error_callback(self, callback: callable):
        """Register a callback for system-wide error events."""
        self.error_callbacks.append(callback)
    
    async def handle_bolt_error(
        self,
        error: BoltException,
        context: Dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        """Handle Bolt system errors with integrated response."""
        context = context or {}
        
        try:
            # Record system error event
            error_level = self._determine_system_error_level(error, "bolt")
            
            # Handle the error using Bolt recovery
            success, result = await self.bolt_recovery.handle_error(error, context)
            
            # Record the event
            event = SystemErrorEvent(
                timestamp=time.time(),
                source_system="bolt",
                error_level=error_level,
                component=error.category.value,
                message=error.message,
                recovery_action=error.recovery_strategy.value,
                success=success,
                metadata={
                    'error_code': error.error_code,
                    'severity': error.severity.value,
                    'context': context
                }
            )
            
            await self._record_system_error(event)
            
            # Check for error cascade
            if not success or error_level in [SystemErrorLevel.MAJOR, SystemErrorLevel.CRITICAL]:
                await self._check_error_cascade()
            
            # Update statistics
            self.stats['bolt_errors'] += 1
            if success:
                self.stats['successful_recoveries'] += 1
            else:
                self.stats['failed_recoveries'] += 1
            
            return success, result
            
        except Exception as integration_error:
            self.logger.error(f"Integrated Bolt error handling failed: {integration_error}")
            return False, None
    
    async def handle_einstein_error(
        self,
        error: Any,  # EinsteinException or generic exception
        context: Dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        """Handle Einstein system errors with integrated response."""
        if not EINSTEIN_AVAILABLE or not self.einstein_recovery:
            self.logger.warning("Einstein error handling not available")
            return False, None
        
        context = context or {}
        
        try:
            # Convert to EinsteinException if needed
            if not hasattr(error, 'recovery_strategy'):
                # This is a generic exception, wrap it
                from einstein.error_handling.exceptions import wrap_einstein_exception
                error = wrap_einstein_exception(error)
            
            # Record system error event
            error_level = self._determine_system_error_level(error, "einstein")
            
            # Handle the error using Einstein recovery
            success, result = await self.einstein_recovery.handle_error(error, context)
            
            # Record the event
            event = SystemErrorEvent(
                timestamp=time.time(),
                source_system="einstein",
                error_level=error_level,
                component=getattr(error, 'category', 'unknown').value if hasattr(getattr(error, 'category', None), 'value') else 'unknown',
                message=str(error),
                recovery_action=getattr(error, 'recovery_strategy', 'unknown').value if hasattr(getattr(error, 'recovery_strategy', None), 'value') else 'unknown',
                success=success,
                metadata={
                    'error_code': getattr(error, 'error_code', 'unknown'),
                    'context': context
                }
            )
            
            await self._record_system_error(event)
            
            # Check for error cascade
            if not success or error_level in [SystemErrorLevel.MAJOR, SystemErrorLevel.CRITICAL]:
                await self._check_error_cascade()
            
            # Update statistics
            self.stats['einstein_errors'] += 1
            if success:
                self.stats['successful_recoveries'] += 1
            else:
                self.stats['failed_recoveries'] += 1
            
            return success, result
            
        except Exception as integration_error:
            self.logger.error(f"Integrated Einstein error handling failed: {integration_error}")
            return False, None
    
    def _determine_system_error_level(self, error: Any, source: str) -> SystemErrorLevel:
        """Determine system-wide error level from component error."""
        
        # Handle Bolt errors
        if hasattr(error, 'severity') and hasattr(error, 'category'):
            if error.severity.value == "critical":
                return SystemErrorLevel.CRITICAL
            elif error.severity.value == "high":
                if error.category.value in ["system", "hardware"]:
                    return SystemErrorLevel.MAJOR
                else:
                    return SystemErrorLevel.MODERATE
            elif error.severity.value == "medium":
                return SystemErrorLevel.MODERATE
            else:
                return SystemErrorLevel.MINOR
        
        # Handle Einstein errors
        elif hasattr(error, 'category'):
            error_str = str(error).lower()
            if any(term in error_str for term in ["critical", "fatal", "crash"]):
                return SystemErrorLevel.CRITICAL
            elif any(term in error_str for term in ["index", "search", "database"]):
                return SystemErrorLevel.MAJOR
            elif any(term in error_str for term in ["embedding", "file", "config"]):
                return SystemErrorLevel.MODERATE
            else:
                return SystemErrorLevel.MINOR
        
        # Generic error
        else:
            error_str = str(error).lower()
            if any(term in error_str for term in ["memory", "disk", "critical"]):
                return SystemErrorLevel.MAJOR
            else:
                return SystemErrorLevel.MODERATE
    
    async def _record_system_error(self, event: SystemErrorEvent):
        """Record and process a system error event."""
        self.system_errors.append(event)
        self.stats['total_system_errors'] += 1
        
        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {e}")
        
        # Log the event
        level_emoji = {
            SystemErrorLevel.MINOR: "ℹ️",
            SystemErrorLevel.MODERATE: "⚠️",
            SystemErrorLevel.MAJOR: "🔴",
            SystemErrorLevel.CRITICAL: "💥",
            SystemErrorLevel.CATASTROPHIC: "☠️"
        }
        
        emoji = level_emoji.get(event.error_level, "❓")
        
        self.logger.info(
            f"{emoji} System Error [{event.source_system.upper()}]: "
            f"{event.component} - {event.message} "
            f"(Recovery: {'✅' if event.success else '❌'})"
        )
    
    async def _check_error_cascade(self):
        """Check for error cascade and trigger system-wide response."""
        current_time = time.time()
        
        # Count recent errors
        recent_errors = [
            error for error in self.system_errors
            if current_time - error.timestamp < self.cascade_window
        ]
        
        # Count high-severity recent errors
        high_severity_errors = [
            error for error in recent_errors
            if error.error_level in [SystemErrorLevel.MAJOR, SystemErrorLevel.CRITICAL]
        ]
        
        # Check for cascade conditions
        cascade_triggered = False
        
        if len(high_severity_errors) >= self.error_cascade_threshold:
            cascade_triggered = True
            self.logger.warning(
                f"Error cascade detected: {len(high_severity_errors)} high-severity errors "
                f"in {self.cascade_window/60:.1f} minutes"
            )
        
        # Check for failed recoveries
        failed_recoveries = [
            error for error in recent_errors
            if not error.success
        ]
        
        if len(failed_recoveries) >= self.error_cascade_threshold:
            cascade_triggered = True
            self.logger.warning(
                f"Recovery failure cascade detected: {len(failed_recoveries)} failed recoveries "
                f"in {self.cascade_window/60:.1f} minutes"
            )
        
        if cascade_triggered:
            await self._trigger_cascade_response(recent_errors)
    
    async def _trigger_cascade_response(self, recent_errors: List[SystemErrorEvent]):
        """Trigger system-wide cascade response."""
        self.stats['cascade_events'] += 1
        
        self.logger.warning("Triggering system-wide cascade response")
        
        # Determine appropriate degradation level
        critical_errors = [
            error for error in recent_errors
            if error.error_level == SystemErrorLevel.CRITICAL
        ]
        
        major_errors = [
            error for error in recent_errors
            if error.error_level == SystemErrorLevel.MAJOR
        ]
        
        if critical_errors:
            target_level = DegradationLevel.EMERGENCY
            reason = f"Critical error cascade: {len(critical_errors)} critical errors"
        elif len(major_errors) >= 3:
            target_level = DegradationLevel.MINIMAL
            reason = f"Major error cascade: {len(major_errors)} major errors"
        elif len(recent_errors) >= 5:
            target_level = DegradationLevel.REDUCED
            reason = f"Error cascade: {len(recent_errors)} recent errors"
        else:
            target_level = DegradationLevel.REDUCED
            reason = "Preventive degradation due to error patterns"
        
        # Trigger degradation if not already at or below target level
        if self.current_degradation_level.value < target_level.value:
            success = await self.degradation_manager.trigger_degradation(
                target_level,
                reason,
                triggered_by="cascade_response"
            )
            
            if success:
                self.current_degradation_level = target_level
                self.stats['degradation_triggers'] += 1
                
                # Notify both systems about degradation
                await self._notify_systems_degradation(target_level)
    
    async def _notify_systems_degradation(self, level: DegradationLevel):
        """Notify both Bolt and Einstein systems about degradation."""
        self.logger.info(f"Notifying systems of degradation level: {level.value}")
        
        # Set environment variables for both systems
        import os
        os.environ['SYSTEM_DEGRADATION_LEVEL'] = level.value
        os.environ['BOLT_DEGRADATION_ACTIVE'] = 'true'
        os.environ['EINSTEIN_DEGRADATION_ACTIVE'] = 'true'
        
        # Specific degradation actions
        if level == DegradationLevel.EMERGENCY:
            os.environ['BOLT_EMERGENCY_MODE'] = 'true'
            os.environ['EINSTEIN_EMERGENCY_MODE'] = 'true'
            os.environ['SYSTEM_MAX_OPERATIONS'] = '1'
        elif level == DegradationLevel.MINIMAL:
            os.environ['BOLT_MINIMAL_MODE'] = 'true'
            os.environ['EINSTEIN_MINIMAL_MODE'] = 'true'
            os.environ['SYSTEM_MAX_OPERATIONS'] = '2'
        elif level == DegradationLevel.REDUCED:
            os.environ['BOLT_REDUCED_MODE'] = 'true'
            os.environ['EINSTEIN_REDUCED_MODE'] = 'true'
            os.environ['SYSTEM_MAX_OPERATIONS'] = '4'
    
    async def attempt_system_recovery(self) -> bool:
        """Attempt system-wide recovery from degraded state."""
        if self.current_degradation_level == DegradationLevel.NORMAL:
            return True
        
        self.logger.info(f"Attempting system recovery from {self.current_degradation_level.value}")
        
        try:
            # Check recent error patterns
            current_time = time.time()
            recent_errors = [
                error for error in self.system_errors
                if current_time - error.timestamp < 600  # Last 10 minutes
            ]
            
            # If no recent errors, attempt recovery
            if not recent_errors:
                success = await self.degradation_manager.attempt_recovery()
                
                if success:
                    self.current_degradation_level = self.degradation_manager.current_state.level
                    
                    # Clear degradation environment variables
                    import os
                    for var in ['SYSTEM_DEGRADATION_LEVEL', 'BOLT_DEGRADATION_ACTIVE', 
                               'EINSTEIN_DEGRADATION_ACTIVE', 'BOLT_EMERGENCY_MODE', 
                               'EINSTEIN_EMERGENCY_MODE', 'BOLT_MINIMAL_MODE',
                               'EINSTEIN_MINIMAL_MODE', 'BOLT_REDUCED_MODE',
                               'EINSTEIN_REDUCED_MODE']:
                        os.environ.pop(var, None)
                    
                    self.logger.info(f"System recovery successful: {self.current_degradation_level.value}")
                    return True
                else:
                    self.logger.warning("System recovery failed")
                    return False
            else:
                self.logger.info(f"System recovery deferred: {len(recent_errors)} recent errors")
                return False
                
        except Exception as e:
            self.logger.error(f"System recovery attempt failed: {e}")
            return False
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        current_time = time.time()
        
        # Recent error analysis
        recent_errors = [
            error for error in self.system_errors
            if current_time - error.timestamp < 3600  # Last hour
        ]
        
        error_by_level = {}
        for level in SystemErrorLevel:
            error_by_level[level.value] = len([
                error for error in recent_errors
                if error.error_level == level
            ])
        
        error_by_system = {
            'bolt': len([error for error in recent_errors if error.source_system == "bolt"]),
            'einstein': len([error for error in recent_errors if error.source_system == "einstein"])
        }
        
        # Recovery success rate
        total_recent = len(recent_errors)
        successful_recent = len([error for error in recent_errors if error.success])
        recovery_rate = (successful_recent / max(1, total_recent)) * 100
        
        return {
            'current_degradation_level': self.current_degradation_level.value,
            'total_errors_last_hour': total_recent,
            'errors_by_level': error_by_level,
            'errors_by_system': error_by_system,
            'recovery_success_rate': recovery_rate,
            'cascade_events': self.stats['cascade_events'],
            'degradation_triggers': self.stats['degradation_triggers'],
            'overall_stats': self.stats.copy(),
            'system_status': self._determine_system_status(recent_errors),
            'recommendations': self._generate_recommendations(recent_errors)
        }
    
    def _determine_system_status(self, recent_errors: List[SystemErrorEvent]) -> str:
        """Determine overall system status."""
        if self.current_degradation_level == DegradationLevel.EMERGENCY:
            return "emergency"
        elif self.current_degradation_level == DegradationLevel.MINIMAL:
            return "minimal"
        elif self.current_degradation_level == DegradationLevel.REDUCED:
            return "reduced"
        elif len(recent_errors) > 10:
            return "stressed"
        elif len(recent_errors) > 5:
            return "warning"
        else:
            return "healthy"
    
    def _generate_recommendations(self, recent_errors: List[SystemErrorEvent]) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []
        
        if self.current_degradation_level != DegradationLevel.NORMAL:
            recommendations.append(f"System is in {self.current_degradation_level.value} mode - consider manual intervention")
        
        # Analyze error patterns
        bolt_errors = [error for error in recent_errors if error.source_system == "bolt"]
        einstein_errors = [error for error in recent_errors if error.source_system == "einstein"]
        
        if len(bolt_errors) > len(einstein_errors) * 2:
            recommendations.append("High Bolt error rate - check system resources and hardware")
        elif len(einstein_errors) > len(bolt_errors) * 2:
            recommendations.append("High Einstein error rate - check indexing and search components")
        
        # Check for specific error patterns
        memory_errors = [
            error for error in recent_errors
            if "memory" in error.message.lower()
        ]
        
        if len(memory_errors) >= 3:
            recommendations.append("Multiple memory-related errors - consider increasing memory or reducing batch sizes")
        
        # Recovery rate
        failed_recoveries = [error for error in recent_errors if not error.success]
        if len(failed_recoveries) > len(recent_errors) * 0.5:
            recommendations.append("High recovery failure rate - consider manual system restart")
        
        return recommendations


# Global integrated error manager
_integrated_manager: IntegratedErrorManager | None = None


def get_integrated_error_manager() -> IntegratedErrorManager:
    """Get or create the global integrated error manager."""
    global _integrated_manager
    if _integrated_manager is None:
        _integrated_manager = IntegratedErrorManager()
    return _integrated_manager


# Convenience functions for error handling integration

async def handle_any_error(
    error: Exception,
    source_hint: str | None = None,
    context: Dict[str, Any] | None = None
) -> tuple[bool, Any]:
    """Handle any error using the integrated error management system."""
    manager = get_integrated_error_manager()
    
    # Determine source system
    if source_hint == "bolt" or isinstance(error, BoltException):
        return await manager.handle_bolt_error(error, context)
    elif source_hint == "einstein" or (EINSTEIN_AVAILABLE and isinstance(error, EinsteinException)):
        return await manager.handle_einstein_error(error, context)
    else:
        # Try to infer from error message or type
        error_str = str(error).lower()
        if any(term in error_str for term in ["bolt", "agent", "task", "gpu"]):
            # Wrap as Bolt error
            from .exceptions import wrap_exception
            bolt_error = wrap_exception(error)
            return await manager.handle_bolt_error(bolt_error, context)
        elif any(term in error_str for term in ["einstein", "index", "search", "embedding"]):
            # Wrap as Einstein error
            return await manager.handle_einstein_error(error, context)
        else:
            # Default to Bolt error handling
            from .exceptions import wrap_exception
            bolt_error = wrap_exception(error)
            return await manager.handle_bolt_error(bolt_error, context)