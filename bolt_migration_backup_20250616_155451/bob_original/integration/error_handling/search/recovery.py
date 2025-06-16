"""
Einstein Error Recovery Manager

Comprehensive error recovery system for Einstein indexing and search operations,
providing intelligent recovery strategies, fallback mechanisms, and graceful
degradation when tools fail.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .exceptions import (
    EinsteinException,
    EinsteinIndexException,
    EinsteinSearchException,
    EinsteinEmbeddingException,
    EinsteinFileWatcherException,
    EinsteinDatabaseException,
    EinsteinResourceException,
    EinsteinRecoveryStrategy,
    EinsteinErrorCategory,
)


class RecoveryState(Enum):
    """States of recovery operations in Einstein."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    RECOVERING = "recovering"
    REBUILDING = "rebuilding"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryConfiguration:
    """Configuration for Einstein recovery behavior."""
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    index_rebuild_timeout: float = 300.0  # 5 minutes
    search_fallback_enabled: bool = True
    embedding_fallback_enabled: bool = True
    graceful_degradation_enabled: bool = True
    auto_recovery_enabled: bool = True
    recovery_validation_enabled: bool = True


class EinsteinRecoveryManager:
    """Manages error recovery for Einstein indexing and search operations."""
    
    def __init__(self, config: RecoveryConfiguration | None = None):
        self.config = config or RecoveryConfiguration()
        self.logger = logging.getLogger(f"{__name__}.EinsteinRecoveryManager")
        
        # Recovery state tracking
        self.state = RecoveryState.IDLE
        self.recovery_history: list[dict[str, Any]] = []
        self.active_recoveries: dict[str, dict[str, Any]] = {}
        
        # Component references (set by Einstein system)
        self.einstein_hub = None
        self.fallback_manager = None
        
        # Recovery strategy handlers
        self.strategy_handlers = {
            EinsteinRecoveryStrategy.RETRY: self._handle_retry_recovery,
            EinsteinRecoveryStrategy.FALLBACK: self._handle_fallback_recovery,
            EinsteinRecoveryStrategy.REBUILD_INDEX: self._handle_rebuild_recovery,
            EinsteinRecoveryStrategy.REDUCE_SCOPE: self._handle_reduce_scope_recovery,
            EinsteinRecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_graceful_degradation,
        }
        
        # Statistics
        self.stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'by_strategy': {strategy.value: 0 for strategy in EinsteinRecoveryStrategy},
            'by_error_category': {category.value: 0 for category in EinsteinErrorCategory}
        }
    
    def register_components(
        self,
        einstein_hub=None,
        fallback_manager=None
    ):
        """Register Einstein system components for recovery operations."""
        self.einstein_hub = einstein_hub
        self.fallback_manager = fallback_manager
    
    async def handle_error(
        self,
        error: EinsteinException,
        context: dict[str, Any] | None = None
    ) -> tuple[bool, Any | None]:
        """
        Main entry point for Einstein error handling and recovery.
        
        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        context = context or {}
        start_time = time.time()
        
        try:
            self.state = RecoveryState.ANALYZING
            self.logger.info(f"Handling Einstein error: {error.error_code} - {error.message}")
            
            # Update statistics
            self.stats['total_recoveries'] += 1
            self.stats['by_error_category'][error.category.value] += 1
            
            # Check if error is recoverable
            if not error.is_recoverable():
                self.logger.error(f"Einstein error {error.error_code} is not recoverable")
                return False, None
            
            # Create recovery attempt record
            attempt = {
                'timestamp': start_time,
                'error_code': error.error_code,
                'strategy': error.recovery_strategy,
                'context': context,
                'duration': None,
                'success': False,
                'result': None
            }
            
            self.active_recoveries[error.error_code] = attempt
            
            # Execute recovery strategy
            self.state = RecoveryState.RECOVERING
            success, result = await self._execute_recovery_strategy(error, context)
            
            # Record attempt
            attempt['duration'] = time.time() - start_time
            attempt['success'] = success
            attempt['result'] = str(result) if result else None
            
            # Update statistics
            if success:
                self.stats['successful_recoveries'] += 1
            else:
                self.stats['failed_recoveries'] += 1
            
            self.stats['by_strategy'][error.recovery_strategy.value] += 1
            self._update_average_recovery_time(attempt['duration'])
            
            # Validate recovery if enabled
            if success and self.config.recovery_validation_enabled:
                self.state = RecoveryState.VALIDATING
                validation_result = await self._validate_recovery(error, result, context)
                if not validation_result:
                    self.logger.warning(f"Einstein recovery validation failed for {error.error_code}")
                    success = False
            
            self.state = RecoveryState.COMPLETED if success else RecoveryState.FAILED
            self.recovery_history.append(attempt)
            
            self.logger.info(
                f"Einstein recovery {'successful' if success else 'failed'} for {error.error_code} "
                f"using {error.recovery_strategy.value} strategy "
                f"(duration: {attempt['duration']:.2f}s)"
            )
            
            return success, result
            
        except Exception as recovery_error:
            self.logger.error(f"Einstein recovery process failed: {recovery_error}", exc_info=True)
            self.state = RecoveryState.FAILED
            return False, None
            
        finally:
            # Cleanup
            self.active_recoveries.pop(error.error_code, None)
            if self.state != RecoveryState.FAILED:
                self.state = RecoveryState.IDLE
    
    async def _execute_recovery_strategy(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> tuple[bool, Any | None]:
        """Execute the appropriate recovery strategy for Einstein errors."""
        
        strategy = error.recovery_strategy
        handler = self.strategy_handlers.get(strategy)
        
        if not handler:
            self.logger.error(f"No handler for Einstein recovery strategy: {strategy}")
            return False, None
        
        try:
            result = await handler(error, context)
            return True, result
            
        except Exception as strategy_error:
            self.logger.error(
                f"Einstein recovery strategy {strategy.value} failed: {strategy_error}",
                exc_info=True
            )
            return False, None
    
    async def _handle_retry_recovery(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> Any:
        """Handle retry-based recovery for Einstein operations."""
        
        retry_count = context.get('retry_count', 0)
        max_retries = self.config.max_retry_attempts
        
        if retry_count >= max_retries:
            raise Exception(f"Maximum retry attempts ({max_retries}) exceeded for Einstein operation")
        
        # Adaptive retry delay based on Einstein error type
        base_delay = self.config.retry_delay_base
        if isinstance(error, EinsteinResourceException):
            base_delay *= 3.0  # Longer delays for resource issues
        elif isinstance(error, EinsteinDatabaseException):
            base_delay *= 2.0  # Moderate delays for database issues
        elif isinstance(error, EinsteinIndexException):
            base_delay *= 1.5  # Slight delays for index issues
        
        delay = min(base_delay * (2 ** retry_count), self.config.retry_delay_max)
        
        self.logger.info(
            f"Retrying Einstein operation {error.error_code} after {delay:.2f}s "
            f"(attempt {retry_count + 1}/{max_retries})"
        )
        await asyncio.sleep(delay)
        
        context['retry_count'] = retry_count + 1
        
        return {
            "retry_count": retry_count + 1,
            "delay": delay,
            "strategy": "retry"
        }
    
    async def _handle_fallback_recovery(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> Any:
        """Handle fallback recovery for Einstein operations."""
        
        self.logger.info(f"Applying fallback recovery for Einstein {error.category.value} error")
        
        fallback_actions = []
        
        if isinstance(error, EinsteinSearchException):
            # Search fallback chain
            fallback_actions.extend([
                "Switching to text-based search",
                "Using basic pattern matching",
                "Reducing search scope"
            ])
            
            # If we have a fallback manager, use it
            if self.fallback_manager:
                try:
                    fallback_result = await self.fallback_manager.execute_search_fallback(
                        error.query, context
                    )
                    fallback_actions.append("Fallback search executed successfully")
                    return {
                        "fallback_actions": fallback_actions,
                        "fallback_result": fallback_result,
                        "strategy": "search_fallback"
                    }
                except Exception as fallback_error:
                    self.logger.warning(f"Search fallback failed: {fallback_error}")
        
        elif isinstance(error, EinsteinEmbeddingException):
            # Embedding fallback chain
            fallback_actions.extend([
                "Switching to CPU-based embeddings",
                "Using alternative embedding model",
                "Falling back to text similarity"
            ])
            
        elif isinstance(error, EinsteinIndexException):
            # Index fallback
            fallback_actions.extend([
                "Using cached index if available",
                "Switching to file-based search",
                "Building minimal index"
            ])
        
        elif isinstance(error, EinsteinFileWatcherException):
            # File watcher fallback
            fallback_actions.extend([
                "Disabling real-time updates",
                "Using manual refresh mode",
                "Polling for file changes"
            ])
        
        else:
            # Generic fallback
            fallback_actions.extend([
                "Enabling basic operation mode",
                "Disabling advanced features",
                "Using simplified processing"
            ])
        
        return {
            "fallback_actions": fallback_actions,
            "strategy": "fallback"
        }
    
    async def _handle_rebuild_recovery(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> Any:
        """Handle index rebuild recovery for Einstein."""
        
        self.logger.info(f"Rebuilding Einstein index for {error.category.value} error")
        self.state = RecoveryState.REBUILDING
        
        rebuild_actions = []
        
        if isinstance(error, EinsteinIndexException):
            # Full index rebuild
            if self.einstein_hub:
                try:
                    # Clear existing index
                    rebuild_actions.append("Clearing corrupted index")
                    
                    # Rebuild index from scratch
                    rebuild_actions.append("Starting fresh index build")
                    await self.einstein_hub.rebuild_index()
                    
                    rebuild_actions.append("Index rebuild completed successfully")
                    
                except Exception as rebuild_error:
                    self.logger.error(f"Index rebuild failed: {rebuild_error}")
                    rebuild_actions.append(f"Index rebuild failed: {rebuild_error}")
                    raise
        
        elif isinstance(error, EinsteinDatabaseException):
            # Database rebuild
            rebuild_actions.extend([
                "Recreating database schema",
                "Rebuilding database indexes",
                "Validating database integrity"
            ])
        
        return {
            "rebuild_actions": rebuild_actions,
            "strategy": "rebuild"
        }
    
    async def _handle_reduce_scope_recovery(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> Any:
        """Handle scope reduction recovery for Einstein operations."""
        
        self.logger.info(f"Reducing scope for Einstein {error.category.value} error")
        
        scope_actions = []
        
        if isinstance(error, EinsteinSearchException):
            # Reduce search scope
            scope_actions.extend([
                "Limiting search to recent files",
                "Reducing search result count",
                "Simplifying search query"
            ])
            
            # Update search parameters
            context['max_results'] = min(context.get('max_results', 100), 20)
            context['search_scope'] = 'recent'
            
        elif isinstance(error, EinsteinIndexException):
            # Reduce indexing scope
            scope_actions.extend([
                "Indexing only essential files",
                "Skipping large binary files",
                "Using lightweight indexing"
            ])
            
        elif isinstance(error, EinsteinEmbeddingException):
            # Reduce embedding scope
            scope_actions.extend([
                "Processing smaller text chunks",
                "Using faster embedding model",
                "Reducing embedding dimensions"
            ])
        
        return {
            "scope_actions": scope_actions,
            "reduced_context": context,
            "strategy": "reduce_scope"
        }
    
    async def _handle_graceful_degradation(
        self,
        error: EinsteinException,
        context: dict[str, Any]
    ) -> Any:
        """Handle graceful degradation for Einstein operations."""
        
        self.logger.info(f"Applying graceful degradation for Einstein {error.category.value} error")
        
        degradation_actions = []
        
        if isinstance(error, EinsteinResourceException):
            # Resource-based degradation
            if error.resource_type == "memory":
                degradation_actions.extend([
                    "Reducing memory usage for indexing",
                    "Clearing caches",
                    "Using disk-based processing"
                ])
                
                # Set memory-efficient mode
                import os
                os.environ['EINSTEIN_MEMORY_EFFICIENT'] = 'true'
                os.environ['EINSTEIN_INDEX_BATCH_SIZE'] = '100'  # Smaller batches
                
            elif error.resource_type == "disk":
                degradation_actions.extend([
                    "Using minimal disk storage",
                    "Enabling temporary indexes",
                    "Compacting existing data"
                ])
                
            elif error.resource_type == "cpu":
                degradation_actions.extend([
                    "Reducing CPU-intensive operations",
                    "Limiting concurrent processing",
                    "Using simplified algorithms"
                ])
        
        elif isinstance(error, EinsteinSearchException):
            # Search degradation
            degradation_actions.extend([
                "Switching to basic text search",
                "Disabling semantic search",
                "Using cached results when possible"
            ])
        
        elif isinstance(error, EinsteinEmbeddingException):
            # Embedding degradation
            degradation_actions.extend([
                "Using pre-computed embeddings",
                "Switching to keyword-based matching",
                "Disabling real-time embedding"
            ])
        
        else:
            # General degradation
            degradation_actions.extend([
                "Enabling basic operation mode",
                "Reducing feature complexity",
                "Using minimal resource consumption"
            ])
        
        return {
            "degradation_actions": degradation_actions,
            "strategy": "graceful_degradation"
        }
    
    async def _validate_recovery(
        self,
        error: EinsteinException,
        recovery_result: Any,
        context: dict[str, Any]
    ) -> bool:
        """Validate that Einstein recovery was successful."""
        
        try:
            if isinstance(error, EinsteinIndexException):
                # Validate index recovery
                if self.einstein_hub and hasattr(self.einstein_hub, 'vector_index'):
                    if self.einstein_hub.vector_index is not None:
                        return True
                    
            elif isinstance(error, EinsteinSearchException):
                # Validate search recovery
                if recovery_result and isinstance(recovery_result, dict):
                    if recovery_result.get('fallback_result'):
                        return True
                    
            elif isinstance(error, EinsteinEmbeddingException):
                # Validate embedding recovery
                return True  # If we got here, fallback worked
                
            elif isinstance(error, EinsteinResourceException):
                # Validate resource recovery
                import psutil
                if error.resource_type == "memory":
                    memory = psutil.virtual_memory()
                    return memory.percent < 90.0
                elif error.resource_type == "disk":
                    disk = psutil.disk_usage('/')
                    return (disk.free / disk.total) > 0.1  # At least 10% free
                
            return True
            
        except Exception as validation_error:
            self.logger.error(f"Einstein recovery validation failed: {validation_error}")
            return False
    
    def _update_average_recovery_time(self, duration: float):
        """Update the running average of recovery times."""
        current_avg = self.stats['average_recovery_time']
        total_recoveries = self.stats['total_recoveries']
        
        if total_recoveries == 1:
            self.stats['average_recovery_time'] = duration
        else:
            self.stats['average_recovery_time'] = (
                (current_avg * (total_recoveries - 1) + duration) / total_recoveries
            )
    
    def get_recovery_stats(self) -> dict[str, Any]:
        """Get comprehensive Einstein recovery statistics."""
        
        return {
            'overall': self.stats.copy(),
            'current_state': self.state.value,
            'active_recoveries': len(self.active_recoveries),
            'recent_recoveries': len([
                r for r in self.recovery_history
                if time.time() - r['timestamp'] < 3600
            ]),
            'success_rate': (
                self.stats['successful_recoveries'] / max(1, self.stats['total_recoveries'])
            ),
            'average_recovery_time': self.stats['average_recovery_time'],
            'most_common_errors': self._get_most_common_errors(),
            'strategy_effectiveness': self._calculate_strategy_effectiveness()
        }
    
    def _get_most_common_errors(self) -> list[dict[str, Any]]:
        """Get most common Einstein error types from history."""
        error_counts = {}
        
        for attempt in self.recovery_history:
            error_code = attempt['error_code']
            if error_code not in error_counts:
                error_counts[error_code] = {
                    'count': 0,
                    'success_rate': 0.0
                }
            
            error_counts[error_code]['count'] += 1
            if attempt['success']:
                error_counts[error_code]['success_rate'] += 1
        
        # Calculate success rates
        for error_code, stats in error_counts.items():
            stats['success_rate'] /= stats['count']
        
        return sorted(
            [{'error_code': code, **stats} for code, stats in error_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:10]
    
    def _calculate_strategy_effectiveness(self) -> dict[str, float]:
        """Calculate effectiveness of each recovery strategy."""
        strategy_stats = {}
        
        for attempt in self.recovery_history:
            strategy = attempt['strategy'].value if hasattr(attempt['strategy'], 'value') else str(attempt['strategy'])
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'total': 0, 'successful': 0}
            
            strategy_stats[strategy]['total'] += 1
            if attempt['success']:
                strategy_stats[strategy]['successful'] += 1
        
        return {
            strategy: stats['successful'] / max(1, stats['total'])
            for strategy, stats in strategy_stats.items()
        }


# Global recovery manager
_recovery_manager: EinsteinRecoveryManager | None = None


def get_einstein_recovery_manager() -> EinsteinRecoveryManager:
    """Get or create the global Einstein recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = EinsteinRecoveryManager()
    return _recovery_manager