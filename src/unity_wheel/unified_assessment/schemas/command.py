"""
Command schemas for the Unified Assessment Engine.

Defines the data structures for commands, results, and metrics
that flow through the unified assessment pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class CommandStatus(Enum):
    """Status of command processing."""
    PENDING = "pending"
    GATHERING_CONTEXT = "gathering_context"
    ANALYZING_INTENT = "analyzing_intent"
    PLANNING_ACTIONS = "planning_actions"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class IntentCategory(Enum):
    """Categories of user intent."""
    FIX = "fix"
    CREATE = "create"
    OPTIMIZE = "optimize"
    ANALYZE = "analyze"
    REFACTOR = "refactor"
    TEST = "test"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    QUERY = "query"
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Types of tasks in execution plans."""
    SEARCH = "search"
    ANALYZE = "analyze"
    MODIFY = "modify" 
    CREATE = "create"
    DELETE = "delete"
    VALIDATE = "validate"
    TEST = "test"
    DEPLOY = "deploy"
    MONITOR = "monitor"


@dataclass
class CommandMetrics:
    """Performance metrics for command processing."""
    
    # Timing metrics (in milliseconds)
    total_duration_ms: float = 0.0
    context_gathering_ms: float = 0.0
    intent_analysis_ms: float = 0.0
    action_planning_ms: float = 0.0
    execution_ms: float = 0.0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Success metrics
    context_confidence: float = 0.0
    intent_confidence: float = 0.0
    execution_success_rate: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Execution metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    files_modified: int = 0
    lines_changed: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandError:
    """Error information for failed commands."""
    
    error_type: str
    error_message: str
    error_details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandResult:
    """Result of processing a command through the unified assessment engine."""
    
    # Command identification
    command_id: str = field(default_factory=lambda: str(uuid4()))
    original_command: str = ""
    
    # Processing status
    status: CommandStatus = CommandStatus.PENDING
    success: bool = False
    
    # Results
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    
    # Files and changes
    files_affected: List[str] = field(default_factory=list)
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    
    # Errors and warnings
    errors: List[CommandError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    metrics: CommandMetrics = field(default_factory=CommandMetrics)
    context_data: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def mark_completed(self, success: bool = True):
        """Mark the command as completed."""
        self.status = CommandStatus.COMPLETED if success else CommandStatus.FAILED
        self.success = success
        self.completed_at = datetime.now()
        self.metrics.total_duration_ms = (
            (self.completed_at - self.created_at).total_seconds() * 1000
        )
    
    def add_error(self, error_type: str, message: str, **kwargs):
        """Add an error to the result."""
        error = CommandError(
            error_type=error_type,
            error_message=message,
            error_details=kwargs
        )
        self.errors.append(error)
        
        if self.status != CommandStatus.FAILED:
            self.status = CommandStatus.PARTIALLY_COMPLETED
    
    def add_finding(self, finding: str):
        """Add a finding to the result."""
        self.findings.append(finding)
    
    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the result."""
        self.recommendations.append(recommendation)
    
    def add_action(self, action: str):
        """Add an action taken to the result."""
        self.actions_taken.append(action)
    
    def log_execution_step(self, step: str, details: Dict[str, Any] = None):
        """Log an execution step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details or {}
        }
        self.execution_log.append(log_entry)
    
    def get_duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate based on completed vs failed tasks."""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks == 0:
            return 1.0 if self.success else 0.0
        return self.metrics.tasks_completed / total_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "command_id": self.command_id,
            "original_command": self.original_command,
            "status": self.status.value,
            "success": self.success,
            "summary": self.summary,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "actions_taken": self.actions_taken,
            "files_affected": self.files_affected,
            "changes_made": self.changes_made,
            "errors": [
                {
                    "type": err.error_type,
                    "message": err.error_message,
                    "details": err.error_details
                }
                for err in self.errors
            ],
            "warnings": self.warnings,
            "metrics": {
                "total_duration_ms": self.metrics.total_duration_ms,
                "context_gathering_ms": self.metrics.context_gathering_ms,
                "intent_analysis_ms": self.metrics.intent_analysis_ms,
                "action_planning_ms": self.metrics.action_planning_ms,
                "execution_ms": self.metrics.execution_ms,
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "gpu_usage_percent": self.metrics.gpu_usage_percent,
                "context_confidence": self.metrics.context_confidence,
                "intent_confidence": self.metrics.intent_confidence,
                "execution_success_rate": self.metrics.execution_success_rate,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "files_modified": self.metrics.files_modified,
                "lines_changed": self.metrics.lines_changed
            },
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.get_duration_seconds(),
            "success_rate": self.get_success_rate()
        }


@dataclass
class Target:
    """Target for an intent (what to act upon)."""
    
    target_type: str  # file, function, class, module, system
    identifier: str   # name or path
    location: Optional[str] = None  # file path if applicable
    line_range: Optional[tuple[int, int]] = None  # (start, end) line numbers
    confidence: float = 1.0


@dataclass
class Constraint:
    """Constraint for an intent (limitations or requirements)."""
    
    constraint_type: str  # compatibility, performance, safety, style
    description: str
    severity: str = "medium"  # low, medium, high, critical
    enforced: bool = True