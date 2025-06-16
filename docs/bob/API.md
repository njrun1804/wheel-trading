# BOB API Reference

## Core Classes

### BoltIntegration

The main entry point for BOB system integration.

```python
class BoltIntegration:
    """Ultra-fast integration layer for the 8-agent Bolt system."""
    
    def __init__(
        self,
        num_agents: int = 8,
        config_path: Optional[str] = None,
        debug: bool = False,
        trace_tasks: bool = False
    ):
        """
        Initialize BOB system.
        
        Args:
            num_agents: Number of parallel agents (default: 8)
            config_path: Path to configuration file
            debug: Enable debug logging
            trace_tasks: Enable detailed task tracing
        """
```

#### Methods

##### initialize()

```python
async def initialize(self) -> None:
    """
    Initialize all BOB components.
    
    Performs parallel initialization of:
    - Hardware detection and optimization
    - Agent pool creation
    - Einstein search index
    - Accelerated tools
    - Monitoring systems
    
    Raises:
        BoltException: If initialization fails
    """
```

##### execute_query()

```python
async def execute_query(
    self,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None
) -> QueryResult:
    """
    Execute a natural language query.
    
    Args:
        query: Natural language query
        context: Additional context for query
        timeout: Maximum execution time in seconds
        
    Returns:
        QueryResult object containing:
        - query: Original query
        - tasks: List of executed tasks
        - results: Task results
        - total_duration: Total execution time
        - system_state: System metrics
        
    Raises:
        BoltTimeoutException: If timeout exceeded
        BoltTaskException: If task execution fails
    """
```

##### analyze_query()

```python
async def analyze_query(
    self,
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
    """
    Analyze query without execution.
    
    Args:
        query: Natural language query
        context: Additional context
        
    Returns:
        AnalysisResult with:
        - relevant_files: List of relevant files
        - tasks: Planned task decomposition
        - estimated_duration: Time estimate
        - resource_requirements: CPU/GPU/memory needs
    """
```

##### shutdown()

```python
async def shutdown(self) -> None:
    """
    Gracefully shutdown BOB system.
    
    Performs:
    - Agent pool termination
    - Resource cleanup
    - Cache persistence
    - Metric export
    """
```

### AgentOrchestrator

Manages the pool of parallel agents.

```python
class AgentOrchestrator:
    """Ultra-fast orchestrator for parallel agents."""
    
    def __init__(
        self,
        num_agents: int = 8,
        enable_work_stealing: bool = True,
        task_batch_size: int = 16
    ):
        """
        Initialize orchestrator.
        
        Args:
            num_agents: Number of agents in pool
            enable_work_stealing: Enable work stealing
            task_batch_size: Tasks per batch
        """
```

#### Methods

##### execute_tasks()

```python
async def execute_tasks(
    self,
    tasks: List[Task]
) -> List[TaskResult]:
    """
    Execute tasks in parallel across agents.
    
    Args:
        tasks: List of tasks to execute
        
    Returns:
        List of TaskResult objects
        
    Features:
    - Automatic load balancing
    - Work stealing
    - Dependency resolution
    - Error recovery
    """
```

##### get_orchestrator_stats()

```python
def get_orchestrator_stats(self) -> Dict[str, Any]:
    """
    Get comprehensive performance statistics.
    
    Returns:
        Dict with:
        - total_tasks: Total tasks executed
        - success_rate: Percentage of successful tasks
        - avg_task_duration: Average task time
        - throughput: Tasks per second
        - agent_utilization: Per-agent statistics
    """
```

### Task Types

#### Task

```python
@dataclass
class Task:
    """Represents a unit of work."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    
class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
```

#### TaskResult

```python
@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    agent_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
```

### Hardware Monitoring

#### HardwareStateMonitor

```python
class HardwareStateMonitor:
    """Real-time hardware monitoring."""
    
    async def get_system_state(self) -> SystemState:
        """
        Get current system state.
        
        Returns:
            SystemState with:
            - cpu_percent: CPU usage
            - memory_percent: Memory usage
            - gpu_percent: GPU usage
            - gpu_memory_used_gb: GPU memory
            - thermal_state: Thermal pressure
            - power_state: Power consumption
        """
    
    async def get_resource_availability(self) -> ResourceAvailability:
        """Check available resources for task execution."""
```

### Accelerated Tools

#### RipgrepTurbo

```python
class RipgrepTurbo:
    """Hardware-accelerated text search."""
    
    async def search(
        self,
        pattern: str,
        path: str = ".",
        file_type: Optional[str] = None,
        max_results: int = 1000
    ) -> List[SearchResult]:
        """
        Parallel text search across files.
        
        Args:
            pattern: Search pattern (regex supported)
            path: Search path
            file_type: Filter by file type
            max_results: Maximum results
            
        Returns:
            List of SearchResult objects
        """
    
    async def parallel_search(
        self,
        patterns: List[str],
        path: str = "."
    ) -> Dict[str, List[SearchResult]]:
        """Search multiple patterns in parallel."""
```

#### PythonAnalyzer

```python
class PythonAnalyzer:
    """MLX-accelerated Python code analysis."""
    
    async def analyze_file(
        self,
        file_path: str
    ) -> FileAnalysis:
        """
        Analyze Python file.
        
        Returns:
            FileAnalysis with:
            - imports: Import statements
            - functions: Function definitions
            - classes: Class definitions
            - complexity: Cyclomatic complexity
            - issues: Potential issues
        """
    
    async def analyze_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> DirectoryAnalysis:
        """Analyze entire directory in parallel."""
```

#### DuckDBTurbo

```python
class DuckDBTurbo:
    """Native DuckDB integration."""
    
    async def query(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute SQL query."""
    
    async def query_to_pandas(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> pd.DataFrame:
        """Execute query and return pandas DataFrame."""
```

### Error Handling

#### Exception Types

```python
class BoltException(Exception):
    """Base exception for BOB system."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity,
        category: ErrorCategory,
        recovery_hints: Optional[List[str]] = None,
        diagnostic_data: Optional[Dict[str, Any]] = None
    ):
        """Initialize exception with recovery information."""

class BoltTimeoutException(BoltException):
    """Timeout during execution."""

class BoltResourceException(BoltException):
    """Resource exhaustion."""

class BoltTaskException(BoltException):
    """Task execution failure."""
```

#### Error Severity

```python
class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1      # Informational
    MEDIUM = 2   # Warning
    HIGH = 3     # Error
    CRITICAL = 4 # System failure
```

### Configuration

#### Config Schema

```python
@dataclass
class BoltConfig:
    """BOB system configuration."""
    
    # Agent settings
    num_agents: int = 8
    enable_work_stealing: bool = True
    task_batch_size: int = 16
    
    # Hardware settings
    cpu_cores: Optional[int] = None
    gpu_backend: str = "auto"  # auto, metal, mlx, none
    memory_limit_gb: Optional[float] = None
    
    # Tool settings
    tool_cache_size_mb: int = 2048
    tool_parallelism: int = 12
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: float = 1.0
    
    # Error handling
    circuit_breaker_threshold: int = 5
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
```

### Async Context Managers

#### Resource Guards

```python
async with bob.resource_guard(
    memory_limit_gb=10,
    cpu_limit_percent=80
) as guard:
    # Execute resource-intensive operation
    result = await bob.execute_query("complex task")
```

#### Performance Monitoring

```python
async with bob.performance_monitor() as monitor:
    # Execute operations
    result = await bob.execute_query("analyze codebase")
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"Peak memory: {metrics.peak_memory_gb:.1f}GB")
    print(f"Avg CPU: {metrics.avg_cpu_percent:.1f}%")
```

### Utility Functions

#### System Status

```python
def get_system_status() -> Dict[str, Any]:
    """
    Get current BOB system status.
    
    Returns:
        Dict with:
        - hardware: Hardware configuration
        - gpu_backend: Active GPU backend
        - memory_gb: Available memory
        - cpu_cores: CPU core count
        - accelerated_tools: Tool availability
    """
```

#### Performance Benchmarks

```python
async def run_performance_benchmark(
    quick: bool = True
) -> BenchmarkResults:
    """
    Run system performance benchmarks.
    
    Args:
        quick: Run quick benchmark (default: True)
        
    Returns:
        BenchmarkResults with performance metrics
    """
```

## Usage Examples

### Basic Query Execution

```python
import asyncio
from bolt.core.integration import BoltIntegration

async def main():
    # Initialize BOB
    bob = BoltIntegration(num_agents=8)
    await bob.initialize()
    
    try:
        # Execute query
        result = await bob.execute_query(
            "find and fix performance issues in the trading module"
        )
        
        # Process results
        for task_result in result.results:
            print(f"{task_result.task}: {task_result.status}")
            
    finally:
        await bob.shutdown()

asyncio.run(main())
```

### Advanced Configuration

```python
from bolt.core.config import BoltConfig

# Custom configuration
config = BoltConfig(
    num_agents=6,
    gpu_backend="metal",
    memory_limit_gb=16,
    tool_cache_size_mb=4096,
    circuit_breaker_threshold=3
)

# Initialize with config
bob = BoltIntegration(config=config)
```

### Error Handling

```python
try:
    result = await bob.execute_query(
        "complex analysis task",
        timeout=30.0
    )
except BoltTimeoutException as e:
    print(f"Task timed out: {e.message}")
    print(f"Recovery hints: {e.recovery_hints}")
except BoltResourceException as e:
    print(f"Resource exhausted: {e.message}")
    # Reduce resource usage and retry
except BoltException as e:
    print(f"General error: {e.message}")
    if e.severity == ErrorSeverity.CRITICAL:
        # System failure, shutdown required
        await bob.shutdown()
```

### Trading Integration

```python
from bolt.trading_system_integration import TradingBoltIntegration

# Create trading-specific instance
trading_bob = TradingBoltIntegration(
    num_agents=8,
    trading_database="data/wheel_trading.duckdb"
)

# Analyze positions
result = await trading_bob.analyze_trading_positions(
    "evaluate current wheel positions and recommend adjustments"
)

# Get trade signals
signals = await trading_bob.generate_trade_signals(
    symbol="U",
    strategy="wheel"
)