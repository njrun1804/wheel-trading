# Enhanced Error Handling and Logging System

A comprehensive error handling and logging framework designed for the Unity Wheel trading system, providing structured error propagation, async context handling, circuit breaker patterns, and debugging visibility.

## 🎯 Overview

This system addresses critical needs for:
- **Structured Error Information**: Rich context and debugging data
- **Async Error Propagation**: Proper handling in async/await contexts
- **Timeout Management**: Automatic timeout handling with context
- **Circuit Breaker Patterns**: Protection against cascading failures
- **Debugging Visibility**: Comprehensive logging and monitoring
- **Recovery Strategies**: Automated retry and fallback mechanisms

## 📁 System Architecture

```
src/unity_wheel/core_utilities/error_handling/
├── __init__.py              # Main exports and public API
├── exceptions.py            # Enhanced exception classes and decorators
├── logging_enhanced.py      # Structured logging with context tracking
├── recovery.py             # Retry mechanisms and circuit breakers
├── monitoring.py           # Error monitoring and health checking
└── examples.py             # Integration examples and demo code
```

## 🚀 Quick Start

### Basic Error Handling

```python
from unity_wheel.core_utilities.error_handling import (
    error_handler,
    ValidationError,
    get_enhanced_logger
)

logger = get_enhanced_logger("my_component")

@error_handler(
    component="user_service",
    operation="validate_user",
    reraise=True
)
def validate_user(user_data):
    if not user_data.get("email"):
        raise ValidationError(
            "Email is required",
            field="email",
            component="user_service"
        )
    return True
```

### Async Error Handling with Timeout

```python
from unity_wheel.core_utilities.error_handling import (
    async_error_handler,
    AsyncLogContext,
    TimeoutError
)

@async_error_handler(
    component="trading_api",
    operation="get_market_data",
    timeout_seconds=30.0
)
async def get_market_data(symbol: str):
    async with AsyncLogContext(operation="fetch_data", symbol=symbol):
        # Your async operation here
        return await fetch_from_api(symbol)
```

### Retry with Exponential Backoff

```python
from unity_wheel.core_utilities.error_handling import (
    exponential_backoff,
    ExternalServiceError
)

@exponential_backoff(
    max_attempts=3,
    base_delay=1.0,
    retry_on=[ExternalServiceError, ConnectionError]
)
def call_external_api(endpoint):
    # API call that might fail
    return make_request(endpoint)
```

### Circuit Breaker Protection

```python
from unity_wheel.core_utilities.error_handling import circuit_breaker

@circuit_breaker(
    failure_threshold=5,
    timeout_seconds=60.0
)
def unreliable_service():
    # Service that might become unavailable
    return external_service_call()
```

## 🔧 Core Components

### 1. Exception Hierarchy

```python
UnityWheelError (base)
├── ValidationError          # Data validation failures
├── DatabaseError           # Database operation failures  
├── ExternalServiceError    # Third-party service failures
├── TimeoutError           # Operation timeout failures
├── AsyncOperationError    # Async-specific failures
├── ResourceError          # Resource exhaustion
├── ConfigurationError     # Configuration issues
└── CircuitBreakerError   # Circuit breaker activation
```

Each exception includes:
- **Structured Context**: Component, operation, and debugging data
- **Error Codes**: Unique identifiers for tracking
- **Severity Levels**: Critical, High, Medium, Low, Debug
- **Recovery Hints**: Suggestions for resolution
- **Timestamp Tracking**: When errors occurred
- **Cause Chaining**: Original exception preservation

### 2. Enhanced Logging

```python
from unity_wheel.core_utilities.error_handling import (
    get_enhanced_logger,
    LogContext,
    structured_error_log
)

logger = get_enhanced_logger("component_name")

# Structured logging with context
with LogContext(user_id="12345", operation="process_order"):
    logger.info("Processing order", extra={
        "order_id": "ORD-001",
        "amount": 1000.00,
        "symbol": "AAPL"
    })

# Performance logging
logger.performance("database_query", duration_ms=45.2, rows_affected=10)

# Error logging with full context
try:
    risky_operation()
except Exception as e:
    structured_error_log(logger, e, order_id="ORD-001", user_id="12345")
```

### 3. Recovery and Retry System

```python
from unity_wheel.core_utilities.error_handling import (
    RetryConfig,
    BackoffStrategy,
    with_retry,
    async_with_retry
)

# Custom retry configuration
config = RetryConfig(
    max_attempts=5,
    base_delay_seconds=2.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    retry_on_exceptions=[ConnectionError, TimeoutError],
    stop_on_exceptions=[ValidationError]
)

@with_retry(config)
def database_operation():
    return execute_query("SELECT * FROM orders")

@async_with_retry(config)
async def async_api_call():
    return await call_trading_api()
```

### 4. Error Monitoring and Alerting

```python
from unity_wheel.core_utilities.error_handling import (
    get_error_monitor,
    get_health_checker,
    track_error_patterns
)

# Error monitoring
error_monitor = get_error_monitor()
metrics = error_monitor.get_metrics()
patterns = error_monitor.get_patterns()
alerts = error_monitor.get_alerts()

# Health checking
health_checker = get_health_checker()
health_status = health_checker.get_health_status()

# Custom health check
def check_database_health():
    return {
        "database_connections": get_active_connections(),
        "response_time_ms": measure_query_time()
    }

health_checker.add_health_check(check_database_health)
```

## 📊 Monitoring and Metrics

### Error Metrics
- **Error Count**: Total errors over time
- **Error Rate**: Errors per minute
- **Error Types**: Distribution by exception type
- **Component Errors**: Errors by system component
- **Pattern Detection**: Recurring error patterns
- **Recovery Success**: Retry and recovery statistics

### Health Metrics
- **System Status**: Healthy, Degraded, Critical, Down
- **Success Rate**: Percentage of successful operations
- **Response Times**: Average operation duration
- **Resource Usage**: Memory, CPU, connections
- **Uptime**: System availability metrics

### Performance Tracking
- **Operation Timing**: Function execution times
- **Async Context**: Task and coroutine tracking
- **Cache Performance**: Hit rates and memory usage
- **Parallel Processing**: Core utilization metrics

## 🔄 Integration Examples

### Trading API Client

```python
from unity_wheel.core_utilities.error_handling import *

class TradingAPIClient:
    def __init__(self):
        self.logger = get_enhanced_logger("trading_api_client")
    
    @async_error_handler(
        component="trading_api_client",
        timeout_seconds=30.0
    )
    @exponential_backoff(max_attempts=3)
    @log_execution_time("api_call")
    async def get_market_data(self, symbol: str):
        async with AsyncLogContext(operation="get_market_data", symbol=symbol):
            try:
                data = await self._fetch_data(symbol)
                return data
            except Exception as e:
                if isinstance(e, UnityWheelError):
                    track_error_patterns(e)
                raise
```

### Database Manager

```python
class DatabaseManager:
    @error_handler(component="database_manager", reraise=True)
    @exponential_backoff(max_attempts=3, base_delay=1.0)
    def execute_query(self, query: str, params: dict = None):
        try:
            return self._execute(query, params)
        except Exception as e:
            raise DatabaseError(
                f"Query execution failed: {str(e)}",
                query=query[:100],  # Truncated for security
                operation="execute_query",
                cause=e
            )
```

### Resource Manager

```python
class ResourceManager:
    @error_handler(component="resource_manager")
    def allocate_memory(self, amount_mb: int):
        if self.current_usage + amount_mb > self.limit:
            raise ResourceError(
                f"Memory allocation failed: {amount_mb}MB requested",
                resource_type="memory",
                current_usage=self.current_usage,
                limit=self.limit
            )
```

## 🛠️ Configuration

### Logging Configuration

```python
# Enable structured logging globally
import logging
from unity_wheel.core_utilities.error_handling import get_enhanced_logger

# Set log levels
logging.getLogger("unity_wheel").setLevel(logging.INFO)
logging.getLogger("unity_wheel.trading").setLevel(logging.DEBUG)

# Configure handlers (JSON, console, file)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(handler)
```

### Error Monitor Configuration

```python
from unity_wheel.core_utilities.error_handling import get_error_monitor

error_monitor = get_error_monitor()

# Configure alert thresholds
error_monitor.error_rate_threshold = 20  # errors per minute
error_monitor.critical_error_threshold = 10  # critical errors per hour
error_monitor.pattern_threshold = 3  # occurrences to detect pattern

# Add custom alert handler
def slack_alert_handler(alert):
    send_slack_message(f"🚨 {alert.title}: {alert.message}")

error_monitor.add_alert_callback(slack_alert_handler)
```

### Circuit Breaker Configuration

```python
from unity_wheel.core_utilities.error_handling import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,        # Failures before opening
    recovery_timeout_seconds=60.0,  # Time before retry
    success_threshold=2,        # Successes to close
    half_open_max_calls=3      # Max calls in half-open state
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_enhanced_error_handling.py
```

Test coverage includes:
- Structured logging functionality
- Error monitoring and pattern detection
- Health checking and metrics
- Recovery and retry mechanisms
- Async error handling
- Component integration

## 📈 Performance Impact

The enhanced error handling system is designed for minimal performance overhead:

- **Logging**: < 1ms per log entry with caching
- **Error Creation**: < 0.1ms per exception with context
- **Monitoring**: Async processing, no blocking
- **Recovery**: Exponential backoff prevents resource waste
- **Circuit Breakers**: Fast-fail protection reduces load

## 🔒 Security Considerations

- **Sensitive Data**: Automatic filtering of sensitive information
- **Log Sanitization**: SQL queries and API keys are truncated/masked
- **Context Isolation**: Thread-safe and async-safe context handling
- **Error Exposure**: User-friendly messages vs. detailed internal logs

## 🚀 Migration Guide

### From Basic Exception Handling

**Before:**
```python
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
    return None
```

**After:**
```python
@error_handler(component="my_service", operation="risky_operation")
def safe_operation():
    try:
        return risky_operation()
    except Exception as e:
        raise UnityWheelError(
            f"Operation failed: {str(e)}",
            cause=e,
            component="my_service",
            operation="risky_operation"
        )
```

### From Basic Logging

**Before:**
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Processing user data")
```

**After:**
```python
from unity_wheel.core_utilities.error_handling import get_enhanced_logger

logger = get_enhanced_logger(__name__)
logger.info("Processing user data", extra={
    "user_id": user_id,
    "operation": "process_data",
    "timestamp": time.time()
})
```

## 🤝 Contributing

When adding new components:

1. **Use Enhanced Exceptions**: Inherit from `UnityWheelError`
2. **Add Structured Logging**: Use `get_enhanced_logger()`
3. **Include Error Decorators**: Apply `@error_handler` or `@async_error_handler`
4. **Track Patterns**: Call `track_error_patterns()` for monitoring
5. **Add Health Checks**: Register with `get_health_checker()`

## 📚 API Reference

### Core Functions

- `get_enhanced_logger(name)` - Get structured logger instance
- `get_error_monitor()` - Get global error monitoring
- `get_health_checker()` - Get system health checker
- `get_recovery_manager()` - Get retry/recovery manager
- `track_error_patterns(error)` - Track error for pattern detection

### Decorators

- `@error_handler(...)` - Sync function error handling
- `@async_error_handler(...)` - Async function error handling
- `@timeout_handler(seconds)` - Add timeout protection
- `@circuit_breaker(...)` - Circuit breaker protection
- `@exponential_backoff(...)` - Exponential retry backoff
- `@linear_backoff(...)` - Linear retry backoff
- `@log_execution_time(operation)` - Performance logging

### Context Managers

- `LogContext(**context)` - Sync logging context
- `AsyncLogContext(**context)` - Async logging context

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```python
# If you get import errors, ensure the path is correct:
from src.unity_wheel.core_utilities.error_handling import get_enhanced_logger
```

**Circular Imports:**
```python
# Use lazy imports in __init__.py files:
def get_component():
    from .module import Component
    return Component()
```

**Async Context Issues:**
```python
# Use AsyncLogContext for async operations:
async with AsyncLogContext(operation="async_work"):
    await async_operation()
```

**Memory Usage:**
```python
# Configure cache limits in components:
analyzer = PythonAnalysisTurbo()
analyzer.cache_size_mb = 512  # Limit cache to 512MB
```

## 📝 Changelog

### Version 1.0.0
- Initial release with comprehensive error handling
- Structured logging with context tracking
- Async error propagation and timeout handling
- Circuit breaker patterns and recovery strategies
- Error monitoring and health checking
- Integration examples and documentation

---

For questions or support, refer to the examples in `examples.py` or run the test suite to see the system in action.