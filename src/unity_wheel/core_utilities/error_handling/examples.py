"""
Error Handling Integration Examples

Demonstrates how to integrate the enhanced error handling system
across different components of the Unity Wheel trading system.
"""

import asyncio
import time
from typing import Any

# Import our enhanced error handling system
from . import (  # Exceptions; Decorators; Logging; Recovery; Monitoring
    AsyncLogContext,
    BackoffStrategy,
    DatabaseError,
    ErrorLogger,
    ExternalServiceError,
    LogContext,
    ResourceError,
    RetryConfig,
    TimeoutError,
    UnityWheelError,
    ValidationError,
    alert_on_error,
    async_error_handler,
    async_health_check_decorator,
    async_with_retry,
    circuit_breaker,
    error_handler,
    exponential_backoff,
    get_enhanced_logger,
    get_error_monitor,
    get_health_checker,
    get_recovery_manager,
    log_execution_time,
    log_with_context,
    timeout_handler,
    track_error_patterns,
)


class TradingAPIClient:
    """Example trading API client with comprehensive error handling."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.logger = get_enhanced_logger("trading_api_client")
        self.error_logger = ErrorLogger(self.logger)

        # Register health check
        health_checker = get_health_checker()
        health_checker.add_health_check(self._health_check)

    @async_error_handler(
        component="trading_api_client",
        timeout_seconds=30.0,
        catch_types=[ConnectionError, TimeoutError],
    )
    @async_with_retry(
        RetryConfig(
            max_attempts=3,
            base_delay_seconds=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retry_on_exceptions=[ConnectionError, TimeoutError, ExternalServiceError],
        )
    )
    @log_execution_time("api_call")
    @async_health_check_decorator
    async def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data with comprehensive error handling."""
        async with AsyncLogContext(operation="get_market_data", symbol=symbol):
            self.logger.info(f"Fetching market data for {symbol}")

            try:
                # Simulate API call
                await asyncio.sleep(0.1)  # Simulate network delay

                # Simulate occasional failures for demonstration
                import random

                if random.random() < 0.1:  # 10% failure rate
                    raise ExternalServiceError(
                        f"Market data service temporarily unavailable for {symbol}",
                        service_name="market_data_api",
                        endpoint=f"/market/{symbol}",
                        status_code=503,
                        component="trading_api_client",
                        operation="get_market_data",
                        context={"symbol": symbol},
                    )

                # Return mock data
                return {
                    "symbol": symbol,
                    "price": 150.00,
                    "volume": 1000000,
                    "timestamp": time.time(),
                }

            except Exception as e:
                # Enhanced error logging
                self.error_logger.log_error(
                    e,
                    component="trading_api_client",
                    operation="get_market_data",
                    additional_context={"symbol": symbol},
                )

                # Track error patterns
                if isinstance(e, UnityWheelError):
                    track_error_patterns(e)
                    alert_on_error(e)

                raise

    @circuit_breaker(failure_threshold=5, timeout_seconds=60.0)
    @exponential_backoff(max_attempts=3, base_delay=2.0)
    @log_with_context(component="trading_api_client", operation="place_order")
    def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Place trading order with circuit breaker protection."""
        try:
            # Validate order data
            self._validate_order(order_data)

            # Log order attempt
            self.logger.info(
                "Placing order",
                extra={
                    "order_id": order_data.get("id"),
                    "symbol": order_data.get("symbol"),
                    "quantity": order_data.get("quantity"),
                },
            )

            # Simulate order placement
            time.sleep(0.05)  # Simulate processing delay

            # Simulate occasional failures
            import random

            if random.random() < 0.05:  # 5% failure rate
                raise ExternalServiceError(
                    "Order placement service unavailable",
                    service_name="order_api",
                    endpoint="/orders",
                    status_code=503,
                    component="trading_api_client",
                    operation="place_order",
                )

            return {
                "order_id": order_data["id"],
                "status": "filled",
                "timestamp": time.time(),
            }

        except ValidationError:
            # Don't retry validation errors
            raise
        except Exception as e:
            # Log and track error
            if not isinstance(e, UnityWheelError):
                error = ExternalServiceError(
                    f"Order placement failed: {str(e)}",
                    cause=e,
                    component="trading_api_client",
                    operation="place_order",
                    context={"order_data": order_data},
                )
            else:
                error = e

            self.error_logger.log_error(error)
            track_error_patterns(error)
            raise error

    def _validate_order(self, order_data: dict[str, Any]) -> None:
        """Validate order data."""
        required_fields = ["id", "symbol", "quantity", "side"]

        for field in required_fields:
            if field not in order_data or order_data[field] is None:
                raise ValidationError(
                    f"Missing required field: {field}",
                    field=field,
                    component="trading_api_client",
                    operation="validate_order",
                    context={"order_data": order_data},
                )

        if order_data["quantity"] <= 0:
            raise ValidationError(
                "Order quantity must be positive",
                field="quantity",
                value=order_data["quantity"],
                constraint="quantity > 0",
                component="trading_api_client",
            )

        if order_data["side"] not in ["buy", "sell"]:
            raise ValidationError(
                "Invalid order side",
                field="side",
                value=order_data["side"],
                constraint="side in ['buy', 'sell']",
                component="trading_api_client",
            )

    def _health_check(self) -> dict[str, Any]:
        """Health check for the API client."""
        try:
            # Simulate health check
            return {
                "api_client_healthy": True,
                "response_time_ms": 50.0,
                "active_connections": 5,
            }
        except Exception:
            return {
                "api_client_healthy": False,
                "response_time_ms": 0.0,
                "active_connections": 0,
            }


class DatabaseManager:
    """Example database manager with comprehensive error handling."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = get_enhanced_logger("database_manager")
        self.connection_pool = None
        self._connected = False

    @error_handler(component="database_manager", operation="connect", reraise=True)
    def connect(self) -> None:
        """Connect to database with error handling."""
        try:
            self.logger.info("Connecting to database")

            # Simulate connection
            time.sleep(0.1)

            # Simulate occasional connection failures
            import random

            if random.random() < 0.02:  # 2% failure rate
                raise DatabaseError(
                    "Failed to connect to database",
                    operation="connect",
                    component="database_manager",
                    context={"connection_string": self.connection_string[:20] + "..."},
                )

            self._connected = True
            self.logger.info("Database connected successfully")

        except Exception as e:
            if not isinstance(e, UnityWheelError):
                error = DatabaseError(
                    f"Database connection failed: {str(e)}",
                    cause=e,
                    operation="connect",
                    component="database_manager",
                )
                raise error
            raise

    @async_error_handler(
        component="database_manager",
        timeout_seconds=30.0,
        catch_types=[DatabaseError, TimeoutError],
    )
    @exponential_backoff(max_attempts=3, base_delay=1.0)
    @log_execution_time("database_query")
    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute database query with comprehensive error handling."""
        if not self._connected:
            raise DatabaseError(
                "Database not connected",
                operation="execute_query",
                component="database_manager",
                recovery_hint="Call connect() first",
            )

        async with AsyncLogContext(
            operation="execute_query",
            query_hash=hash(query) % 10000,  # Don't log full query for security
            param_count=len(params) if params else 0,
        ):
            try:
                self.logger.debug_context(
                    "Executing database query",
                    query_length=len(query),
                    has_params=bool(params),
                )

                # Simulate query execution
                await asyncio.sleep(0.05)

                # Simulate occasional query failures
                import random

                if random.random() < 0.03:  # 3% failure rate
                    raise DatabaseError(
                        "Query execution failed",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        operation="execute_query",
                        component="database_manager",
                        context={"param_count": len(params) if params else 0},
                    )

                # Return mock results
                return [{"id": 1, "data": "sample"}]

            except Exception as e:
                # Enhanced error logging with context
                self.logger.error(
                    "Database query failed",
                    extra={
                        "query_length": len(query),
                        "has_params": bool(params),
                        "connection_status": self._connected,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

                # Track error patterns
                if isinstance(e, UnityWheelError):
                    track_error_patterns(e)

                raise


class ResourceManager:
    """Example resource manager with memory and CPU monitoring."""

    def __init__(self, max_memory_mb: int = 1000, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.current_memory_mb = 0
        self.current_cpu_percent = 0.0
        self.logger = get_enhanced_logger("resource_manager")

        # Register health check
        health_checker = get_health_checker()
        health_checker.add_health_check(self._resource_health_check)

    @error_handler(
        component="resource_manager", catch_types=[ResourceError], reraise=True
    )
    def allocate_memory(self, amount_mb: int) -> bool:
        """Allocate memory with resource checking."""
        with LogContext(operation="allocate_memory", amount_mb=amount_mb):
            if self.current_memory_mb + amount_mb > self.max_memory_mb:
                raise ResourceError(
                    f"Insufficient memory: requested {amount_mb}MB, available {self.max_memory_mb - self.current_memory_mb}MB",
                    resource_type="memory",
                    current_usage=self.current_memory_mb,
                    limit=self.max_memory_mb,
                    component="resource_manager",
                    operation="allocate_memory",
                    context={"requested_mb": amount_mb},
                )

            self.current_memory_mb += amount_mb
            self.logger.info(
                f"Allocated {amount_mb}MB memory, total: {self.current_memory_mb}MB"
            )
            return True

    @timeout_handler(timeout_seconds=5.0)
    async def intensive_operation(self, duration_seconds: float = 2.0) -> str:
        """Simulate CPU-intensive operation with timeout protection."""
        async with AsyncLogContext(
            operation="intensive_operation", duration_seconds=duration_seconds
        ):
            self.logger.info(f"Starting intensive operation for {duration_seconds}s")

            # Check CPU usage
            if self.current_cpu_percent > self.max_cpu_percent:
                raise ResourceError(
                    f"CPU usage too high: {self.current_cpu_percent}% > {self.max_cpu_percent}%",
                    resource_type="cpu",
                    current_usage=self.current_cpu_percent,
                    limit=self.max_cpu_percent,
                    component="resource_manager",
                    operation="intensive_operation",
                )

            # Simulate work
            start_time = time.perf_counter()
            while (time.perf_counter() - start_time) < duration_seconds:
                await asyncio.sleep(0.1)
                # Simulate CPU usage increase
                self.current_cpu_percent = min(100.0, self.current_cpu_percent + 5.0)

            # Reset CPU usage
            self.current_cpu_percent = max(0.0, self.current_cpu_percent - 20.0)

            return "Operation completed successfully"

    def _resource_health_check(self) -> dict[str, Any]:
        """Health check for resource usage."""
        memory_usage_percent = (self.current_memory_mb / self.max_memory_mb) * 100
        cpu_usage_percent = self.current_cpu_percent

        return {
            "memory_usage_percent": memory_usage_percent,
            "cpu_usage_percent": cpu_usage_percent,
            "memory_available_mb": self.max_memory_mb - self.current_memory_mb,
            "resource_health": "good"
            if memory_usage_percent < 80 and cpu_usage_percent < 80
            else "degraded",
        }


async def demo_error_handling():
    """Demonstrate the enhanced error handling system."""
    print("🔧 Enhanced Error Handling System Demo")
    print("=" * 50)

    # Initialize components
    api_client = TradingAPIClient("https://api.example.com")
    db_manager = DatabaseManager("postgresql://localhost/trading")
    resource_manager = ResourceManager(max_memory_mb=500)

    # Connect to database
    try:
        db_manager.connect()
        print("✅ Database connected")
    except DatabaseError as e:
        print(f"❌ Database connection failed: {e}")

    # Test API calls with retry logic
    print("\n📊 Testing API calls with retry logic...")
    for i in range(5):
        try:
            market_data = await api_client.get_market_data("AAPL")
            print(
                f"✅ Market data retrieved: {market_data['symbol']} @ ${market_data['price']}"
            )
        except ExternalServiceError as e:
            print(f"❌ API call failed: {e.message}")

    # Test order placement with circuit breaker
    print("\n📈 Testing order placement with circuit breaker...")
    orders = [
        {"id": f"order_{i}", "symbol": "AAPL", "quantity": 100, "side": "buy"}
        for i in range(3)
    ]

    for order in orders:
        try:
            result = api_client.place_order(order)
            print(f"✅ Order placed: {result['order_id']} - {result['status']}")
        except (ExternalServiceError, ValidationError) as e:
            print(f"❌ Order failed: {e.message}")

    # Test database queries
    print("\n🗄️ Testing database queries...")
    try:
        results = await db_manager.execute_query(
            "SELECT * FROM trades WHERE symbol = %s", {"symbol": "AAPL"}
        )
        print(f"✅ Query executed, got {len(results)} results")
    except DatabaseError as e:
        print(f"❌ Query failed: {e.message}")

    # Test resource management
    print("\n💾 Testing resource management...")
    try:
        resource_manager.allocate_memory(200)
        print("✅ Memory allocated successfully")

        result = await resource_manager.intensive_operation(1.0)
        print(f"✅ Intensive operation completed: {result}")

        # Try to allocate too much memory
        resource_manager.allocate_memory(500)  # Should fail

    except ResourceError as e:
        print(f"❌ Resource error: {e.message}")
    except TimeoutError as e:
        print(f"⏰ Operation timed out: {e.message}")

    # Show monitoring statistics
    print("\n📈 System Monitoring Statistics")
    print("-" * 30)

    # Error monitoring
    error_monitor = get_error_monitor()
    metrics = error_monitor.get_metrics()
    print(f"Total errors: {metrics.error_count}")
    print(f"Error rate: {metrics.error_rate_per_minute}/min")
    print(f"Most frequent error: {metrics.most_frequent_error}")

    # Health status
    health_checker = get_health_checker()
    health = health_checker.get_health_status()
    print(f"System status: {health.status.value}")
    print(f"Success rate: {health.success_rate:.2%}")
    print(f"Average response time: {health.average_response_time_ms:.2f}ms")

    # Recovery statistics
    recovery_manager = get_recovery_manager()
    recovery_stats = recovery_manager.get_all_stats()
    print(f"Circuit breakers: {len(recovery_stats['circuit_breakers'])}")
    print(f"Retry operations tracked: {len(recovery_stats['retry_operations'])}")

    # Alert information
    alerts = error_monitor.get_alerts()
    print(f"Active alerts: {len(alerts)}")
    for alert in alerts[-3:]:  # Show last 3 alerts
        print(f"  - {alert['severity'].upper()}: {alert['title']}")


if __name__ == "__main__":
    asyncio.run(demo_error_handling())
