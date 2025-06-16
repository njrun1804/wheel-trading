# Comprehensive Resource Cleanup Strategy

## Overview

After analyzing the codebase, I've identified several critical resource management issues that can lead to file descriptor exhaustion and system instability. This document provides a comprehensive strategy to fix resource leaks, improve cleanup procedures, and prevent future issues.

## Executive Summary

### Key Findings

1. **Database Connection Pooling Issues**: DuckDB connections not properly managed in turbo implementations
2. **File Handle Leaks**: Stream processors and file operations lacking proper cleanup
3. **Subprocess Resource Management**: MCP servers and subprocess wrappers need better lifecycle management
4. **Async Resource Cleanup**: Mixed async/sync patterns causing resource retention
5. **Missing Context Managers**: Many resources created without proper cleanup guarantees

### Solution Components

1. **Resource Manager**: Centralized tracking and cleanup system
2. **Resource Monitor**: Real-time monitoring with intelligent alerting
3. **Best Practices Enforcement**: Runtime validation and guidance
4. **Integration Points**: Seamless integration with existing codebase

## Critical Resource Leak Patterns Identified

### 1. Database Connection Issues

**Location**: `/src/unity_wheel/accelerated_tools/duckdb_turbo.py`

**Problem**: 
- Connection pools not properly cleaned up
- No connection timeout handling
- Singleton pattern prevents proper cleanup

**Evidence**:
```python
# Lines 99-112: Context manager doesn't ensure cleanup
@asynccontextmanager
async def get_connection(self):
    conn = self.connections[hash(asyncio.current_task()) % len(self.connections)]
    try:
        yield conn
    finally:
        pass  # Connection stays in pool - NO CLEANUP!
```

**Fix Priority**: CRITICAL

### 2. File Handle Management

**Location**: `/src/unity_wheel/storage/optimized_storage.py`

**Problem**:
- Parquet files opened without context managers
- Cache files not cleaned up on errors
- No file handle limits enforced

**Evidence**:
```python
# Lines 57-58: Direct file access without cleanup
return pq.read_table(cache_file)

# Lines 84-85: Write without proper error handling
pq.write_table(table, cache_file)
```

**Fix Priority**: HIGH

### 3. Subprocess Resource Leaks

**Location**: `/bolt/macos_subprocess_wrapper.py`

**Problem**:
- Process tracking without guaranteed cleanup
- Thread pool shutdown not always called
- Memory usage in active processes dict

**Evidence**:
```python
# Lines 90-92: Active processes tracking without cleanup guarantees
self.active_processes: dict[int, subprocess.Popen] = {}
self.process_lock = threading.Lock()
```

**Fix Priority**: HIGH

### 4. Stream Processing Resource Issues

**Location**: `/src/unity_wheel/utils/stream_processors.py`

**Problem**:
- Temporary files not cleaned up on errors
- Memory monitoring but no enforcement
- Async generators without proper cleanup

**Evidence**:
```python
# Lines 197-232: Cleanup only in __aexit__, not on errors
async def cleanup(self) -> None:
    # Only called if __aexit__ is reached
```

**Fix Priority**: MEDIUM

## Implementation Plan

### Phase 1: Immediate Fixes (Week 1)

#### 1.1 Deploy Resource Management System

**Files to Update**:
- Add `/src/unity_wheel/utils/resource_manager.py` ✅ (Created)
- Add `/src/unity_wheel/utils/resource_monitor.py` ✅ (Created)
- Add `/src/unity_wheel/utils/resource_best_practices.py` ✅ (Created)

**Integration Points**:
```python
# Add to main application initialization
from unity_wheel.utils.resource_manager import init_resource_management
from unity_wheel.utils.resource_monitor import start_monitoring

# In startup.py or main entry point
init_resource_management()
await start_monitoring(interval=30.0)
```

#### 1.2 Fix Critical Database Issues

**Update `/src/unity_wheel/accelerated_tools/duckdb_turbo.py`**:

```python
# Replace lines 99-112 with:
@asynccontextmanager
async def get_connection(self):
    """Get a connection from the pool with proper cleanup."""
    if not self.connections:
        raise RuntimeError("No DuckDB connections available in pool")
    
    conn = self.connections[hash(asyncio.current_task()) % len(self.connections)]
    try:
        yield conn
    except Exception as e:
        # Log error and potentially recreate connection
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        # Ensure connection is still valid
        try:
            conn.execute("SELECT 1")
        except Exception:
            # Recreate connection if corrupted
            self._recreate_connection(conn)

# Add cleanup method:
def cleanup(self):
    """Close all connections and cleanup."""
    for conn in self.connections:
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    self.connections.clear()
    self._executor.shutdown(wait=True)
    
    # Unregister from resource tracker
    from unity_wheel.utils.resource_manager import get_resource_tracker
    tracker = get_resource_tracker()
    tracker.unregister_resource(self, 'database_pool')
```

#### 1.3 Fix File Handle Issues

**Update `/src/unity_wheel/storage/optimized_storage.py`**:

```python
# Replace file operations with context managers
from unity_wheel.utils.resource_manager import get_file_manager

async def get_options_arrow(self, symbol: str, lookback_hours: int = 24) -> pa.Table:
    # ... existing code ...
    
    if cache_file.exists() and cache_age < timedelta(minutes=config.performance.cache_ttl_minutes):
        # Use file manager for proper cleanup
        file_manager = get_file_manager()
        with file_manager.open_file(cache_file, 'rb') as f:
            return pq.read_table(f)
    
    # ... rest of method with proper file handling
```

### Phase 2: Enhanced Monitoring (Week 2)

#### 2.1 Real-time Resource Monitoring

**Integration Script** (`scripts/start_resource_monitoring.py`):

```python
#!/usr/bin/env python3
"""Start comprehensive resource monitoring."""

import asyncio
import logging
from pathlib import Path

from unity_wheel.utils.resource_monitor import (
    get_resource_monitor, 
    start_monitoring,
    add_email_alerts
)

async def main():
    # Setup monitoring with custom thresholds
    monitor = get_resource_monitor()
    
    # Configure email alerts (optional)
    # add_email_alerts(
    #     smtp_server="smtp.gmail.com",
    #     smtp_port=587,
    #     username="alerts@example.com",
    #     password="app_password",
    #     from_email="alerts@example.com",
    #     to_emails=["admin@example.com"]
    # )
    
    # Start monitoring every 30 seconds
    await start_monitoring(interval=30.0)
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
            # Generate report every minute
            report = monitor.generate_report()
            
            # Log high-level stats
            metrics = report['current_metrics']
            logging.info(
                f"Resources: {metrics['open_files']} files, "
                f"{metrics['memory_mb']:.1f}MB, "
                f"{metrics['processes']} processes"
            )
            
    except KeyboardInterrupt:
        logging.info("Shutting down resource monitoring...")
        await monitor.stop_monitoring()
        monitor.save_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

#### 2.2 Automated Cleanup Triggers

**Add to main application** (`src/unity_wheel/cli/run.py`):

```python
# Add resource monitoring to main entry points
from unity_wheel.utils.resource_manager import get_resource_tracker
from unity_wheel.utils.resource_monitor import get_resource_monitor

async def main():
    # Initialize resource management
    tracker = get_resource_tracker()
    monitor = get_resource_monitor()
    
    # Setup emergency cleanup triggers
    def emergency_cleanup(metrics):
        logging.error("Emergency cleanup triggered!")
        tracker.cleanup_all()
    
    monitor.add_alert_handler(emergency_cleanup)
    
    # Start monitoring
    await monitor.start_monitoring(interval=30.0)
    
    try:
        # Your existing main logic here
        await run_trading_system()
    finally:
        # Cleanup on exit
        await monitor.stop_monitoring()
        tracker.cleanup_all()
```

### Phase 3: Prevention & Best Practices (Week 3)

#### 3.1 Code Validation Integration

**Pre-commit Hook** (`.githooks/pre-commit`):

```bash
#!/bin/bash
# Add resource validation to pre-commit

python -c "
from unity_wheel.utils.resource_best_practices import validate_codebase
import sys

if not validate_codebase('.'):
    print('❌ Resource management validation failed!')
    print('Run: python -m unity_wheel.utils.resource_best_practices --fix')
    sys.exit(1)
else:
    print('✅ Resource management validation passed')
"
```

#### 3.2 Runtime Enforcement

**Add to development environment**:

```python
# In development/debug mode
from unity_wheel.utils.resource_best_practices import enable_runtime_enforcement

if DEBUG:
    enable_runtime_enforcement(strict=True)  # Raise exceptions
else:
    enable_runtime_enforcement(strict=False)  # Just warnings
```

### Phase 4: Monitoring & Maintenance (Ongoing)

#### 4.1 Regular Health Checks

**Daily Health Check Script** (`scripts/daily_resource_check.py`):

```python
#!/usr/bin/env python3
"""Daily resource health check."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from unity_wheel.utils.resource_monitor import get_resource_monitor

def main():
    monitor = get_resource_monitor()
    report = monitor.generate_report()
    
    # Check for issues
    issues = []
    metrics = report['current_metrics']
    
    # Check file descriptor usage
    fd_ratio = metrics['open_files'] / max(metrics['max_files'], 1)
    if fd_ratio > 0.8:
        issues.append(f"High file descriptor usage: {fd_ratio:.1%}")
    
    # Check memory usage
    if metrics['memory_mb'] > 1500:
        issues.append(f"High memory usage: {metrics['memory_mb']:.1f}MB")
    
    # Check active alerts
    if report['active_alerts']:
        issues.append(f"{len(report['active_alerts'])} active alerts")
    
    # Generate summary
    status = "❌ ISSUES FOUND" if issues else "✅ HEALTHY"
    print(f"Resource Health Check - {datetime.now()}: {status}")
    
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Save detailed report
    report_file = Path(f"health_check_{datetime.now().strftime('%Y%m%d')}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
```

#### 4.2 Automated Remediation

**Add to resource monitor callbacks**:

```python
# In resource_monitor.py - add to ResourceMonitor class
def _auto_remediate(self, alert: ResourceAlert):
    """Attempt automatic remediation for common issues."""
    
    if alert.resource_type == "file_descriptors" and alert.level == "critical":
        # Force garbage collection
        import gc
        gc.collect()
        
        # Cleanup known resource managers
        from .resource_manager import force_cleanup
        force_cleanup()
        
        logger.info("Attempted automatic file descriptor cleanup")
    
    elif alert.resource_type == "memory" and alert.level == "error":
        # Clear caches
        import gc
        gc.collect()
        
        # Trigger cleanup in major components
        from ..accelerated_tools.duckdb_turbo import get_duckdb_turbo
        db = get_duckdb_turbo()
        db.cleanup()
        
        logger.info("Attempted automatic memory cleanup")
```

## Integration Guide

### Quick Start (5 minutes)

1. **Add resource management to your main entry point**:

```python
# At the top of your main script
from unity_wheel.utils.resource_manager import init_resource_management, get_resource_tracker
from unity_wheel.utils.resource_monitor import start_monitoring

async def main():
    # Initialize resource management
    init_resource_management()
    tracker = get_resource_tracker()
    
    # Start monitoring
    await start_monitoring(interval=30.0)
    
    # Your existing code...
    
    # Cleanup on exit
    tracker.cleanup_all()
```

2. **Update database operations**:

```python
# Replace direct DuckDB usage with managed connections
from unity_wheel.utils.resource_manager import get_db_manager

db_manager = get_db_manager()
with db_manager.get_connection("trading_db", lambda: duckdb.connect("trading.db")):
    # Use connection here
    pass
```

3. **Update file operations**:

```python
# Replace direct file operations
from unity_wheel.utils.resource_manager import get_file_manager

file_manager = get_file_manager()
with file_manager.open_file("data.json", "r") as f:
    data = json.load(f)
```

### Advanced Integration

#### For DuckDB Turbo (Critical)

**File**: `/src/unity_wheel/accelerated_tools/duckdb_turbo.py`

**Add at the end of `__init__` method**:
```python
# Register with resource tracker
from unity_wheel.utils.resource_manager import get_resource_tracker
tracker = get_resource_tracker()
tracker.register_resource(self, 'database_pool', self.cleanup)
```

#### For Stream Processors

**File**: `/src/unity_wheel/utils/stream_processors.py`

**Add to `StreamProcessor.__init__`**:
```python
from .resource_manager import get_resource_tracker
tracker = get_resource_tracker()
tracker.register_resource(self, 'stream_processor', self.cleanup)
```

#### For MCP Servers

**File**: `/src/unity_wheel/mcp/base_server.py`

**Add to `HealthCheckMCP.__init__`**:
```python
from ..utils.resource_manager import get_resource_tracker
tracker = get_resource_tracker()
tracker.register_resource(self, 'mcp_server', self.shutdown)
```

## Monitoring Dashboards

### Command Line Monitoring

```bash
# Check current resource usage
python -c "
from unity_wheel.utils.resource_manager import log_resource_usage
log_resource_usage('Current Status')
"

# Generate detailed report
python -c "
from unity_wheel.utils.resource_monitor import save_report
report_path = save_report()
print(f'Report saved to: {report_path}')
"

# Validate codebase
python -c "
from unity_wheel.utils.resource_best_practices import validate_codebase
validate_codebase('.')
"
```

### Web Dashboard (Optional)

**Create** `scripts/resource_dashboard.py`:

```python
#!/usr/bin/env python3
"""Simple web dashboard for resource monitoring."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from unity_wheel.utils.resource_monitor import get_resource_monitor

class ResourceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            monitor = get_resource_monitor()
            report = monitor.generate_report()
            
            html = f"""
            <html>
            <head><title>Resource Monitor</title></head>
            <body>
                <h1>Resource Monitoring Dashboard</h1>
                <h2>Current Metrics</h2>
                <ul>
                    <li>Open Files: {report['current_metrics']['open_files']}</li>
                    <li>Memory: {report['current_metrics']['memory_mb']:.1f}MB</li>
                    <li>Processes: {report['current_metrics']['processes']}</li>
                    <li>Threads: {report['current_metrics']['threads']}</li>
                </ul>
                <h2>Active Alerts</h2>
                <ul>
                    {"".join(f"<li>{alert['message']}</li>" for alert in report['active_alerts'])}
                </ul>
                <script>setTimeout(() => location.reload(), 10000);</script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            monitor = get_resource_monitor()
            report = monitor.generate_report()
            self.wfile.write(json.dumps(report).encode())

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8080), ResourceHandler)
    print("Resource dashboard running at http://localhost:8080")
    server.serve_forever()
```

## Testing Strategy

### Unit Tests

**Create** `tests/test_resource_management.py`:

```python
import pytest
import asyncio
from unity_wheel.utils.resource_manager import ResourceTracker, get_resource_tracker
from unity_wheel.utils.resource_monitor import ResourceMonitor

class TestResourceManager:
    def test_resource_tracking(self):
        tracker = ResourceTracker()
        
        # Test resource registration
        test_resource = "test_file_handle"
        tracker.register_resource(test_resource, "file")
        
        assert tracker._resource_counts["file"] == 1
        
        # Test cleanup
        tracker.unregister_resource(test_resource, "file")
        assert tracker._resource_counts["file"] == 0
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self):
        monitor = ResourceMonitor()
        
        # Test metrics collection
        metrics = monitor.check_resources()
        assert metrics.open_files >= 0
        assert metrics.memory_mb > 0
        
        # Test monitoring start/stop
        await monitor.start_monitoring(interval=0.1)
        await asyncio.sleep(0.2)
        await monitor.stop_monitoring()

class TestResourceLeakDetection:
    def test_file_handle_leak_detection(self):
        """Test detection of file handle leaks."""
        # This would intentionally create a leak for testing
        pass
    
    def test_database_connection_leak_detection(self):
        """Test detection of database connection leaks."""
        pass
```

### Integration Tests

**Create** `tests/test_integration_resource_cleanup.py`:

```python
import pytest
import asyncio
from unity_wheel.utils.resource_manager import get_resource_tracker
from unity_wheel.accelerated_tools.duckdb_turbo import get_duckdb_turbo

class TestIntegrationResourceCleanup:
    @pytest.mark.asyncio
    async def test_duckdb_cleanup_integration(self):
        """Test that DuckDB connections are properly cleaned up."""
        tracker = get_resource_tracker()
        initial_connections = tracker._resource_counts.get('database', 0)
        
        # Create DuckDB instance
        db = get_duckdb_turbo()
        
        # Use the database
        await db.execute("SELECT 1")
        
        # Cleanup
        db.cleanup()
        
        # Verify cleanup
        final_connections = tracker._resource_counts.get('database', 0)
        assert final_connections <= initial_connections
    
    def test_file_operations_cleanup(self):
        """Test file operations are properly cleaned up."""
        from unity_wheel.utils.resource_manager import get_file_manager
        
        file_manager = get_file_manager()
        
        # Use file manager
        with file_manager.open_file("test_file.txt", "w") as f:
            f.write("test")
        
        # Verify no open files remain
        assert len(file_manager._open_files) == 0
```

## Success Metrics

### Key Performance Indicators

1. **File Descriptor Usage**: < 50% of system limit
2. **Memory Growth Rate**: < 10MB/hour during steady state
3. **Connection Pool Size**: Stable (not growing unbounded)
4. **Resource Leak Incidents**: Zero per day
5. **Cleanup Success Rate**: > 99%

### Monitoring Alerts

- **WARNING**: FD usage > 70%, Memory > 1.5GB
- **ERROR**: FD usage > 85%, Memory > 2GB  
- **CRITICAL**: FD usage > 95%, Memory > 2.5GB

### Reporting

- **Daily**: Automated health check reports
- **Weekly**: Resource usage trend analysis
- **Monthly**: Resource optimization recommendations

## Rollback Plan

If issues arise during implementation:

1. **Immediate Rollback**: Remove resource tracking imports
2. **Partial Rollback**: Disable enforcement but keep monitoring
3. **Configuration Rollback**: Adjust thresholds via config files

**Rollback Script** (`scripts/rollback_resource_management.py`):

```python
#!/usr/bin/env python3
"""Rollback resource management features."""

import logging
from unity_wheel.utils.resource_manager import get_resource_tracker
from unity_wheel.utils.resource_monitor import get_resource_monitor

def rollback():
    """Safely rollback resource management."""
    try:
        # Stop monitoring
        monitor = get_resource_monitor()
        if monitor.monitoring_active:
            asyncio.run(monitor.stop_monitoring())
        
        # Cleanup tracker
        tracker = get_resource_tracker()
        tracker.cleanup_all()
        
        logging.info("Resource management safely rolled back")
        return True
    except Exception as e:
        logging.error(f"Rollback failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    success = rollback()
    exit(0 if success else 1)
```

## Conclusion

This comprehensive resource cleanup strategy addresses the critical file descriptor exhaustion issues through:

1. **Immediate fixes** for database connection pooling and file handle management
2. **Proactive monitoring** with intelligent alerting and automatic remediation  
3. **Prevention tools** including code validation and runtime enforcement
4. **Long-term maintenance** with automated health checks and optimization

The implementation provides multiple layers of protection while being minimally invasive to existing code. The monitoring system enables early detection of issues before they become critical, and the automated cleanup prevents system-wide failures.

**Next Steps**:
1. Implement Phase 1 critical fixes immediately
2. Deploy monitoring in development environment  
3. Gradually roll out to production with careful monitoring
4. Establish ongoing maintenance procedures

This strategy will eliminate resource leaks and provide a robust foundation for scalable, reliable operation of the trading system.