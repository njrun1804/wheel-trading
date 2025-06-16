#!/usr/bin/env python3
"""
Memory Stress Testing Script for Overflow Prevention

Comprehensive stress testing to validate memory management, overflow prevention,
and system stability under extreme load conditions.
"""

import asyncio
import gc
import json
import logging
import multiprocessing
import os
import psutil
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.memory.unified_memory_manager import (
    AllocationPriority,
    PressureLevel,
    get_memory_manager,
)
from src.unity_wheel.monitoring.unified_monitor import (
    MonitoredOperation,
    get_unified_monitor,
)
from bolt.core.output_token_manager import (
    OutputTokenManager,
    ResponseStrategy,
    get_output_token_manager,
)
from tests.utils.data_generators import (
    generate_large_dataset,
    generate_memory_stress_data,
    generate_streaming_data,
    StreamingDataGenerator,
    create_test_database_content,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    
    # Test duration and intensity
    duration_minutes: int = 30
    max_memory_gb: float = 16.0  # Maximum memory to use (leave 8GB for system)
    concurrent_workers: int = 12  # M4 Pro has 12 cores
    
    # Memory stress parameters
    large_allocation_mb: int = 1000
    streaming_data_rate_mb_per_second: float = 10.0
    token_stress_size_mb: int = 100
    
    # Test scenarios
    test_memory_exhaustion: bool = True
    test_concurrent_pressure: bool = True
    test_streaming_overflow: bool = True
    test_token_limits: bool = True
    test_recovery_mechanisms: bool = True
    
    # Performance thresholds
    max_response_time_ms: float = 5000
    min_success_rate: float = 0.90
    max_memory_pressure_events: int = 100


@dataclass
class StressTestResults:
    """Results from stress testing"""
    
    # Test execution
    start_time: float
    end_time: float
    duration_seconds: float
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Memory metrics
    peak_memory_usage_gb: float = 0.0
    memory_pressure_events: int = 0
    memory_allocations: int = 0
    memory_deallocations: int = 0
    memory_allocation_failures: int = 0
    
    # Overflow prevention
    overflow_incidents: int = 0
    token_optimizations: int = 0
    streaming_buffer_overflows: int = 0
    recovery_events: int = 0
    
    # System health
    cpu_peak_percent: float = 0.0
    cpu_average_percent: float = 0.0
    system_errors: List[str] = None
    
    def __post_init__(self):
        if self.system_errors is None:
            self.system_errors = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second"""
        if self.duration_seconds == 0:
            return 0.0
        return self.total_operations / self.duration_seconds


class MemoryStressTester:
    """Comprehensive memory stress tester"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results = StressTestResults(
            start_time=time.time(),
            end_time=0,
            duration_seconds=0,
        )
        
        # System components
        self.memory_manager = get_memory_manager()
        self.monitor = get_unified_monitor()
        self.token_manager = get_output_token_manager()
        
        # Test state
        self.running = True
        self.workers = []
        self.streaming_generators = []
        self.temp_files = []
        
        # Performance tracking
        self.response_times = []
        self.memory_samples = []
        self.cpu_samples = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register monitoring callbacks
        self._setup_monitoring()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def _setup_monitoring(self):
        """Setup monitoring callbacks"""
        def memory_pressure_callback(level: PressureLevel):
            self.results.memory_pressure_events += 1
            logger.warning(f"Memory pressure event: {level.name}")
            
        def overflow_callback(incident_type: str, details: Dict[str, Any]):
            self.results.overflow_incidents += 1
            logger.error(f"Overflow incident: {incident_type} - {details}")
            
        # Register callbacks
        self.memory_manager.register_pressure_callback(
            PressureLevel.MEDIUM, memory_pressure_callback
        )
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.monitor.capture_baseline()
        
    async def run_stress_test(self) -> StressTestResults:
        """Run comprehensive stress test"""
        logger.info("🚀 Starting Memory Stress Test")
        logger.info(f"Configuration: {asdict(self.config)}")
        
        try:
            # Start background monitoring
            monitor_task = asyncio.create_task(self._background_monitoring())
            
            # Create stress test tasks
            tasks = []
            
            if self.config.test_memory_exhaustion:
                tasks.append(self._test_memory_exhaustion())
                
            if self.config.test_concurrent_pressure:
                tasks.append(self._test_concurrent_pressure())
                
            if self.config.test_streaming_overflow:
                tasks.append(self._test_streaming_overflow())
                
            if self.config.test_token_limits:
                tasks.append(self._test_token_limits())
                
            if self.config.test_recovery_mechanisms:
                tasks.append(self._test_recovery_mechanisms())
                
            # Run all stress tests concurrently
            logger.info(f"Running {len(tasks)} concurrent stress test scenarios...")
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Stop monitoring
            monitor_task.cancel()
            
        except Exception as e:
            logger.error(f"Stress test error: {e}")
            self.results.system_errors.append(str(e))
            
        finally:
            await self._cleanup()
            
        # Finalize results
        self.results.end_time = time.time()
        self.results.duration_seconds = self.results.end_time - self.results.start_time
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        logger.info("✅ Memory Stress Test Completed")
        return self.results
    
    async def _test_memory_exhaustion(self):
        """Test memory exhaustion scenarios"""
        logger.info("🧪 Testing memory exhaustion scenarios")
        
        allocations = []
        
        try:
            # Progressively allocate larger chunks until we hit limits
            for size_mb in [100, 500, 1000, 2000, 3000, 5000]:
                if not self.running:
                    break
                    
                start_time = time.time()
                
                try:
                    alloc_id = self.memory_manager.allocate(
                        "stress_test",
                        size_mb,
                        f"exhaustion_test_{size_mb}MB",
                        priority=AllocationPriority.STANDARD
                    )
                    
                    if alloc_id:
                        allocations.append(alloc_id)
                        self.results.memory_allocations += 1
                        self.results.successful_operations += 1
                        
                        # Simulate work
                        await asyncio.sleep(0.1)
                        
                    else:
                        self.results.memory_allocation_failures += 1
                        self.results.failed_operations += 1
                        
                except MemoryError:
                    self.results.memory_allocation_failures += 1
                    self.results.failed_operations += 1
                    
                self.results.total_operations += 1
                self.response_times.append((time.time() - start_time) * 1000)
                
        finally:
            # Cleanup allocations
            for alloc_id in allocations:
                self.memory_manager.deallocate(alloc_id)
                self.results.memory_deallocations += 1
    
    async def _test_concurrent_pressure(self):
        """Test concurrent memory pressure scenarios"""
        logger.info("🧪 Testing concurrent memory pressure")
        
        async def worker_task(worker_id: int):
            """Individual worker task"""
            worker_allocations = []
            
            try:
                for i in range(20):  # 20 operations per worker
                    if not self.running:
                        break
                        
                    start_time = time.time()
                    
                    try:
                        # Mix of different operations
                        if i % 3 == 0:
                            # Memory allocation
                            alloc_id = self.memory_manager.allocate(
                                "stress_test",
                                random.randint(10, 200),
                                f"worker_{worker_id}_alloc_{i}",
                                priority=random.choice(list(AllocationPriority))
                            )
                            if alloc_id:
                                worker_allocations.append(alloc_id)
                                self.results.memory_allocations += 1
                                
                        elif i % 3 == 1:
                            # Generate stress data
                            stress_data = generate_memory_stress_data(
                                random.randint(1, 10)
                            )
                            
                            # Apply token optimization
                            optimized = self.token_manager.optimize_response(
                                stress_data, ResponseStrategy.PRIORITIZE
                            )
                            self.results.token_optimizations += 1
                            
                        else:
                            # Streaming simulation
                            stream_data = generate_streaming_data(
                                random.randint(100, 1000)
                            )
                            # Process streaming data
                            await self._process_streaming_batch(stream_data)
                            
                        self.results.successful_operations += 1
                        
                    except Exception as e:
                        self.results.failed_operations += 1
                        logger.debug(f"Worker {worker_id} operation {i} failed: {e}")
                        
                    self.results.total_operations += 1
                    self.response_times.append((time.time() - start_time) * 1000)
                    
            finally:
                # Cleanup worker allocations
                for alloc_id in worker_allocations:
                    self.memory_manager.deallocate(alloc_id)
                    self.results.memory_deallocations += 1
        
        # Launch concurrent workers
        tasks = []
        for worker_id in range(self.config.concurrent_workers):
            tasks.append(worker_task(worker_id))
            
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _test_streaming_overflow(self):
        """Test streaming data overflow scenarios"""
        logger.info("🧪 Testing streaming overflow scenarios")
        
        # Create high-volume streaming generator
        generator = StreamingDataGenerator()
        
        try:
            buffer = []
            max_buffer_size = 10_000_000  # 10MB buffer
            current_buffer_size = 0
            
            # Generate streaming data for test duration
            stream = generator.generate_stream(
                duration_seconds=min(60, self.config.duration_minutes * 60 // 4)
            )
            
            for chunk in stream:
                if not self.running:
                    break
                    
                start_time = time.time()
                
                try:
                    chunk_size = len(chunk.encode('utf-8'))
                    
                    # Check buffer overflow
                    if current_buffer_size + chunk_size > max_buffer_size:
                        # Apply overflow prevention
                        if len(buffer) > 1000:
                            # Keep first 500 and last 500 items
                            buffer = buffer[:500] + buffer[-500:]
                            current_buffer_size = sum(
                                len(item.encode('utf-8')) for item in buffer
                            )
                            self.results.streaming_buffer_overflows += 1
                    
                    if current_buffer_size + chunk_size <= max_buffer_size:
                        buffer.append(chunk)
                        current_buffer_size += chunk_size
                        self.results.successful_operations += 1
                    else:
                        # Skip chunk to prevent overflow
                        self.results.streaming_buffer_overflows += 1
                        
                except Exception as e:
                    self.results.failed_operations += 1
                    logger.debug(f"Streaming processing error: {e}")
                    
                self.results.total_operations += 1
                self.response_times.append((time.time() - start_time) * 1000)
                
        finally:
            generator.stop()
    
    async def _test_token_limits(self):
        """Test token limit handling"""
        logger.info("🧪 Testing token limit scenarios")
        
        # Test different response sizes
        test_sizes = [1, 10, 50, 100, 200, 500]  # MB
        
        for size_mb in test_sizes:
            if not self.running:
                break
                
            for strategy in ResponseStrategy:
                try:
                    start_time = time.time()
                    
                    # Generate large response
                    large_data = generate_memory_stress_data(size_mb)
                    
                    # Apply token optimization
                    optimized = self.token_manager.optimize_response(
                        large_data, strategy
                    )
                    
                    # Validate response
                    is_valid, token_count, status = self.token_manager.validate_response_size(
                        optimized
                    )
                    
                    if is_valid:
                        self.results.successful_operations += 1
                        self.results.token_optimizations += 1
                    else:
                        self.results.failed_operations += 1
                        
                except Exception as e:
                    self.results.failed_operations += 1
                    logger.debug(f"Token optimization error: {e}")
                    
                self.results.total_operations += 1
                self.response_times.append((time.time() - start_time) * 1000)
    
    async def _test_recovery_mechanisms(self):
        """Test system recovery mechanisms"""
        logger.info("🧪 Testing recovery mechanisms")
        
        # Test recovery from various pressure levels
        pressure_levels = [
            PressureLevel.MEDIUM,
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        ]
        
        for pressure_level in pressure_levels:
            if not self.running:
                break
                
            try:
                start_time = time.time()
                
                # Trigger pressure event
                self.memory_manager.handle_pressure(pressure_level)
                
                # Wait for recovery
                await asyncio.sleep(1.0)
                
                # Test allocation after recovery
                alloc_id = self.memory_manager.allocate(
                    "stress_test",
                    50,
                    f"recovery_test_{pressure_level.name}",
                    priority=AllocationPriority.IMPORTANT
                )
                
                if alloc_id:
                    self.memory_manager.deallocate(alloc_id)
                    self.results.successful_operations += 1
                    self.results.recovery_events += 1
                else:
                    self.results.failed_operations += 1
                    
            except Exception as e:
                self.results.failed_operations += 1
                logger.debug(f"Recovery test error: {e}")
                
            self.results.total_operations += 1
            self.response_times.append((time.time() - start_time) * 1000)
    
    async def _process_streaming_batch(self, stream_data: List[str]):
        """Process a batch of streaming data"""
        # Simulate processing work
        total_size = sum(len(chunk) for chunk in stream_data)
        
        # Apply token management if data is too large
        if total_size > 1_000_000:  # 1MB
            optimized = self.token_manager.optimize_response(
                {"stream_data": stream_data}, ResponseStrategy.TRUNCATE
            )
            self.results.token_optimizations += 1
    
    async def _background_monitoring(self):
        """Background system monitoring"""
        while self.running:
            try:
                # Sample system metrics
                if psutil:
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent(interval=1.0)
                    
                    # Track peak values
                    memory_gb = memory.used / (1024**3)
                    self.results.peak_memory_usage_gb = max(
                        self.results.peak_memory_usage_gb, memory_gb
                    )
                    self.results.cpu_peak_percent = max(
                        self.results.cpu_peak_percent, cpu_percent
                    )
                    
                    # Store samples
                    self.memory_samples.append(memory_gb)
                    self.cpu_samples.append(cpu_percent)
                    
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        if self.response_times:
            self.results.average_response_time_ms = sum(self.response_times) / len(self.response_times)
            
            sorted_times = sorted(self.response_times)
            self.results.p95_response_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
            self.results.p99_response_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
            
        if self.cpu_samples:
            self.results.cpu_average_percent = sum(self.cpu_samples) / len(self.cpu_samples)
    
    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up stress test resources")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Stop streaming generators
        for generator in self.streaming_generators:
            generator.stop()
            
        # Cleanup temp files
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
                
        # Force garbage collection
        gc.collect()


class StressTestReporter:
    """Generate comprehensive stress test reports"""
    
    def __init__(self, results: StressTestResults, config: StressTestConfig):
        self.results = results
        self.config = config
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            "test_configuration": asdict(self.config),
            "test_results": asdict(self.results),
            "performance_analysis": self._analyze_performance(),
            "memory_analysis": self._analyze_memory(),
            "overflow_analysis": self._analyze_overflow_prevention(),
            "recommendations": self._generate_recommendations(),
            "pass_fail_status": self._determine_pass_fail(),
        }
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        return {
            "success_rate": self.results.success_rate,
            "operations_per_second": self.results.operations_per_second,
            "response_time_analysis": {
                "average_ms": self.results.average_response_time_ms,
                "p95_ms": self.results.p95_response_time_ms,
                "p99_ms": self.results.p99_response_time_ms,
                "meets_sla": self.results.p95_response_time_ms < self.config.max_response_time_ms,
            },
            "throughput_analysis": {
                "total_operations": self.results.total_operations,
                "duration_minutes": self.results.duration_seconds / 60,
                "sustained_load": self.results.total_operations > 1000,
            }
        }
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        return {
            "peak_usage_gb": self.results.peak_memory_usage_gb,
            "within_limits": self.results.peak_memory_usage_gb < self.config.max_memory_gb,
            "allocation_success_rate": (
                self.results.memory_allocations / 
                max(1, self.results.memory_allocations + self.results.memory_allocation_failures)
            ),
            "pressure_management": {
                "pressure_events": self.results.memory_pressure_events,
                "recovery_events": self.results.recovery_events,
                "pressure_handled": self.results.memory_pressure_events <= self.config.max_memory_pressure_events,
            },
            "memory_efficiency": {
                "allocations": self.results.memory_allocations,
                "deallocations": self.results.memory_deallocations,
                "leak_indicators": abs(self.results.memory_allocations - self.results.memory_deallocations),
            }
        }
    
    def _analyze_overflow_prevention(self) -> Dict[str, Any]:
        """Analyze overflow prevention effectiveness"""
        return {
            "overflow_incidents": self.results.overflow_incidents,
            "prevention_effectiveness": {
                "token_optimizations": self.results.token_optimizations,
                "streaming_overflows_prevented": self.results.streaming_buffer_overflows,
                "zero_critical_failures": self.results.overflow_incidents == 0,
            },
            "system_stability": {
                "system_errors": len(self.results.system_errors),
                "stable_operation": len(self.results.system_errors) == 0,
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        if self.results.success_rate < self.config.min_success_rate:
            recommendations.append(
                f"Success rate ({self.results.success_rate:.1%}) below target "
                f"({self.config.min_success_rate:.1%}). Consider optimizing error handling."
            )
            
        if self.results.p95_response_time_ms > self.config.max_response_time_ms:
            recommendations.append(
                f"P95 response time ({self.results.p95_response_time_ms:.1f}ms) exceeds "
                f"target ({self.config.max_response_time_ms:.1f}ms). Consider performance tuning."
            )
        
        # Memory recommendations
        if self.results.peak_memory_usage_gb > self.config.max_memory_gb * 0.9:
            recommendations.append(
                f"Peak memory usage ({self.results.peak_memory_usage_gb:.1f}GB) near limit. "
                "Consider memory optimization."
            )
            
        if self.results.memory_pressure_events > self.config.max_memory_pressure_events:
            recommendations.append(
                f"High number of memory pressure events ({self.results.memory_pressure_events}). "
                "Consider increasing memory limits or improving allocation strategies."
            )
        
        # Overflow prevention recommendations
        if self.results.overflow_incidents > 0:
            recommendations.append(
                f"Overflow incidents detected ({self.results.overflow_incidents}). "
                "Review overflow prevention mechanisms."
            )
        
        if not recommendations:
            recommendations.append("All tests passed within acceptable parameters. System performing well.")
            
        return recommendations
    
    def _determine_pass_fail(self) -> Dict[str, Any]:
        """Determine overall pass/fail status"""
        criteria = {
            "success_rate_pass": self.results.success_rate >= self.config.min_success_rate,
            "response_time_pass": self.results.p95_response_time_ms <= self.config.max_response_time_ms,
            "memory_usage_pass": self.results.peak_memory_usage_gb <= self.config.max_memory_gb,
            "overflow_prevention_pass": self.results.overflow_incidents == 0,
            "system_stability_pass": len(self.results.system_errors) == 0,
        }
        
        overall_pass = all(criteria.values())
        
        return {
            "overall_pass": overall_pass,
            "criteria": criteria,
            "passed_criteria": sum(criteria.values()),
            "total_criteria": len(criteria),
            "pass_percentage": sum(criteria.values()) / len(criteria) * 100,
        }
    
    def print_summary(self):
        """Print test summary to console"""
        print("\n" + "="*80)
        print("🧪 MEMORY STRESS TEST SUMMARY")
        print("="*80)
        
        print(f"\n📊 Test Execution:")
        print(f"   Duration: {self.results.duration_seconds/60:.1f} minutes")
        print(f"   Total Operations: {self.results.total_operations:,}")
        print(f"   Success Rate: {self.results.success_rate:.1%}")
        print(f"   Operations/Second: {self.results.operations_per_second:.1f}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Response Time: {self.results.average_response_time_ms:.1f}ms")
        print(f"   P95 Response Time: {self.results.p95_response_time_ms:.1f}ms")
        print(f"   P99 Response Time: {self.results.p99_response_time_ms:.1f}ms")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Peak Memory Usage: {self.results.peak_memory_usage_gb:.2f}GB")
        print(f"   Memory Allocations: {self.results.memory_allocations:,}")
        print(f"   Memory Pressure Events: {self.results.memory_pressure_events}")
        
        print(f"\n🛡️ Overflow Prevention:")
        print(f"   Overflow Incidents: {self.results.overflow_incidents}")
        print(f"   Token Optimizations: {self.results.token_optimizations:,}")
        print(f"   Buffer Overflows Prevented: {self.results.streaming_buffer_overflows}")
        
        # Pass/fail status
        pass_fail = self._determine_pass_fail()
        status = "✅ PASSED" if pass_fail["overall_pass"] else "❌ FAILED"
        print(f"\n🎯 Overall Result: {status}")
        print(f"   Criteria Passed: {pass_fail['passed_criteria']}/{pass_fail['total_criteria']}")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main stress testing function"""
    
    # Configuration
    config = StressTestConfig(
        duration_minutes=10,  # Reduced for demo
        max_memory_gb=16.0,
        concurrent_workers=8,  # Conservative for testing
        test_memory_exhaustion=True,
        test_concurrent_pressure=True,
        test_streaming_overflow=True,
        test_token_limits=True,
        test_recovery_mechanisms=True,
    )
    
    # Run stress test
    tester = MemoryStressTester(config)
    
    try:
        print("🚀 Starting comprehensive memory stress test...")
        results = await tester.run_stress_test()
        
        # Generate report
        reporter = StressTestReporter(results, config)
        report = reporter.generate_report()
        
        # Print summary
        reporter.print_summary()
        
        # Save detailed report
        report_file = f"stress_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report
        
    except KeyboardInterrupt:
        print("\n⚠️ Stress test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Stress test failed: {e}")
        return None


if __name__ == "__main__":
    import random
    
    # Ensure reproducible results for testing
    random.seed(42)
    
    # Run the stress test
    report = asyncio.run(main())
    
    if report:
        # Exit with appropriate code
        overall_pass = report.get("pass_fail_status", {}).get("overall_pass", False)
        sys.exit(0 if overall_pass else 1)
    else:
        sys.exit(1)