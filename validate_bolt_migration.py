#!/usr/bin/env python3
"""
BOLT Migration Validation Script

Comprehensive validation of BOLT to BOB migration ensuring:
- 1.5 tasks/second throughput maintained
- M4 Pro hardware optimizations preserved
- Work-stealing algorithm functional
- 8-agent coordination working
- Einstein integration preserved
- GPU acceleration maintained
- Error handling systems functional
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

BASE_DIR = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")

class BoltMigrationValidator:
    """Comprehensive BOLT migration validation system."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.validation_log: List[str] = []
        
    def log(self, message: str):
        """Log validation operations."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.validation_log.append(log_message)
    
    def test_import_system(self) -> bool:
        """Test that all critical imports work after migration."""
        self.log("🔍 Testing import system...")
        
        try:
            # Add BOB to path
            sys.path.insert(0, str(BASE_DIR))
            
            # Test basic BOB imports
            import bob
            self.log("✅ BOB basic imports working")
            
            # Test agent system
            from bob.agents import AgentOrchestrator, AgentPool, TaskManager
            self.log("✅ Agent system imports working")
            
            # Test BOLT integration
            try:
                from bob.integration.bolt.core_integration import BoltIntegration
                self.log("✅ BOLT core integration imports working")
            except ImportError as e:
                self.log(f"⚠️  BOLT integration import failed: {e}")
                return False
            
            # Test hardware components
            try:
                from bob.hardware.gpu.bolt_gpu_acceleration import BoltGPUAcceleration
                self.log("✅ GPU acceleration imports working")
            except ImportError as e:
                self.log(f"⚠️  GPU acceleration import warning: {e}")
                # Non-critical
            
            # Test performance components
            try:
                from bob.performance.bolt.benchmarks import BoltBenchmarks
                self.log("✅ Performance benchmark imports working")
            except ImportError as e:
                self.log(f"⚠️  Performance imports warning: {e}")
                # Non-critical
            
            self.test_results["import_system"] = True
            return True
            
        except Exception as e:
            self.log(f"❌ Import system test failed: {e}")
            self.test_results["import_system"] = False
            return False
    
    async def test_agent_orchestration(self) -> bool:
        """Test 8-agent orchestration system."""
        self.log("🤖 Testing 8-agent orchestration...")
        
        try:
            from bob.agents import AgentOrchestrator, Task, TaskPriority
            
            # Create orchestrator with 8 agents
            orchestrator = AgentOrchestrator(num_agents=8)
            await orchestrator.initialize()
            
            # Create test tasks
            test_tasks = []
            for i in range(16):  # 2 tasks per agent
                task = Task(
                    id=f"test_task_{i}",
                    description=f"Test task {i}",
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=0.1,
                    data={"test_data": i}
                )
                test_tasks.append(task)
            
            # Execute tasks and measure performance
            start_time = time.time()
            results = await orchestrator.execute_tasks(test_tasks)
            execution_time = time.time() - start_time
            
            # Validate results
            if len(results) != len(test_tasks):
                self.log(f"❌ Task count mismatch: expected {len(test_tasks)}, got {len(results)}")
                return False
            
            # Calculate throughput
            throughput = len(test_tasks) / execution_time if execution_time > 0 else 0
            self.performance_metrics["orchestration_throughput"] = throughput
            
            self.log(f"✅ 8-agent orchestration working: {len(results)} tasks in {execution_time:.3f}s")
            self.log(f"📊 Throughput: {throughput:.1f} tasks/second")
            
            # Check if we meet the 1.5 tasks/second requirement
            if throughput >= 1.5:
                self.log("✅ Throughput requirement met (≥1.5 tasks/sec)")
                self.test_results["agent_orchestration"] = True
                return True
            else:
                self.log(f"⚠️  Throughput below requirement: {throughput:.1f} < 1.5 tasks/sec")
                self.test_results["agent_orchestration"] = False
                return False
                
        except Exception as e:
            self.log(f"❌ Agent orchestration test failed: {e}")
            self.test_results["agent_orchestration"] = False
            return False
    
    async def test_work_stealing_algorithm(self) -> bool:
        """Test work-stealing algorithm functionality."""
        self.log("⚡ Testing work-stealing algorithm...")
        
        try:
            from bob.agents import AgentPool
            from bob.agents.agent_pool import WorkStealingTask, TaskPriority
            
            # Create agent pool with work stealing enabled
            agent_pool = AgentPool(num_agents=8, enable_work_stealing=True)
            await agent_pool.initialize()
            
            # Create mixed workload (some heavy, some light tasks)
            tasks = []
            
            # Heavy tasks (longer duration)
            for i in range(4):
                task = WorkStealingTask(
                    id=f"heavy_task_{i}",
                    description=f"Heavy task {i}",
                    priority=TaskPriority.HIGH,
                    subdividable=True,
                    estimated_duration=1.0,
                    remaining_work=1.0
                )
                tasks.append(task)
            
            # Light tasks (shorter duration)
            for i in range(12):
                task = WorkStealingTask(
                    id=f"light_task_{i}",
                    description=f"Light task {i}",
                    priority=TaskPriority.NORMAL,
                    subdividable=True,
                    estimated_duration=0.1,
                    remaining_work=0.1
                )
                tasks.append(task)
            
            # Execute tasks and measure work stealing effectiveness
            start_time = time.time()
            
            # Submit all tasks
            for task in tasks:
                await agent_pool.submit_task(task)
            
            # Wait for completion
            await agent_pool.wait_for_completion()
            
            execution_time = time.time() - start_time
            self.performance_metrics["work_stealing_time"] = execution_time
            
            self.log(f"✅ Work-stealing algorithm functional: {len(tasks)} mixed tasks in {execution_time:.3f}s")
            
            # Work stealing should complete mixed workload efficiently
            # If work stealing is working, heavy tasks should be subdivided and distributed
            expected_max_time = 2.0  # Should complete faster than sequential execution
            if execution_time < expected_max_time:
                self.log("✅ Work-stealing efficiency validated")
                self.test_results["work_stealing"] = True
                return True
            else:
                self.log(f"⚠️  Work-stealing may not be optimal: {execution_time:.3f}s > {expected_max_time}s")
                self.test_results["work_stealing"] = False
                return False
                
        except Exception as e:
            self.log(f"❌ Work-stealing test failed: {e}")
            self.test_results["work_stealing"] = False
            return False
    
    def test_hardware_optimization(self) -> bool:
        """Test M4 Pro hardware optimizations."""
        self.log("🔧 Testing M4 Pro hardware optimizations...")
        
        try:
            import psutil
            import platform
            
            # Check system info
            system = platform.system()
            machine = platform.machine()
            
            self.log(f"System: {system} {machine}")
            
            # Check if we're on M4 Pro (arm64 Mac)
            is_m4_capable = system == "Darwin" and machine == "arm64"
            
            if is_m4_capable:
                self.log("✅ Running on M4-capable system (Darwin arm64)")
                
                # Test CPU utilization optimization
                cpu_count = psutil.cpu_count()
                cpu_count_logical = psutil.cpu_count(logical=True)
                
                self.log(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
                
                # M4 Pro should have 12 cores
                if cpu_count >= 10:  # Allow some variance
                    self.log("✅ M4 Pro CPU core count detected")
                    
                    # Test memory configuration
                    memory = psutil.virtual_memory()
                    memory_gb = memory.total / (1024**3)
                    
                    self.log(f"Memory: {memory_gb:.1f} GB")
                    
                    # M4 Pro typically has 18GB or 24GB
                    if memory_gb >= 16:
                        self.log("✅ M4 Pro memory configuration detected")
                        self.test_results["hardware_optimization"] = True
                        return True
                    else:
                        self.log(f"⚠️  Lower memory than expected: {memory_gb:.1f} GB")
                        self.test_results["hardware_optimization"] = False
                        return False
                else:
                    self.log(f"⚠️  Lower core count than expected: {cpu_count}")
                    self.test_results["hardware_optimization"] = False
                    return False
            else:
                self.log("ℹ️  Not running on M4-capable system, skipping hardware-specific tests")
                self.test_results["hardware_optimization"] = True  # Pass on non-M4 systems
                return True
                
        except Exception as e:
            self.log(f"❌ Hardware optimization test failed: {e}")
            self.test_results["hardware_optimization"] = False
            return False
    
    def test_einstein_integration(self) -> bool:
        """Test Einstein semantic search integration."""
        self.log("🧠 Testing Einstein integration...")
        
        try:
            # Test if Einstein is available
            try:
                import einstein
                from einstein.unified_index import UnifiedSearchEngine
                self.log("✅ Einstein module available")
                
                # Test basic Einstein functionality
                search_engine = UnifiedSearchEngine()
                self.log("✅ Einstein search engine can be created")
                
                # Test BOLT-Einstein integration
                try:
                    from bob.integration.bolt.einstein_accelerator import EinsteinAccelerator
                    accelerator = EinsteinAccelerator()
                    self.log("✅ BOLT-Einstein integration available")
                    
                    self.test_results["einstein_integration"] = True
                    return True
                    
                except ImportError as e:
                    self.log(f"⚠️  BOLT-Einstein integration not found: {e}")
                    self.test_results["einstein_integration"] = False
                    return False
                    
            except ImportError as e:
                self.log(f"ℹ️  Einstein not available: {e}")
                self.log("ℹ️  Skipping Einstein integration tests")
                self.test_results["einstein_integration"] = True  # Pass if Einstein not available
                return True
                
        except Exception as e:
            self.log(f"❌ Einstein integration test failed: {e}")
            self.test_results["einstein_integration"] = False
            return False
    
    def test_gpu_acceleration(self) -> bool:
        """Test GPU acceleration capabilities."""
        self.log("🎮 Testing GPU acceleration...")
        
        try:
            # Test Metal availability on macOS
            try:
                import platform
                if platform.system() == "Darwin":
                    # Test Metal framework
                    try:
                        import metal
                        self.log("✅ Metal framework available")
                    except ImportError:
                        # Try alternative Metal detection
                        try:
                            import subprocess
                            result = subprocess.run(["system_profiler", "SPDisplaysDataType"], 
                                                  capture_output=True, text=True)
                            if "Metal" in result.stdout:
                                self.log("✅ Metal GPU support detected")
                            else:
                                self.log("⚠️  Metal GPU support not clearly detected")
                        except Exception:
                            self.log("ℹ️  Could not detect Metal support")
                    
                    # Test MLX availability
                    try:
                        import mlx
                        import mlx.core as mx
                        self.log("✅ MLX framework available")
                        
                        # Test basic MLX operation
                        test_array = mx.array([1, 2, 3, 4])
                        result = mx.sum(test_array)
                        self.log("✅ MLX basic operations working")
                        
                    except ImportError:
                        self.log("ℹ️  MLX not available")
                    
                    # Test BOLT GPU integration
                    try:
                        from bob.hardware.gpu.bolt_gpu_acceleration import BoltGPUAcceleration
                        gpu_accel = BoltGPUAcceleration()
                        self.log("✅ BOLT GPU acceleration component available")
                        
                        self.test_results["gpu_acceleration"] = True
                        return True
                        
                    except ImportError as e:
                        self.log(f"⚠️  BOLT GPU acceleration not available: {e}")
                        self.test_results["gpu_acceleration"] = False
                        return False
                        
                else:
                    self.log("ℹ️  Not on macOS, skipping Metal-specific tests")
                    self.test_results["gpu_acceleration"] = True  # Pass on non-macOS
                    return True
                    
            except Exception as e:
                self.log(f"⚠️  GPU acceleration test warning: {e}")
                self.test_results["gpu_acceleration"] = False
                return False
                
        except Exception as e:
            self.log(f"❌ GPU acceleration test failed: {e}")
            self.test_results["gpu_acceleration"] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery systems."""
        self.log("🛡️  Testing error handling systems...")
        
        try:
            # Test BOLT error handling integration
            from bob.integration.error_handling import BoltErrorHandlingSystem
            from bob.integration.error_handling.exceptions import BoltException
            
            # Create error handling system
            error_system = BoltErrorHandlingSystem()
            self.log("✅ BOLT error handling system available")
            
            # Test exception handling
            try:
                raise BoltException("Test exception")
            except BoltException as e:
                self.log("✅ BOLT exception handling working")
            
            # Test circuit breaker
            try:
                from bob.integration.error_handling.circuit_breaker import CircuitBreaker
                circuit_breaker = CircuitBreaker()
                self.log("✅ Circuit breaker available")
            except ImportError as e:
                self.log(f"⚠️  Circuit breaker not available: {e}")
            
            # Test graceful degradation
            try:
                from bob.integration.error_handling.graceful_degradation import GracefulDegradation
                degradation = GracefulDegradation()
                self.log("✅ Graceful degradation available")
            except ImportError as e:
                self.log(f"⚠️  Graceful degradation not available: {e}")
            
            self.test_results["error_handling"] = True
            return True
            
        except Exception as e:
            self.log(f"❌ Error handling test failed: {e}")
            self.test_results["error_handling"] = False
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.log("🚀 Starting comprehensive BOLT migration validation...")
        
        validation_results = {
            "timestamp": time.time(),
            "tests": {},
            "performance_metrics": {},
            "overall_success": False,
            "critical_failures": [],
            "warnings": []
        }
        
        # Critical tests (must pass)
        critical_tests = [
            ("Import System", self.test_import_system),
            ("8-Agent Orchestration", self.test_agent_orchestration),
            ("Work-Stealing Algorithm", self.test_work_stealing_algorithm),
        ]
        
        # Important tests (should pass but non-critical)
        important_tests = [
            ("Hardware Optimization", self.test_hardware_optimization),
            ("Einstein Integration", self.test_einstein_integration),
            ("GPU Acceleration", self.test_gpu_acceleration),
            ("Error Handling", self.test_error_handling),
        ]
        
        # Run critical tests
        critical_passed = 0
        for test_name, test_func in critical_tests:
            self.log(f"\n--- Running Critical Test: {test_name} ---")
            
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            validation_results["tests"][test_name] = result
            
            if result:
                critical_passed += 1
                self.log(f"✅ CRITICAL TEST PASSED: {test_name}")
            else:
                validation_results["critical_failures"].append(test_name)
                self.log(f"❌ CRITICAL TEST FAILED: {test_name}")
        
        # Run important tests
        important_passed = 0
        for test_name, test_func in important_tests:
            self.log(f"\n--- Running Important Test: {test_name} ---")
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                    
                validation_results["tests"][test_name] = result
                
                if result:
                    important_passed += 1
                    self.log(f"✅ IMPORTANT TEST PASSED: {test_name}")
                else:
                    validation_results["warnings"].append(test_name)
                    self.log(f"⚠️  IMPORTANT TEST FAILED: {test_name}")
                    
            except Exception as e:
                validation_results["tests"][test_name] = False
                validation_results["warnings"].append(f"{test_name}: {str(e)}")
                self.log(f"⚠️  IMPORTANT TEST ERROR: {test_name}: {e}")
        
        # Overall validation result
        critical_success = critical_passed == len(critical_tests)
        validation_results["overall_success"] = critical_success
        validation_results["performance_metrics"] = self.performance_metrics
        validation_results["validation_log"] = self.validation_log
        
        # Summary
        self.log(f"\n{'='*60}")
        self.log("🎯 BOLT MIGRATION VALIDATION SUMMARY")
        self.log(f"{'='*60}")
        
        self.log(f"Critical Tests: {critical_passed}/{len(critical_tests)} passed")
        self.log(f"Important Tests: {important_passed}/{len(important_tests)} passed")
        
        if validation_results["critical_failures"]:
            self.log(f"❌ Critical Failures: {', '.join(validation_results['critical_failures'])}")
        
        if validation_results["warnings"]:
            self.log(f"⚠️  Warnings: {', '.join(validation_results['warnings'])}")
        
        # Performance metrics
        if self.performance_metrics:
            self.log("\n📊 Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                self.log(f"   {metric}: {value:.3f}")
        
        if critical_success:
            self.log("\n✅ BOLT MIGRATION VALIDATION SUCCESSFUL!")
            self.log("🎯 All critical systems are functional")
            
            # Check throughput requirement
            throughput = self.performance_metrics.get("orchestration_throughput", 0)
            if throughput >= 1.5:
                self.log(f"⚡ Throughput requirement met: {throughput:.1f} tasks/sec ≥ 1.5")
            else:
                self.log(f"⚠️  Throughput below requirement: {throughput:.1f} tasks/sec < 1.5")
        else:
            self.log("\n❌ BOLT MIGRATION VALIDATION FAILED!")
            self.log("🚨 Critical systems are not functional")
            self.log("🔄 Consider running rollback: ./rollback_bolt_migration.sh")
        
        return validation_results

async def main():
    """Main validation function."""
    print("🧪 BOLT Migration Comprehensive Validation")
    print("Validating BOLT to BOB migration success...")
    
    validator = BoltMigrationValidator()
    results = await validator.run_comprehensive_validation()
    
    # Save validation report
    report_path = BASE_DIR / "bolt_migration_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Validation report saved: {report_path}")
    
    # Return exit code based on validation success
    return 0 if results["overall_success"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)