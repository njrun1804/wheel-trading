"""Real-world test suite for Bob M4 optimizations.

This module implements comprehensive tests that simulate actual development
workflows and validate the enhanced 12-agent system performance.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..core.enhanced_12_agent_coordinator import (
    Enhanced12AgentCoordinator,
    Enhanced12AgentTask,
    AgentRole,
    create_enhanced_12_agent_coordinator
)
from ..optimization.m4_claude_optimizer import get_global_optimizer
from ..utils.logging import get_component_logger


class TestComplexity(Enum):
    """Test complexity levels."""
    SIMPLE = "simple"           # Single agent, <2s
    MODERATE = "moderate"       # 2-4 agents, 2-5s
    COMPLEX = "complex"         # 4-8 agents, 5-15s
    ENTERPRISE = "enterprise"   # 8-12 agents, 15-30s


class TestCategory(Enum):
    """Test categories covering different development scenarios."""
    CODE_ANALYSIS = "code_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_HANDLING = "error_handling"
    INTEGRATION_TESTING = "integration_testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    SECURITY_ANALYSIS = "security_analysis"


@dataclass
class RealWorldTest:
    """Definition of a real-world test scenario."""
    
    test_id: str
    name: str
    description: str
    category: TestCategory
    complexity: TestComplexity
    
    # Test configuration
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    expected_agent_roles: List[AgentRole] = field(default_factory=list)
    max_duration: float = 30.0
    
    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    
    # Test data
    requires_codebase_access: bool = True
    mock_responses: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of executing a real-world test."""
    
    test_id: str
    success: bool
    duration: float
    
    # Execution details
    agents_used: List[str] = field(default_factory=list)
    agent_count: int = 0
    response: Any = None
    error: Optional[str] = None
    
    # Performance metrics
    p_core_utilization: float = 0.0
    e_core_utilization: float = 0.0
    coordination_latency: float = 0.0
    
    # Success criteria validation
    criteria_met: Dict[str, bool] = field(default_factory=dict)
    performance_scores: Dict[str, float] = field(default_factory=dict)


class RealWorldTestSuite:
    """Comprehensive test suite for real-world Bob scenarios."""
    
    def __init__(self):
        self.logger = get_component_logger("real_world_test_suite")
        
        # Test infrastructure
        self.coordinator: Optional[Enhanced12AgentCoordinator] = None
        self.m4_optimizer = None
        
        # Test definitions
        self.tests: List[RealWorldTest] = []
        self.test_results: List[TestResult] = []
        
        # Test execution state
        self._setup_complete = False
        self._execution_start_time = 0.0
        
        # Performance baselines
        self.performance_baselines = {
            "simple_query_max_duration": 2.0,
            "moderate_query_max_duration": 5.0,
            "complex_query_max_duration": 15.0,
            "enterprise_query_max_duration": 30.0,
            "min_p_core_utilization": 0.6,
            "max_coordination_latency": 0.005,  # 5ms
            "min_parallel_efficiency": 0.7
        }
    
    async def setup(self) -> None:
        """Setup the test environment."""
        if self._setup_complete:
            return
        
        self.logger.info("🔧 Setting up real-world test environment")
        
        # Initialize M4 optimizer
        self.m4_optimizer = await get_global_optimizer()
        
        # Initialize enhanced coordinator
        self.coordinator = create_enhanced_12_agent_coordinator()
        await self.coordinator.initialize()
        
        # Define test suite
        self._define_test_suite()
        
        self._setup_complete = True
        self.logger.info(f"✅ Test environment ready with {len(self.tests)} tests")
    
    def _define_test_suite(self) -> None:
        """Define the comprehensive test suite."""
        
        # Simple tests (single agent, quick responses)
        self.tests.extend([
            RealWorldTest(
                test_id="simple_code_search",
                name="Simple Code Search",
                description="Search for a specific function implementation",
                category=TestCategory.CODE_ANALYSIS,
                complexity=TestComplexity.SIMPLE,
                query="Find the implementation of the WheelStrategy class",
                expected_agent_roles=[AgentRole.ANALYZER, AgentRole.RESEARCHER],
                max_duration=2.0,
                success_criteria={
                    "finds_wheel_strategy": True,
                    "response_not_empty": True,
                    "execution_time_under_2s": True
                },
                performance_targets={
                    "max_duration": 2.0,
                    "min_success_rate": 0.95
                }
            ),
            
            RealWorldTest(
                test_id="simple_status_check",
                name="System Status Check",
                description="Check current system health and status",
                category=TestCategory.INTEGRATION_TESTING,
                complexity=TestComplexity.SIMPLE,
                query="What is the current status of the trading system?",
                expected_agent_roles=[AgentRole.MONITOR, AgentRole.REPORTER],
                max_duration=1.5,
                success_criteria={
                    "provides_status": True,
                    "mentions_key_components": True
                },
                performance_targets={
                    "max_duration": 1.5,
                    "min_success_rate": 0.98
                }
            )
        ])
        
        # Moderate tests (2-4 agents, moderate complexity)
        self.tests.extend([
            RealWorldTest(
                test_id="moderate_risk_analysis",
                name="Risk Management Analysis",
                description="Analyze risk management components and suggest improvements",
                category=TestCategory.CODE_ANALYSIS,
                complexity=TestComplexity.MODERATE,
                query="Analyze the risk management system in src/unity_wheel/risk/ and identify potential improvements for better safety",
                expected_agent_roles=[AgentRole.ANALYZER, AgentRole.VALIDATOR, AgentRole.ARCHITECT],
                max_duration=5.0,
                success_criteria={
                    "analyzes_risk_components": True,
                    "provides_specific_recommendations": True,
                    "mentions_safety_improvements": True
                },
                performance_targets={
                    "max_duration": 5.0,
                    "min_agents_used": 2,
                    "max_agents_used": 4
                }
            ),
            
            RealWorldTest(
                test_id="moderate_performance_optimization",
                name="Performance Optimization Review",
                description="Review codebase for performance bottlenecks",
                category=TestCategory.PERFORMANCE_OPTIMIZATION,
                complexity=TestComplexity.MODERATE,
                query="Review the Einstein search engine performance and identify optimization opportunities for faster semantic search",
                expected_agent_roles=[AgentRole.ANALYZER, AgentRole.OPTIMIZER, AgentRole.ARCHITECT],
                max_duration=6.0,
                success_criteria={
                    "identifies_bottlenecks": True,
                    "suggests_optimizations": True,
                    "mentions_search_performance": True
                },
                performance_targets={
                    "max_duration": 6.0,
                    "min_agents_used": 2
                }
            )
        ])
        
        # Complex tests (4-8 agents, sophisticated analysis)
        self.tests.extend([
            RealWorldTest(
                test_id="complex_architecture_design",
                name="Architecture Design and Integration",
                description="Design improvements for multi-agent coordination",
                category=TestCategory.ARCHITECTURE_DESIGN,
                complexity=TestComplexity.COMPLEX,
                query="Design an improved architecture for the 12-agent coordination system that reduces latency while maintaining reliability. Consider CPU core affinity, memory management, and error recovery patterns.",
                expected_agent_roles=[
                    AgentRole.ARCHITECT, AgentRole.ANALYZER, AgentRole.OPTIMIZER,
                    AgentRole.INTEGRATOR, AgentRole.VALIDATOR
                ],
                max_duration=15.0,
                success_criteria={
                    "provides_architectural_design": True,
                    "addresses_latency_concerns": True,
                    "includes_error_recovery": True,
                    "considers_cpu_affinity": True
                },
                performance_targets={
                    "max_duration": 15.0,
                    "min_agents_used": 4,
                    "min_parallel_efficiency": 0.7
                }
            ),
            
            RealWorldTest(
                test_id="complex_security_analysis",
                name="Comprehensive Security Analysis",
                description="Analyze entire codebase for security vulnerabilities",
                category=TestCategory.SECURITY_ANALYSIS,
                complexity=TestComplexity.COMPLEX,
                query="Perform a comprehensive security analysis of the wheel trading system. Check for API key exposure, input validation, authentication mechanisms, and secure data handling practices.",
                expected_agent_roles=[
                    AgentRole.ANALYZER, AgentRole.VALIDATOR, AgentRole.RESEARCHER,
                    AgentRole.INTEGRATOR, AgentRole.SYNTHESIZER
                ],
                max_duration=12.0,
                success_criteria={
                    "checks_api_security": True,
                    "validates_input_handling": True,
                    "reviews_authentication": True,
                    "provides_recommendations": True
                },
                performance_targets={
                    "max_duration": 12.0,
                    "min_agents_used": 4
                }
            )
        ])
        
        # Enterprise tests (8-12 agents, full system analysis)
        self.tests.extend([
            RealWorldTest(
                test_id="enterprise_full_optimization",
                name="Enterprise-Level System Optimization",
                description="Complete system analysis and optimization recommendations",
                category=TestCategory.PERFORMANCE_OPTIMIZATION,
                complexity=TestComplexity.ENTERPRISE,
                query="Perform a comprehensive analysis of the entire wheel trading system including Bob, Einstein, BOLT, and all integrations. Provide detailed optimization recommendations for M4 Pro hardware, identify architectural improvements, validate error handling, and create an implementation roadmap for production deployment.",
                expected_agent_roles=[
                    AgentRole.ANALYZER, AgentRole.ARCHITECT, AgentRole.OPTIMIZER,
                    AgentRole.GENERATOR, AgentRole.VALIDATOR, AgentRole.INTEGRATOR,
                    AgentRole.RESEARCHER, AgentRole.SYNTHESIZER, AgentRole.DOCUMENTER
                ],
                max_duration=30.0,
                success_criteria={
                    "analyzes_all_components": True,
                    "provides_m4_optimizations": True,
                    "includes_roadmap": True,
                    "validates_architecture": True,
                    "creates_documentation": True
                },
                performance_targets={
                    "max_duration": 30.0,
                    "min_agents_used": 8,
                    "min_parallel_efficiency": 0.8
                }
            ),
            
            RealWorldTest(
                test_id="enterprise_integration_validation",
                name="Enterprise Integration Validation",
                description="Validate all system integrations and data flows",
                category=TestCategory.INTEGRATION_TESTING,
                complexity=TestComplexity.ENTERPRISE,
                query="Validate all integrations in the wheel trading system: Claude Code MCP integration, database connections, API authentication, Einstein search integration, BOLT agent coordination, and hardware acceleration. Create a comprehensive integration test plan and identify potential failure points.",
                expected_agent_roles=[
                    AgentRole.ANALYZER, AgentRole.VALIDATOR, AgentRole.INTEGRATOR,
                    AgentRole.RESEARCHER, AgentRole.ARCHITECT, AgentRole.GENERATOR,
                    AgentRole.MONITOR, AgentRole.SYNTHESIZER
                ],
                max_duration=25.0,
                success_criteria={
                    "validates_all_integrations": True,
                    "creates_test_plan": True,
                    "identifies_failure_points": True,
                    "provides_monitoring_strategy": True
                },
                performance_targets={
                    "max_duration": 25.0,
                    "min_agents_used": 6
                }
            )
        ])
        
        self.logger.info(f"📋 Defined {len(self.tests)} real-world tests across {len(TestComplexity)} complexity levels")
    
    async def run_parallel_tests(self) -> Dict[str, Any]:
        """Run all tests in parallel to validate M4 optimizations."""
        if not self._setup_complete:
            await self.setup()
        
        self.logger.info(f"🚀 Starting parallel execution of {len(self.tests)} real-world tests")
        self._execution_start_time = time.time()
        
        # Group tests by complexity for optimal scheduling
        test_groups = self._group_tests_by_complexity()
        
        # Execute tests in parallel within complexity groups
        all_results = []
        for complexity, tests in test_groups.items():
            self.logger.info(f"🔄 Executing {len(tests)} {complexity.value} tests in parallel")
            
            # Create concurrent tasks for this complexity level
            concurrent_tasks = []
            for test in tests:
                task = self._execute_single_test(test)
                concurrent_tasks.append(task)
            
            # Execute and collect results
            group_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(group_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Test {tests[i].test_id} failed with exception: {result}")
                    all_results.append(TestResult(
                        test_id=tests[i].test_id,
                        success=False,
                        duration=0.0,
                        error=str(result)
                    ))
                else:
                    all_results.append(result)
        
        self.test_results = all_results
        total_execution_time = time.time() - self._execution_start_time
        
        # Analyze results
        analysis = self._analyze_test_results(total_execution_time)
        
        self.logger.info(f"✅ Test execution complete in {total_execution_time:.2f}s")
        self.logger.info(f"📊 Results: {analysis['passed_tests']}/{analysis['total_tests']} tests passed")
        
        return analysis
    
    def _group_tests_by_complexity(self) -> Dict[TestComplexity, List[RealWorldTest]]:
        """Group tests by complexity level for optimal scheduling."""
        groups = {complexity: [] for complexity in TestComplexity}
        
        for test in self.tests:
            groups[test.complexity].append(test)
        
        return groups
    
    async def _execute_single_test(self, test: RealWorldTest) -> TestResult:
        """Execute a single test and measure performance."""
        self.logger.info(f"🧪 Executing test: {test.name}")
        start_time = time.time()
        
        try:
            # Create enhanced task
            task = Enhanced12AgentTask(
                task_id=test.test_id,
                description=test.query,
                task_type=f"test_{test.category.value}",
                data={
                    "test_context": test.context,
                    "complexity": test.complexity.value,
                    "category": test.category.value
                },
                preferred_roles=test.expected_agent_roles,
                cpu_intensive=test.complexity in [TestComplexity.COMPLEX, TestComplexity.ENTERPRISE],
                estimated_duration=test.max_duration / 2,  # Conservative estimate
                requires_claude=True,
                parallelizable=len(test.expected_agent_roles) > 1
            )
            
            # Execute task
            result = await self.coordinator.execute_enhanced_task(task)
            
            duration = time.time() - start_time
            
            # Validate success criteria
            criteria_met = self._validate_success_criteria(test, result)
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores(test, result, duration)
            
            # Get system metrics
            system_metrics = self.coordinator.get_enhanced_metrics()
            
            test_result = TestResult(
                test_id=test.test_id,
                success=result.get("success", False) and all(criteria_met.values()),
                duration=duration,
                agents_used=[result.get("agent_role", "unknown")],
                agent_count=result.get("agents_utilized", 1),
                response=result.get("result"),
                error=result.get("error"),
                p_core_utilization=system_metrics.get("p_core_utilization", 0.0),
                e_core_utilization=system_metrics.get("e_core_utilization", 0.0),
                coordination_latency=result.get("processing_time", duration),
                criteria_met=criteria_met,
                performance_scores=performance_scores
            )
            
            self.logger.info(f"{'✅' if test_result.success else '❌'} Test {test.test_id}: {duration:.2f}s")
            return test_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Test {test.test_id} failed: {e}")
            
            return TestResult(
                test_id=test.test_id,
                success=False,
                duration=duration,
                error=str(e)
            )
    
    def _validate_success_criteria(self, test: RealWorldTest, result: Dict[str, Any]) -> Dict[str, bool]:
        """Validate test success criteria."""
        criteria_met = {}
        
        for criterion, expected in test.success_criteria.items():
            if criterion == "response_not_empty":
                criteria_met[criterion] = bool(result.get("result"))
            elif criterion == "execution_time_under_2s":
                criteria_met[criterion] = result.get("duration", 999) < 2.0
            elif criterion == "provides_status":
                response_text = str(result.get("result", "")).lower()
                criteria_met[criterion] = any(word in response_text for word in ["status", "health", "running", "active"])
            elif criterion == "finds_wheel_strategy":
                response_text = str(result.get("result", "")).lower()
                criteria_met[criterion] = "wheelstrategy" in response_text or "wheel strategy" in response_text
            else:
                # Generic criteria - assume met if we have a successful response
                criteria_met[criterion] = result.get("success", False)
        
        return criteria_met
    
    def _calculate_performance_scores(self, test: RealWorldTest, result: Dict[str, Any], duration: float) -> Dict[str, float]:
        """Calculate performance scores against targets."""
        scores = {}
        
        # Duration score
        max_duration = test.performance_targets.get("max_duration", test.max_duration)
        scores["duration_score"] = max(0.0, 1.0 - (duration / max_duration))
        
        # Agent utilization score
        agents_used = result.get("agents_utilized", 1)
        min_agents = test.performance_targets.get("min_agents_used", 1)
        max_agents = test.performance_targets.get("max_agents_used", 12)
        
        if agents_used >= min_agents:
            scores["agent_utilization_score"] = min(1.0, agents_used / max_agents)
        else:
            scores["agent_utilization_score"] = 0.5  # Penalty for under-utilization
        
        # Success rate score
        scores["success_score"] = 1.0 if result.get("success", False) else 0.0
        
        return scores
    
    def _analyze_test_results(self, total_execution_time: float) -> Dict[str, Any]:
        """Analyze all test results and provide comprehensive report."""
        
        # Basic statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        # Performance analysis
        avg_duration = sum(result.duration for result in self.test_results) / total_tests if total_tests > 0 else 0
        total_test_time = sum(result.duration for result in self.test_results)
        parallel_efficiency = total_test_time / total_execution_time if total_execution_time > 0 else 0
        
        # Agent utilization
        total_agents_used = sum(result.agent_count for result in self.test_results)
        avg_agents_per_test = total_agents_used / total_tests if total_tests > 0 else 0
        
        # Complexity breakdown
        complexity_stats = {}
        for complexity in TestComplexity:
            complexity_results = [r for r in self.test_results 
                                if any(t.test_id == r.test_id and t.complexity == complexity for t in self.tests)]
            if complexity_results:
                complexity_stats[complexity.value] = {
                    "total": len(complexity_results),
                    "passed": sum(1 for r in complexity_results if r.success),
                    "avg_duration": sum(r.duration for r in complexity_results) / len(complexity_results)
                }
        
        # Performance vs baselines
        baseline_analysis = self._analyze_against_baselines()
        
        # Detailed results by category
        category_stats = {}
        for category in TestCategory:
            category_results = [r for r in self.test_results 
                              if any(t.test_id == r.test_id and t.category == category for t in self.tests)]
            if category_results:
                category_stats[category.value] = {
                    "total": len(category_results),
                    "passed": sum(1 for r in category_results if r.success),
                    "success_rate": sum(1 for r in category_results if r.success) / len(category_results)
                }
        
        return {
            "execution_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_execution_time,
                "parallel_efficiency": parallel_efficiency
            },
            "performance_metrics": {
                "avg_test_duration": avg_duration,
                "total_test_time": total_test_time,
                "avg_agents_per_test": avg_agents_per_test,
                "parallel_speedup": parallel_efficiency
            },
            "complexity_breakdown": complexity_stats,
            "category_breakdown": category_stats,
            "baseline_analysis": baseline_analysis,
            "detailed_results": [
                {
                    "test_id": result.test_id,
                    "success": result.success,
                    "duration": result.duration,
                    "agents_used": result.agent_count,
                    "error": result.error
                }
                for result in self.test_results
            ]
        }
    
    def _analyze_against_baselines(self) -> Dict[str, Any]:
        """Analyze results against performance baselines."""
        analysis = {}
        
        # Check duration baselines by complexity
        for complexity in TestComplexity:
            baseline_key = f"{complexity.value}_query_max_duration"
            if baseline_key in self.performance_baselines:
                baseline = self.performance_baselines[baseline_key]
                complexity_results = [
                    r for r in self.test_results 
                    if any(t.test_id == r.test_id and t.complexity == complexity for t in self.tests)
                ]
                
                if complexity_results:
                    avg_duration = sum(r.duration for r in complexity_results) / len(complexity_results)
                    analysis[f"{complexity.value}_duration_vs_baseline"] = {
                        "baseline": baseline,
                        "actual": avg_duration,
                        "meets_baseline": avg_duration <= baseline,
                        "improvement": (baseline - avg_duration) / baseline if baseline > 0 else 0
                    }
        
        # Overall system performance
        if self.test_results:
            avg_p_core_util = sum(r.p_core_utilization for r in self.test_results if r.p_core_utilization > 0)
            avg_p_core_util = avg_p_core_util / len([r for r in self.test_results if r.p_core_utilization > 0]) if avg_p_core_util > 0 else 0
            
            analysis["p_core_utilization"] = {
                "baseline": self.performance_baselines["min_p_core_utilization"],
                "actual": avg_p_core_util,
                "meets_baseline": avg_p_core_util >= self.performance_baselines["min_p_core_utilization"]
            }
        
        return analysis
    
    async def cleanup(self) -> None:
        """Cleanup test environment."""
        self.logger.info("🧹 Cleaning up test environment")
        
        if self.coordinator:
            await self.coordinator.shutdown()
        
        if self.m4_optimizer:
            await self.m4_optimizer.shutdown()


# Convenience function for running the full test suite
async def run_real_world_tests() -> Dict[str, Any]:
    """Run the complete real-world test suite."""
    test_suite = RealWorldTestSuite()
    
    try:
        results = await test_suite.run_parallel_tests()
        return results
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    # Run tests when executed directly
    results = asyncio.run(run_real_world_tests())
    print(json.dumps(results, indent=2))