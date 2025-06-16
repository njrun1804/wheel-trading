"""
Automated test runner for NLP validation suite.

Runs comprehensive testing and validation pipeline for natural language
processing system including performance benchmarks and deployment checks.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import test components
sys.path.append(str(Path(__file__).parent.parent))

from src.unity_wheel.orchestrator.natural_language_processor import NaturalLanguageProcessor
from src.unity_wheel.orchestrator.nlp_validator import NLPValidator
from src.unity_wheel.orchestrator.nlp_metrics import get_metrics_collector


class NLPValidationSuite:
    """Comprehensive validation suite for NLP system."""
    
    def __init__(self):
        """Initialize validation suite."""
        self.processor = NaturalLanguageProcessor()
        self.validator = NLPValidator()
        self.metrics_collector = get_metrics_collector()
        self.results = {}
    
    async def run_classification_tests(self) -> Dict[str, Any]:
        """Run classification accuracy tests."""
        logger.info("Running classification tests...")
        
        test_cases = [
            # Fix commands
            ("fix the database connection error", "fix"),
            ("resolve import issues in storage module", "fix"),
            ("debug the slow query performance", "fix"),
            
            # Create commands  
            ("create unit tests for risk module", "create"),
            ("build a new volatility analyzer", "create"),
            ("generate configuration for production", "create"),
            
            # Optimize commands
            ("optimize the trading strategy performance", "optimize"),
            ("improve memory usage in backtesting", "optimize"),
            ("speed up option pricing calculations", "optimize"),
            
            # Analyze commands
            ("analyze portfolio risk exposure", "analyze"),
            ("review code quality metrics", "analyze"),
            ("examine trading performance patterns", "analyze"),
        ]
        
        correct_classifications = 0
        total_tests = len(test_cases)
        classification_times = []
        
        for command, expected_type in test_cases:
            start_time = time.time()
            intent = await self.processor.classify_intent(command)
            classification_time = time.time() - start_time
            
            classification_times.append(classification_time * 1000)  # Convert to ms
            
            if intent.command_type.value == expected_type:
                correct_classifications += 1
            else:
                logger.warning(
                    f"Misclassified: '{command}' -> {intent.command_type.value} "
                    f"(expected: {expected_type})"
                )
            
            # Record metrics
            await self.metrics_collector.record_classification(
                command=command,
                predicted_type=intent.command_type,
                actual_type=None,  # Would be from user feedback
                confidence=intent.confidence,
                execution_time_ms=classification_time * 1000
            )
        
        accuracy = correct_classifications / total_tests
        avg_time = sum(classification_times) / len(classification_times)
        
        result = {
            'accuracy': accuracy,
            'correct_classifications': correct_classifications,
            'total_tests': total_tests,
            'average_time_ms': avg_time,
            'max_time_ms': max(classification_times),
            'min_time_ms': min(classification_times),
            'pass': accuracy >= 0.9
        }
        
        logger.info(f"Classification accuracy: {accuracy:.1%} ({correct_classifications}/{total_tests})")
        logger.info(f"Average classification time: {avg_time:.1f}ms")
        
        return result
    
    async def run_context_extraction_tests(self) -> Dict[str, Any]:
        """Run context extraction accuracy tests."""
        logger.info("Running context extraction tests...")
        
        test_cases = [
            {
                'command': "fix import error in src/unity_wheel/storage/storage.py",
                'expected_files': ['storage.py'],
                'expected_confidence': 0.8
            },
            {
                'command': "optimize wheel strategy performance for Unity stock",
                'expected_files': ['wheel.py', 'strategy'],
                'expected_confidence': 0.7
            },
            {
                'command': "analyze risk management across all modules",
                'expected_files': ['risk'],
                'expected_confidence': 0.6
            }
        ]
        
        total_tests = len(test_cases)
        passed_tests = 0
        context_times = []
        
        for test_case in test_cases:
            start_time = time.time()
            context = await self.processor.extract_context(test_case['command'])
            context_time = time.time() - start_time
            
            context_times.append(context_time * 1000)
            
            # Check if any expected files are found
            files_found = any(
                any(expected in file for expected in test_case['expected_files'])
                for file in context.relevant_files
            )
            
            confidence_ok = context.confidence_score >= test_case['expected_confidence']
            
            if files_found and confidence_ok:
                passed_tests += 1
            else:
                logger.warning(
                    f"Context extraction failed: '{test_case['command']}' -> "
                    f"files_found={files_found}, confidence={context.confidence_score:.2f}"
                )
        
        accuracy = passed_tests / total_tests
        avg_time = sum(context_times) / len(context_times)
        
        result = {
            'accuracy': accuracy,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'average_time_ms': avg_time,
            'max_time_ms': max(context_times),
            'min_time_ms': min(context_times),
            'pass': accuracy >= 0.7
        }
        
        logger.info(f"Context extraction accuracy: {accuracy:.1%} ({passed_tests}/{total_tests})")
        logger.info(f"Average context extraction time: {avg_time:.1f}ms")
        
        return result
    
    async def run_end_to_end_tests(self) -> Dict[str, Any]:
        """Run end-to-end processing tests."""
        logger.info("Running end-to-end tests...")
        
        test_commands = [
            "fix database connection issues in storage module",
            "create comprehensive tests for wheel strategy",
            "optimize option pricing performance",
            "analyze current portfolio risk metrics"
        ]
        
        total_tests = len(test_commands)
        successful_tests = 0
        execution_times = []
        error_details = []
        
        for command in test_commands:
            try:
                start_time = time.time()
                result = await self.processor.process_command(command)
                execution_time = time.time() - start_time
                
                execution_times.append(execution_time * 1000)
                
                if result.success or not result.needs_clarification:
                    successful_tests += 1
                else:
                    error_details.append({
                        'command': command,
                        'error': result.error_message,
                        'needs_clarification': result.needs_clarification
                    })
                
                # Record execution metrics
                await self.metrics_collector.record_execution(
                    command=command,
                    command_type=await self._get_command_type(command),
                    success=result.success,
                    execution_time=execution_time
                )
                
            except Exception as e:
                logger.error(f"End-to-end test failed for '{command}': {e}")
                error_details.append({
                    'command': command,
                    'error': str(e),
                    'exception': True
                })
        
        success_rate = successful_tests / total_tests
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        result = {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'average_time_ms': avg_time,
            'max_time_ms': max(execution_times) if execution_times else 0,
            'min_time_ms': min(execution_times) if execution_times else 0,
            'error_details': error_details,
            'pass': success_rate >= 0.8
        }
        
        logger.info(f"End-to-end success rate: {success_rate:.1%} ({successful_tests}/{total_tests})")
        logger.info(f"Average execution time: {avg_time:.1f}ms")
        
        return result
    
    async def _get_command_type(self, command: str):
        """Helper to get command type for metrics."""
        intent = await self.processor.classify_intent(command)
        return intent.command_type
    
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        logger.info("Running performance benchmarks...")
        
        benchmark_result = await self.validator.benchmark_performance(num_iterations=50)
        
        # Performance thresholds
        thresholds = {
            'mean_latency_ms': 500,
            'p95_latency_ms': 1000,
            'p99_latency_ms': 2000
        }
        
        passed_benchmarks = 0
        total_benchmarks = len(thresholds)
        
        for metric, threshold in thresholds.items():
            if benchmark_result[metric] <= threshold:
                passed_benchmarks += 1
            else:
                logger.warning(
                    f"Performance benchmark failed: {metric}={benchmark_result[metric]:.1f} "
                    f"(threshold: {threshold})"
                )
        
        benchmark_result.update({
            'passed_benchmarks': passed_benchmarks,
            'total_benchmarks': total_benchmarks,
            'pass': passed_benchmarks == total_benchmarks
        })
        
        logger.info(f"Performance benchmarks: {passed_benchmarks}/{total_benchmarks} passed")
        logger.info(f"Mean latency: {benchmark_result['mean_latency_ms']:.1f}ms")
        
        return benchmark_result
    
    async def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests."""
        logger.info("Running regression tests...")
        
        regression_result = await self.validator.run_regression_tests()
        
        result = {
            'pass_rate': regression_result.pass_rate,
            'passed_count': regression_result.passed_count,
            'failed_count': regression_result.failed_count,
            'total_count': regression_result.total_count,
            'regression_detected': regression_result.regression_detected,
            'failed_tests': regression_result.failed_tests,
            'performance_issues': regression_result.performance_issues,
            'pass': regression_result.success
        }
        
        logger.info(f"Regression tests: {regression_result.pass_rate:.1%} pass rate")
        if regression_result.regression_detected:
            logger.error("REGRESSION DETECTED!")
        
        return result
    
    async def run_deployment_validation(self) -> Dict[str, Any]:
        """Run deployment readiness validation."""
        logger.info("Running deployment validation...")
        
        deployment_result = await self.validator.validate_for_deployment()
        
        result = {
            'ready_for_deployment': deployment_result.ready_for_deployment,
            'classification_accuracy': deployment_result.classification_accuracy,
            'average_latency_ms': deployment_result.average_latency,
            'error_rate': deployment_result.error_rate,
            'blocking_issues': deployment_result.blocking_issues,
            'recommendations': deployment_result.recommendations,
            'pass': deployment_result.ready_for_deployment
        }
        
        logger.info(f"Deployment ready: {deployment_result.ready_for_deployment}")
        if deployment_result.blocking_issues:
            logger.error(f"Blocking issues: {deployment_result.blocking_issues}")
        
        return result
    
    async def run_full_suite(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting NLP validation suite...")
        suite_start_time = time.time()
        
        # Run all test categories
        self.results['classification'] = await self.run_classification_tests()
        self.results['context_extraction'] = await self.run_context_extraction_tests()
        self.results['end_to_end'] = await self.run_end_to_end_tests()
        self.results['performance'] = await self.run_performance_benchmarks()
        self.results['regression'] = await self.run_regression_tests()
        self.results['deployment'] = await self.run_deployment_validation()
        
        # Calculate overall results
        suite_time = time.time() - suite_start_time
        
        # Determine if all tests passed
        test_categories = ['classification', 'context_extraction', 'end_to_end', 
                          'performance', 'regression', 'deployment']
        passed_categories = sum(1 for cat in test_categories if self.results[cat]['pass'])
        
        overall_pass = passed_categories == len(test_categories)
        
        self.results['summary'] = {
            'overall_pass': overall_pass,
            'passed_categories': passed_categories,
            'total_categories': len(test_categories),
            'suite_execution_time_s': suite_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log summary
        logger.info(f"Validation suite completed in {suite_time:.1f}s")
        logger.info(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")
        logger.info(f"Categories passed: {passed_categories}/{len(test_categories)}")
        
        return self.results
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate comprehensive validation report."""
        if not self.results:
            raise ValueError("No results available. Run validation first.")
        
        # Default output path
        if not output_path:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = f"nlp_validation_report_{timestamp}.json"
        
        # Add metrics data
        self.results['metrics'] = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {output_path}")
        return output_path


async def main():
    """Main entry point for validation suite."""
    suite = NLPValidationSuite()
    
    try:
        # Run full validation suite
        results = await suite.run_full_suite()
        
        # Generate report
        report_path = suite.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("NLP VALIDATION SUITE SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"Overall Result: {'PASS' if summary['overall_pass'] else 'FAIL'}")
        print(f"Execution Time: {summary['suite_execution_time_s']:.1f}s")
        print(f"Categories Passed: {summary['passed_categories']}/{summary['total_categories']}")
        
        print("\nCategory Results:")
        for category, result in results.items():
            if category != 'summary' and category != 'metrics':
                status = "PASS" if result.get('pass', False) else "FAIL"
                print(f"  {category.replace('_', ' ').title()}: {status}")
        
        print(f"\nReport saved to: {report_path}")
        
        # Exit with appropriate code
        sys.exit(0 if summary['overall_pass'] else 1)
        
    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())