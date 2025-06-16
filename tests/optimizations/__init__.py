"""
Comprehensive Test Suite for Hardware Optimizations

This module provides comprehensive testing for both buffer fix and ANE acceleration
optimizations with regression testing, performance validation, and CI/CD integration.
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .test_ane_acceleration import *
from .test_buffer_optimization import *
from .test_ci_validation import *
from .test_integration_suite import *
from .test_performance_regression import *

__all__ = [
    "BufferOptimizationTestSuite",
    "ANEAccelerationTestSuite",
    "IntegrationTestSuite",
    "PerformanceRegressionTestSuite",
    "CIValidationTestSuite",
    "run_all_optimization_tests",
    "generate_optimization_report",
]
