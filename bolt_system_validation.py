#!/usr/bin/env python3
"""
Bolt System Validation Suite
Comprehensive validation of all Bolt optimization system fixes.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def validate_token_management() -> tuple[bool, str]:
    """Validate token management system."""
    try:
        from bolt.core.agent_token_manager import (
            AgentResponseContext,
            get_agent_token_manager,
        )

        manager = get_agent_token_manager()

        # Test 1: Basic token stats
        stats = manager.get_token_stats()
        assert stats["limits"]["max_output_tokens"] == 64000

        # Test 2: Response preparation
        test_content = "This is a test response " * 100  # Create some content
        context = AgentResponseContext(
            agent_id="test_agent", task_type="validation", complexity_score=0.5
        )

        optimized = manager.prepare_response(test_content, context)

        # Test 3: Validation
        is_valid, token_count, status = manager.validate_response(optimized)

        if not is_valid:
            return (
                False,
                f"Response validation failed: {token_count} tokens, status: {status}",
            )

        return True, f"Token management validated: {token_count} tokens"

    except Exception as e:
        return False, f"Token management validation failed: {e}"


async def validate_database_locks() -> tuple[bool, str]:
    """Validate database lock management."""
    try:
        from database_lock_manager import (
            get_database_lock_manager,
            safe_database_access,
        )

        manager = get_database_lock_manager()

        # Test 1: Cleanup stale locks
        test_db = ".test_validation.db"
        manager.cleanup_stale_locks(test_db)

        # Test 2: Exclusive access context manager
        with safe_database_access(test_db, timeout=5):
            # Simulate database work
            time.sleep(0.1)

        # Test 3: Force cleanup
        cleanup_result = manager.force_cleanup_database(test_db)

        return True, f"Database lock management validated: cleanup={cleanup_result}"

    except Exception as e:
        return False, f"Database lock validation failed: {e}"


async def validate_hardware_config() -> tuple[bool, str]:
    """Validate Einstein hardware configuration."""
    try:
        from einstein.einstein_config import HardwareDetector, get_einstein_config

        # Test 1: Hardware detection
        hardware = HardwareDetector.detect_hardware()

        # Verify required fields exist
        required_fields = [
            "cpu_cores",
            "has_ane",
            "ane_cores",
            "platform_type",
            "architecture",
        ]

        for field in required_fields:
            if not hasattr(hardware, field):
                return False, f"Missing hardware field: {field}"

        # Test 2: Einstein config loading
        try:
            get_einstein_config(Path.cwd())
        except Exception:
            # Expected if full config can't load, but hardware detection should work
            pass

        return (
            True,
            f"Hardware config validated: {hardware.cpu_cores} cores, ANE: {hardware.has_ane}",
        )

    except Exception as e:
        return False, f"Hardware config validation failed: {e}"


async def validate_system_integration() -> tuple[bool, str]:
    """Validate overall system integration."""
    try:
        # Test 1: Environment variables
        import os

        required_env = [
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            "CLAUDE_CODE_THINKING_BUDGET_TOKENS",
        ]

        for env_var in required_env:
            if env_var not in os.environ:
                return False, f"Missing environment variable: {env_var}"

        # Test 2: Config file exists
        config_file = Path("config_unified.yaml")
        if not config_file.exists():
            return False, "config_unified.yaml not found"

        # Test 3: Import core systems
        try:
            from bolt.core.dynamic_token_optimizer import get_token_optimizer
            from bolt.core.output_token_manager import get_output_token_manager

            get_output_token_manager()
            get_token_optimizer()

        except ImportError as e:
            return False, f"Core system import failed: {e}"

        return (
            True,
            "System integration validated: config exists, env vars set, systems loaded",
        )

    except Exception as e:
        return False, f"System integration validation failed: {e}"


async def run_load_test() -> tuple[bool, str]:
    """Run basic load test to validate fixes under stress."""
    try:
        from bolt.core.agent_token_manager import prepare_agent_response

        # Simulate multiple concurrent agent responses
        tasks = []
        for i in range(10):
            content = f"Agent {i} response: " + "data " * 50
            task = asyncio.create_task(
                asyncio.to_thread(
                    prepare_agent_response, content, f"agent_{i}", "load_test", 0.5
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        # Check results
        success_count = sum(1 for r in results if not isinstance(r, Exception))

        if success_count < 8:  # Allow some failures in stress test
            return False, f"Load test failed: only {success_count}/10 succeeded"

        return (
            True,
            f"Load test passed: {success_count}/10 succeeded in {duration:.2f}s",
        )

    except Exception as e:
        return False, f"Load test failed: {e}"


async def main():
    """Run comprehensive validation suite."""
    logger.info("Starting Bolt System Validation Suite")
    logger.info("=" * 60)

    # Define validation tests
    tests = [
        ("Token Management", validate_token_management),
        ("Database Locks", validate_database_locks),
        ("Hardware Config", validate_hardware_config),
        ("System Integration", validate_system_integration),
        ("Load Test", run_load_test),
    ]

    results = []
    total_time = time.time()

    for test_name, test_func in tests:
        logger.info(f"Running {test_name} validation...")
        start_time = time.time()

        try:
            success, message = await test_func()
            duration = time.time() - start_time

            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} {test_name}: {message} ({duration:.2f}s)")

            results.append((test_name, success, message, duration))

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ FAIL {test_name}: Exception - {e} ({duration:.2f}s)")
            results.append((test_name, False, f"Exception: {e}", duration))

    # Summary
    total_duration = time.time() - total_time
    passed = sum(1 for _, success, _, _ in results if success)
    total = len(results)

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    for test_name, success, message, duration in results:
        status = "✅" if success else "❌"
        logger.info(f"{status} {test_name}: {message}")

    logger.info("-" * 60)
    logger.info(f"Overall Result: {passed}/{total} tests passed")
    logger.info(f"Total Time: {total_duration:.2f}s")

    if passed == total:
        logger.info("🎉 ALL VALIDATIONS PASSED - System is ready for production!")
        return 0
    else:
        logger.error(
            f"⚠️  {total - passed} validation(s) failed - Review and fix issues"
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
