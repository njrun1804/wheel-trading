"""
Buffer Guards and Runtime Assertions

This module provides runtime guards and assertions for critical GPU buffer operations
to prevent buffer-stride bugs from recurring.

Key Features:
- Buffer size and shape validation
- Memory alignment checks
- MLX array validation
- Performance monitoring
- Automatic fallback mechanisms
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)


class BufferGuardError(Exception):
    """Exception raised when buffer validation fails."""

    pass


class BufferGuard:
    """Runtime buffer validation and guard system."""

    def __init__(self, strict_mode: bool = False, performance_tracking: bool = True):
        """
        Initialize buffer guard system.

        Args:
            strict_mode: If True, raises exceptions on validation failures
            performance_tracking: If True, tracks performance metrics
        """
        self.strict_mode = strict_mode
        self.performance_tracking = performance_tracking
        self.validation_stats = {
            "total_validations": 0,
            "failures": 0,
            "warnings": 0,
            "avg_validation_time": 0.0,
        }

    def validate_numpy_buffer(
        self,
        array: np.ndarray,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: np.dtype | None = None,
        min_size: int = 1,
        max_size: int | None = None,
        require_contiguous: bool = False,
        require_aligned: bool = False,
        alignment: int = 32,
    ) -> bool:
        """
        Comprehensive numpy buffer validation.

        Args:
            array: Array to validate
            expected_shape: Expected shape tuple
            expected_dtype: Expected numpy dtype
            min_size: Minimum number of elements
            max_size: Maximum number of elements
            require_contiguous: Whether array must be contiguous
            require_aligned: Whether array must be memory aligned
            alignment: Required memory alignment in bytes

        Returns:
            True if validation passes

        Raises:
            BufferGuardError: If validation fails and strict_mode is True
        """
        start_time = time.perf_counter() if self.performance_tracking else 0

        try:
            self.validation_stats["total_validations"] += 1

            # Basic null check
            if array is None:
                self._handle_validation_error("Array is None")
                return False

            # Type check
            if not isinstance(array, np.ndarray):
                self._handle_validation_error(
                    f"Expected numpy array, got {type(array)}"
                )
                return False

            # Shape validation
            if expected_shape is not None and array.shape != expected_shape:
                self._handle_validation_error(
                    f"Shape mismatch: expected {expected_shape}, got {array.shape}"
                )
                return False

            # Dtype validation
            if expected_dtype is not None and array.dtype != expected_dtype:
                self._handle_validation_error(
                    f"Dtype mismatch: expected {expected_dtype}, got {array.dtype}"
                )
                return False

            # Size validation
            if array.size < min_size:
                self._handle_validation_error(
                    f"Array too small: {array.size} < {min_size}"
                )
                return False

            if max_size is not None and array.size > max_size:
                self._handle_validation_error(
                    f"Array too large: {array.size} > {max_size}"
                )
                return False

            # Contiguity check
            if require_contiguous and not array.flags.c_contiguous:
                self._handle_validation_error("Array is not C-contiguous")
                return False

            # Memory alignment check
            if require_aligned and hasattr(array, "ctypes"):
                try:
                    address = array.ctypes.data
                    if address % alignment != 0:
                        self._handle_validation_error(
                            f"Array not aligned: address {address} % {alignment} != 0"
                        )
                        return False
                except (AttributeError, ValueError):
                    self._handle_validation_warning("Could not check memory alignment")

            return True

        except Exception as e:
            self._handle_validation_error(f"Validation error: {e}")
            return False

        finally:
            if self.performance_tracking:
                elapsed = time.perf_counter() - start_time
                self._update_performance_stats(elapsed)

    def validate_mlx_buffer(
        self,
        array,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: str | None = None,
        min_size: int = 1,
        max_size: int | None = None,
    ) -> bool:
        """
        MLX array buffer validation.

        Args:
            array: MLX array to validate
            expected_shape: Expected shape tuple
            expected_dtype: Expected MLX dtype string
            min_size: Minimum number of elements
            max_size: Maximum number of elements

        Returns:
            True if validation passes
        """
        if not MLX_AVAILABLE:
            logger.warning("MLX not available, skipping MLX buffer validation")
            return True

        start_time = time.perf_counter() if self.performance_tracking else 0

        try:
            self.validation_stats["total_validations"] += 1

            # Basic checks
            if array is None:
                self._handle_validation_error("MLX array is None")
                return False

            # Check if it's an MLX array
            if not hasattr(array, "shape") or not hasattr(array, "dtype"):
                self._handle_validation_error(f"Not a valid MLX array: {type(array)}")
                return False

            # Shape validation
            if expected_shape is not None and array.shape != expected_shape:
                self._handle_validation_error(
                    f"MLX shape mismatch: expected {expected_shape}, got {array.shape}"
                )
                return False

            # Dtype validation
            if expected_dtype is not None and str(array.dtype) != expected_dtype:
                self._handle_validation_error(
                    f"MLX dtype mismatch: expected {expected_dtype}, got {array.dtype}"
                )
                return False

            # Size validation
            try:
                size = array.size
                if size < min_size:
                    self._handle_validation_error(
                        f"MLX array too small: {size} < {min_size}"
                    )
                    return False

                if max_size is not None and size > max_size:
                    self._handle_validation_error(
                        f"MLX array too large: {size} > {max_size}"
                    )
                    return False
            except Exception as e:
                self._handle_validation_error(f"Could not get MLX array size: {e}")
                return False

            # Memory corruption check
            try:
                _ = array.nbytes
                _ = str(array.dtype)
            except Exception as e:
                self._handle_validation_error(f"MLX array appears corrupted: {e}")
                return False

            return True

        except Exception as e:
            self._handle_validation_error(f"MLX validation error: {e}")
            return False

        finally:
            if self.performance_tracking:
                elapsed = time.perf_counter() - start_time
                self._update_performance_stats(elapsed)

    def _handle_validation_error(self, message: str):
        """Handle validation error based on strict mode."""
        self.validation_stats["failures"] += 1
        logger.error(f"Buffer validation failed: {message}")

        if self.strict_mode:
            raise BufferGuardError(message)

    def _handle_validation_warning(self, message: str):
        """Handle validation warning."""
        self.validation_stats["warnings"] += 1
        logger.warning(f"Buffer validation warning: {message}")

    def _update_performance_stats(self, elapsed_time: float):
        """Update performance tracking statistics."""
        if self.validation_stats["total_validations"] == 1:
            self.validation_stats["avg_validation_time"] = elapsed_time
        else:
            # Running average
            count = self.validation_stats["total_validations"]
            current_avg = self.validation_stats["avg_validation_time"]
            self.validation_stats["avg_validation_time"] = (
                current_avg * (count - 1) + elapsed_time
            ) / count

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()

    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "failures": 0,
            "warnings": 0,
            "avg_validation_time": 0.0,
        }


# Global guard instance
_global_guard = BufferGuard()


def get_buffer_guard() -> BufferGuard:
    """Get the global buffer guard instance."""
    return _global_guard


def set_strict_mode(enabled: bool):
    """Enable or disable strict validation mode globally."""
    _global_guard.strict_mode = enabled


def validate_buffer_shape(
    array: np.ndarray | Any, expected_shape: tuple[int, ...]
) -> bool:
    """Quick buffer shape validation."""
    guard = get_buffer_guard()

    if isinstance(array, np.ndarray):
        return guard.validate_numpy_buffer(array, expected_shape=expected_shape)
    elif MLX_AVAILABLE and hasattr(array, "shape"):
        return guard.validate_mlx_buffer(array, expected_shape=expected_shape)
    else:
        logger.warning(f"Unknown array type for shape validation: {type(array)}")
        return False


def validate_buffer_dtype(
    array: np.ndarray | Any, expected_dtype: np.dtype | str
) -> bool:
    """Quick buffer dtype validation."""
    guard = get_buffer_guard()

    if isinstance(array, np.ndarray):
        return guard.validate_numpy_buffer(array, expected_dtype=expected_dtype)
    elif MLX_AVAILABLE and hasattr(array, "dtype"):
        return guard.validate_mlx_buffer(array, expected_dtype=str(expected_dtype))
    else:
        logger.warning(f"Unknown array type for dtype validation: {type(array)}")
        return False


def assert_buffer_valid(
    array: np.ndarray | Any,
    name: str = "buffer",
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | str | None = None,
    min_size: int = 1,
    max_size: int | None = None,
):
    """Assert that a buffer is valid, raising exception if not."""
    guard = get_buffer_guard()

    # Temporarily enable strict mode for assertions
    original_strict = guard.strict_mode
    guard.strict_mode = True

    try:
        if isinstance(array, np.ndarray):
            success = guard.validate_numpy_buffer(
                array,
                expected_shape=shape,
                expected_dtype=dtype,
                min_size=min_size,
                max_size=max_size,
            )
        elif MLX_AVAILABLE and hasattr(array, "shape"):
            success = guard.validate_mlx_buffer(
                array,
                expected_shape=shape,
                expected_dtype=str(dtype) if dtype else None,
                min_size=min_size,
                max_size=max_size,
            )
        else:
            raise BufferGuardError(f"Unknown array type for {name}: {type(array)}")

        if not success:
            raise BufferGuardError(f"Buffer validation failed for {name}")

    finally:
        guard.strict_mode = original_strict


def buffer_guard(
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | str | None = None,
    min_size: int = 1,
    max_size: int | None = None,
    require_contiguous: bool = False,
    require_aligned: bool = False,
    validate_inputs: bool = True,
    validate_outputs: bool = True,
):
    """
    Decorator to add buffer validation to functions.

    Args:
        shape: Expected shape for array arguments
        dtype: Expected dtype for array arguments
        min_size: Minimum array size
        max_size: Maximum array size
        require_contiguous: Whether arrays must be contiguous
        require_aligned: Whether arrays must be memory aligned
        validate_inputs: Whether to validate input arrays
        validate_outputs: Whether to validate output arrays
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            guard = get_buffer_guard()

            # Validate inputs
            if validate_inputs:
                for _i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray):
                        guard.validate_numpy_buffer(
                            arg,
                            expected_shape=shape,
                            expected_dtype=dtype,
                            min_size=min_size,
                            max_size=max_size,
                            require_contiguous=require_contiguous,
                            require_aligned=require_aligned,
                        )
                    elif MLX_AVAILABLE and hasattr(arg, "shape"):
                        guard.validate_mlx_buffer(
                            arg,
                            expected_shape=shape,
                            expected_dtype=str(dtype) if dtype else None,
                            min_size=min_size,
                            max_size=max_size,
                        )

                for _key, value in kwargs.items():
                    if isinstance(value, np.ndarray):
                        guard.validate_numpy_buffer(
                            value,
                            expected_shape=shape,
                            expected_dtype=dtype,
                            min_size=min_size,
                            max_size=max_size,
                            require_contiguous=require_contiguous,
                            require_aligned=require_aligned,
                        )
                    elif MLX_AVAILABLE and hasattr(value, "shape"):
                        guard.validate_mlx_buffer(
                            value,
                            expected_shape=shape,
                            expected_dtype=str(dtype) if dtype else None,
                            min_size=min_size,
                            max_size=max_size,
                        )

            # Call original function
            result = func(*args, **kwargs)

            # Validate outputs
            if validate_outputs:
                if isinstance(result, np.ndarray):
                    guard.validate_numpy_buffer(
                        result, min_size=min_size, max_size=max_size
                    )
                elif MLX_AVAILABLE and hasattr(result, "shape"):
                    guard.validate_mlx_buffer(
                        result, min_size=min_size, max_size=max_size
                    )
                elif isinstance(result, list | tuple):
                    for item in result:
                        if isinstance(item, np.ndarray):
                            guard.validate_numpy_buffer(
                                item, min_size=min_size, max_size=max_size
                            )
                        elif MLX_AVAILABLE and hasattr(item, "shape"):
                            guard.validate_mlx_buffer(
                                item, min_size=min_size, max_size=max_size
                            )

            return result

        return wrapper

    return decorator


def async_buffer_guard(
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | str | None = None,
    min_size: int = 1,
    max_size: int | None = None,
    validate_inputs: bool = True,
    validate_outputs: bool = True,
):
    """Async version of buffer_guard decorator."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            guard = get_buffer_guard()

            # Validate inputs (same as sync version)
            if validate_inputs:
                for _i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray):
                        guard.validate_numpy_buffer(
                            arg,
                            expected_shape=shape,
                            expected_dtype=dtype,
                            min_size=min_size,
                            max_size=max_size,
                        )
                    elif MLX_AVAILABLE and hasattr(arg, "shape"):
                        guard.validate_mlx_buffer(
                            arg,
                            expected_shape=shape,
                            expected_dtype=str(dtype) if dtype else None,
                            min_size=min_size,
                            max_size=max_size,
                        )

            # Call original async function
            result = await func(*args, **kwargs)

            # Validate outputs (same as sync version)
            if validate_outputs:
                if isinstance(result, np.ndarray):
                    guard.validate_numpy_buffer(
                        result, min_size=min_size, max_size=max_size
                    )
                elif MLX_AVAILABLE and hasattr(result, "shape"):
                    guard.validate_mlx_buffer(
                        result, min_size=min_size, max_size=max_size
                    )
                elif isinstance(result, list | tuple):
                    for item in result:
                        if isinstance(item, np.ndarray):
                            guard.validate_numpy_buffer(
                                item, min_size=min_size, max_size=max_size
                            )
                        elif MLX_AVAILABLE and hasattr(item, "shape"):
                            guard.validate_mlx_buffer(
                                item, min_size=min_size, max_size=max_size
                            )

            return result

        return wrapper

    return decorator


def memory_usage_guard(max_memory_mb: float = 1024.0):
    """Decorator to monitor memory usage and prevent excessive allocation."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil

                process = psutil.Process()

                # Get memory before
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

                # Call function
                result = func(*args, **kwargs)

                # Get memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before

                # Check if we exceeded the limit
                if memory_diff > max_memory_mb:
                    logger.warning(
                        f"Function {func.__name__} used {memory_diff:.1f}MB "
                        f"(limit: {max_memory_mb:.1f}MB)"
                    )

                return result

            except ImportError:
                # If psutil not available, just run the function
                logger.warning("psutil not available for memory monitoring")
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience functions for common validation patterns
def validate_options_params(params_batch: np.ndarray) -> bool:
    """Validate options parameter batch format."""
    return validate_buffer_shape(
        params_batch, (params_batch.shape[0], 5)
    ) and validate_buffer_dtype(params_batch, np.float32)


def validate_embedding_array(embedding: np.ndarray, expected_dim: int = 1536) -> bool:
    """Validate embedding array format."""
    return validate_buffer_shape(embedding, (expected_dim,)) and validate_buffer_dtype(
        embedding, np.float32
    )


def validate_price_data(prices: np.ndarray) -> bool:
    """Validate price data array."""
    guard = get_buffer_guard()
    return guard.validate_numpy_buffer(
        prices, expected_dtype=np.float32, min_size=1, require_contiguous=True
    ) and np.all(
        prices >= 0
    )  # Prices should be non-negative


if __name__ == "__main__":
    # Test the buffer guard system
    logger.info("Testing buffer guard system...")

    guard = BufferGuard(strict_mode=False)

    # Test numpy array validation
    test_array = np.ones((100, 100), dtype=np.float32)
    success = guard.validate_numpy_buffer(
        test_array,
        expected_shape=(100, 100),
        expected_dtype=np.float32,
        min_size=1000,
        require_contiguous=True,
    )

    print(f"Numpy validation: {success}")

    # Test MLX array validation if available
    if MLX_AVAILABLE:
        mlx_array = mx.ones((50, 50), dtype=mx.float32)
        mx.eval(mlx_array)

        success = guard.validate_mlx_buffer(
            mlx_array, expected_shape=(50, 50), expected_dtype="float32", min_size=1000
        )

        print(f"MLX validation: {success}")

    # Print stats
    print(f"Validation stats: {guard.get_stats()}")
