"""
MLX no_grad Context Manager Fix

This module provides a drop-in replacement for PyTorch's no_grad context manager
when using MLX. MLX uses lazy evaluation by default, so gradients are not computed
unless explicitly requested, making no_grad unnecessary but this provides compatibility.
"""
from collections.abc import Generator
from contextlib import contextmanager

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@contextmanager
def mlx_no_grad() -> Generator[None, None, None]:
    """
    MLX-compatible no_grad context manager.

    In MLX, gradients are not computed by default (lazy evaluation),
    so this is essentially a no-op but provides API compatibility.
    """
    # MLX doesn't need gradient disable - it's lazy by default
    yield


def patch_mlx_no_grad():
    """
    Monkey patch mlx.core to add no_grad compatibility.
    This allows existing PyTorch-style code to work with MLX.
    """
    if MLX_AVAILABLE:
        # Add no_grad to mlx.core module
        mx.no_grad = mlx_no_grad

        # Also add it as a property for compatibility
        if not hasattr(mx, "no_grad"):
            mx.no_grad = mlx_no_grad


def safe_no_grad_context():
    """
    Returns appropriate no_grad context for available backend.
    """
    if MLX_AVAILABLE:
        return mlx_no_grad()
    else:
        # Fallback for when MLX is not available
        try:
            import torch

            return torch.no_grad()
        except ImportError:
            # If neither MLX nor PyTorch available, return dummy context
            return mlx_no_grad()


# Auto-patch when module is imported
if MLX_AVAILABLE and not hasattr(mx, "no_grad"):
    patch_mlx_no_grad()


# Provide convenience functions for common patterns
def evaluate_without_grad(*arrays):
    """
    Evaluate MLX arrays without gradient computation.
    In MLX, this is the default behavior.
    """
    if MLX_AVAILABLE and arrays:
        mx.eval(*arrays)
    return arrays


def disable_gradients(func):
    """
    Decorator to disable gradients for a function.
    For MLX compatibility.
    """

    def wrapper(*args, **kwargs):
        with safe_no_grad_context():
            return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    # Test the fix
    print("Testing MLX no_grad compatibility fix...")

    if MLX_AVAILABLE:
        # Test that no_grad is now available
        try:
            with mx.no_grad():
                test_array = mx.array([1.0, 2.0, 3.0])
                result = mx.sum(test_array)
                mx.eval(result)
            print("✅ mx.no_grad() context manager working")
        except AttributeError as e:
            print(f"❌ mx.no_grad() still not working: {e}")

        # Test the safe context
        try:
            with safe_no_grad_context():
                test_array = mx.array([1.0, 2.0, 3.0])
                result = mx.sum(test_array)
                mx.eval(result)
            print("✅ safe_no_grad_context() working")
        except Exception as e:
            print(f"❌ safe_no_grad_context() failed: {e}")
    else:
        print("ℹ️ MLX not available - fix would work when MLX is installed")
