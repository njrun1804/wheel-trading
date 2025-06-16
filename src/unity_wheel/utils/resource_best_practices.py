"""Resource management best practices and enforcement utilities.

This module provides utilities to enforce resource management best practices
throughout the codebase and prevent common resource leak patterns.
"""

from __future__ import annotations

import ast
import functools
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Type, Union

from .logging import get_logger

logger = get_logger(__name__)


class ResourceLeakWarning(UserWarning):
    """Warning for potential resource leaks."""
    pass


class ResourceBestPractices:
    """Enforces and validates resource management best practices."""
    
    # Patterns that indicate potential resource leaks
    RISKY_PATTERNS = {
        'open_without_with': r'open\s*\(',
        'subprocess_without_with': r'subprocess\.(run|Popen|check_)',
        'db_connect_without_with': r'\.(connect|create_connection)\s*\(',
        'async_without_await': r'async\s+def.*(?!await)',
        'missing_close': r'\.open\s*\(',
    }
    
    # Required context managers for certain operations
    REQUIRED_CONTEXT_MANAGERS = {
        'open': 'Use `with open()` instead of bare `open()`',
        'connect': 'Use connection context managers',
        'subprocess': 'Use subprocess context managers',
        'asyncio.create_subprocess': 'Properly await subprocess operations',
    }
    
    # Resource types that need explicit cleanup
    CLEANUP_REQUIRED = {
        'file', 'socket', 'process', 'thread', 'connection', 'pool', 'client'
    }


def requires_context_manager(func: Callable) -> Callable:
    """Decorator to warn when function is used outside context manager."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're in a context manager by inspecting the call stack
        frame = inspect.currentframe()
        try:
            # Look for __enter__ or __exit__ in the call stack
            in_context = False
            while frame:
                if '__enter__' in frame.f_code.co_names or '__exit__' in frame.f_code.co_names:
                    in_context = True
                    break
                frame = frame.f_back
            
            if not in_context:
                warnings.warn(
                    f"Function {func.__name__} should be used with a context manager to ensure proper resource cleanup",
                    ResourceLeakWarning,
                    stacklevel=2
                )
        finally:
            del frame
        
        return func(*args, **kwargs)
    
    return wrapper


def track_resource_lifecycle(resource_type: str):
    """Decorator to track resource lifecycle."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Creating {resource_type} resource via {func.__name__}")
            result = func(*args, **kwargs)
            
            # Register for tracking if the result has cleanup methods
            if hasattr(result, 'close') or hasattr(result, 'cleanup'):
                from .resource_manager import get_resource_tracker
                tracker = get_resource_tracker()
                tracker.register_resource(result, resource_type)
            
            return result
        
        return wrapper
    return decorator


def enforce_cleanup(cleanup_method: str = 'close'):
    """Decorator to enforce cleanup method calls."""
    
    def decorator(cls: Type) -> Type:
        original_del = getattr(cls, '__del__', None)
        
        def __del__(self):
            if hasattr(self, cleanup_method):
                try:
                    cleanup_func = getattr(self, cleanup_method)
                    if callable(cleanup_func):
                        cleanup_func()
                except Exception as e:
                    logger.warning(f"Error during {cls.__name__} cleanup: {e}")
            
            if original_del:
                original_del(self)
        
        cls.__del__ = __del__
        return cls
    
    return decorator


class ResourceValidator:
    """Validates code for resource management best practices."""
    
    def __init__(self):
        self.violations: List[str] = []
        self.suggestions: List[str] = []
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate a Python file for resource management issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            self._analyze_ast(tree, file_path)
            
            return len(self.violations) == 0
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyze AST for resource management issues."""
        for node in ast.walk(tree):
            self._check_open_calls(node, file_path)
            self._check_subprocess_calls(node, file_path)
            self._check_database_connections(node, file_path)
            self._check_context_managers(node, file_path)
            self._check_async_patterns(node, file_path)
    
    def _check_open_calls(self, node: ast.AST, file_path: Path):
        """Check for file open calls without context managers."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'open':
                # Check if this open call is inside a with statement
                parent = getattr(node, 'parent', None)
                in_with = False
                while parent:
                    if isinstance(parent, ast.withitem):
                        in_with = True
                        break
                    parent = getattr(parent, 'parent', None)
                
                if not in_with:
                    self.violations.append(
                        f"{file_path}:{getattr(node, 'lineno', '?')}: "
                        "open() call without context manager"
                    )
                    self.suggestions.append(
                        "Use 'with open() as f:' to ensure file is properly closed"
                    )
    
    def _check_subprocess_calls(self, node: ast.AST, file_path: Path):
        """Check for subprocess calls without proper handling."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'subprocess' and
                    node.func.attr in ['run', 'Popen', 'check_call', 'check_output']):
                    
                    # Check for timeout parameter
                    has_timeout = any(
                        kw.arg == 'timeout' for kw in node.keywords
                    )
                    
                    if not has_timeout:
                        self.violations.append(
                            f"{file_path}:{getattr(node, 'lineno', '?')}: "
                            "subprocess call without timeout"
                        )
                        self.suggestions.append(
                            "Add timeout parameter to subprocess calls to prevent hanging"
                        )
    
    def _check_database_connections(self, node: ast.AST, file_path: Path):
        """Check for database connections without proper handling."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ['connect', 'create_connection']:
                    # Check if in context manager
                    parent = getattr(node, 'parent', None)
                    in_with = False
                    while parent:
                        if isinstance(parent, ast.withitem):
                            in_with = True
                            break
                        parent = getattr(parent, 'parent', None)
                    
                    if not in_with:
                        self.violations.append(
                            f"{file_path}:{getattr(node, 'lineno', '?')}: "
                            "database connection without context manager"
                        )
                        self.suggestions.append(
                            "Use connection context managers to ensure proper cleanup"
                        )
    
    def _check_context_managers(self, node: ast.AST, file_path: Path):
        """Check for proper context manager usage."""
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    # Good - using context manager
                    continue
                else:
                    self.violations.append(
                        f"{file_path}:{getattr(node, 'lineno', '?')}: "
                        "Invalid context manager usage"
                    )
    
    def _check_async_patterns(self, node: ast.AST, file_path: Path):
        """Check for proper async/await patterns."""
        if isinstance(node, ast.AsyncFunctionDef):
            # Check if async function properly awaits async operations
            has_await = False
            for child in ast.walk(node):
                if isinstance(child, ast.Await):
                    has_await = True
                    break
            
            if not has_await:
                self.suggestions.append(
                    f"{file_path}:{getattr(node, 'lineno', '?')}: "
                    "Async function without await - consider if async is necessary"
                )
    
    def get_report(self) -> str:
        """Get validation report."""
        report = []
        
        if self.violations:
            report.append("VIOLATIONS:")
            for violation in self.violations:
                report.append(f"  - {violation}")
            report.append("")
        
        if self.suggestions:
            report.append("SUGGESTIONS:")
            for suggestion in self.suggestions:
                report.append(f"  - {suggestion}")
            report.append("")
        
        if not self.violations and not self.suggestions:
            report.append("No resource management issues found.")
        
        return "\n".join(report)


class ResourceBestPracticesEnforcer:
    """Enforces resource management best practices at runtime."""
    
    def __init__(self):
        self.enabled = True
        self.strict_mode = False
        self.violations_count = 0
        self.max_violations = 10
    
    def enable_strict_mode(self):
        """Enable strict mode - raise exceptions for violations."""
        self.strict_mode = True
    
    def disable_strict_mode(self):
        """Disable strict mode - only warn for violations."""
        self.strict_mode = False
    
    def check_resource_usage(self, obj: Any, operation: str):
        """Check if resource usage follows best practices."""
        if not self.enabled:
            return
        
        # Check for common patterns
        if hasattr(obj, 'close') and operation == 'create':
            self._warn_closeable_resource(obj)
        
        if hasattr(obj, 'connect') and operation == 'connect':
            self._warn_connection_resource(obj)
    
    def _warn_closeable_resource(self, obj: Any):
        """Warn about closeable resources."""
        message = f"Closeable resource {type(obj).__name__} created - ensure proper cleanup"
        self._handle_violation(message)
    
    def _warn_connection_resource(self, obj: Any):
        """Warn about connection resources."""
        message = f"Connection resource {type(obj).__name__} created - use context manager"
        self._handle_violation(message)
    
    def _handle_violation(self, message: str):
        """Handle a best practices violation."""
        self.violations_count += 1
        
        if self.strict_mode:
            raise ResourceLeakWarning(message)
        else:
            warnings.warn(message, ResourceLeakWarning, stacklevel=3)
        
        logger.warning(f"Resource best practices violation: {message}")
        
        if self.violations_count >= self.max_violations:
            logger.error(f"Too many resource violations ({self.violations_count}), "
                        "consider enabling strict mode or reviewing code")


# Global enforcer instance
_enforcer: Optional[ResourceBestPracticesEnforcer] = None


def get_enforcer() -> ResourceBestPracticesEnforcer:
    """Get or create global best practices enforcer."""
    global _enforcer
    if _enforcer is None:
        _enforcer = ResourceBestPracticesEnforcer()
    return _enforcer


# Convenience functions
def validate_codebase(root_path: Union[str, Path]) -> bool:
    """Validate entire codebase for resource management issues."""
    root_path = Path(root_path)
    validator = ResourceValidator()
    
    all_valid = True
    python_files = list(root_path.rglob("*.py"))
    
    logger.info(f"Validating {len(python_files)} Python files...")
    
    for py_file in python_files:
        # Skip certain directories
        if any(part.startswith('.') for part in py_file.parts):
            continue
        if 'venv' in py_file.parts or '__pycache__' in py_file.parts:
            continue
        
        file_valid = validator.validate_file(py_file)
        if not file_valid:
            all_valid = False
    
    # Print report
    report = validator.get_report()
    if report.strip() != "No resource management issues found.":
        logger.warning(f"Resource validation report:\n{report}")
    else:
        logger.info("All files passed resource management validation")
    
    return all_valid


def enable_runtime_enforcement(strict: bool = False):
    """Enable runtime enforcement of best practices."""
    enforcer = get_enforcer()
    enforcer.enabled = True
    if strict:
        enforcer.enable_strict_mode()
    logger.info(f"Enabled resource best practices enforcement (strict: {strict})")


def disable_runtime_enforcement():
    """Disable runtime enforcement."""
    enforcer = get_enforcer()
    enforcer.enabled = False
    logger.info("Disabled resource best practices enforcement")


# Example best practices documentation
BEST_PRACTICES_GUIDE = """
Resource Management Best Practices
=================================

1. File Operations:
   ✅ with open('file.txt') as f:
   ❌ f = open('file.txt')

2. Database Connections:
   ✅ with connection_pool.get_connection() as conn:
   ❌ conn = connection_pool.get_connection()

3. Subprocess Operations:
   ✅ subprocess.run(['cmd'], timeout=30)
   ❌ subprocess.run(['cmd'])  # No timeout

4. Async Operations:
   ✅ async with aiofiles.open('file') as f:
   ❌ f = await aiofiles.open('file')

5. Resource Cleanup:
   ✅ try: ... finally: resource.close()
   ❌ resource.close()  # Not in finally

6. Context Managers:
   ✅ Implement __enter__ and __exit__
   ❌ Manual resource management

7. Error Handling:
   ✅ Proper exception handling in cleanup
   ❌ Ignoring cleanup exceptions

8. Timeouts:
   ✅ All network/subprocess ops have timeouts
   ❌ Blocking operations without timeouts

9. Resource Limits:
   ✅ Monitor and enforce resource limits
   ❌ Unlimited resource consumption

10. Async Cleanup:
    ✅ Proper async context managers
    ❌ Mixing sync/async without care
"""


def print_best_practices():
    """Print the best practices guide."""
    print(BEST_PRACTICES_GUIDE)