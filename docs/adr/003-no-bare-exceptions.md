# ADR-003: No Bare Exception Handlers

## Status
Accepted (January 2025)

## Context
Bare `except:` clauses catch all exceptions including SystemExit and KeyboardInterrupt, making debugging difficult and potentially hiding critical errors. In a financial system, we need precise error handling with proper logging and recovery strategies.

## Decision
All exception handlers must:
1. Catch specific exception types
2. Log the error with context
3. Return appropriate fallback values
4. Never use bare `except:` or `except Exception:`

## Consequences

### Positive
- Better error diagnostics
- Can't accidentally catch system exceptions
- Forces thinking about specific failure modes
- Improved debugging with error context

### Negative
- More verbose error handling code
- Need to identify all possible exceptions
- Potential for uncaught exceptions initially

### Neutral
- Requires code review to enforce
- May need static analysis tools

## Implementation

### ❌ Bad - What NOT to do
```python
try:
    result = complex_calculation()
except:  # NEVER DO THIS
    return None

try:
    result = risky_operation()
except Exception as e:  # AVOID THIS TOO
    logger.error(f"Failed: {e}")
    return default_value
```

### ✅ Good - Correct pattern
```python
try:
    result = complex_calculation()
except (ValueError, TypeError, KeyError) as e:
    logger.error(
        f"Calculation failed: {e}",
        extra={
            "function": "complex_calculation",
            "inputs": {"x": x, "y": y},
            "error_type": type(e).__name__,
        }
    )
    return CalculationResult(np.nan, 0.0, [f"Error: {str(e)}"])
except ZeroDivisionError:
    logger.warning("Division by zero, returning infinity")
    return CalculationResult(np.inf, 0.5, ["Division by zero"])
```

## Common Exception Types

### Data Processing
- `ValueError`: Invalid function arguments
- `TypeError`: Wrong type passed
- `KeyError`: Missing dictionary key
- `IndexError`: List index out of range
- `AttributeError`: Missing attribute

### Numerical
- `ZeroDivisionError`: Division by zero
- `OverflowError`: Number too large
- `FloatingPointError`: FP operation failed

### External Services
- `aiohttp.ClientError`: HTTP client errors
- `asyncio.TimeoutError`: Operation timeout
- `ConnectionError`: Network issues

### Our Custom Exceptions
- `DataValidationError`: Invalid market data
- `RiskLimitExceeded`: Position limit breached
- `InsufficientDataError`: Not enough data points

## Enforcement
- Pre-commit hook checks for bare except
- Code review checklist includes exception handling
- Static analysis with `bandit` security scanner

## References
- Python Exception Hierarchy: https://docs.python.org/3/library/exceptions.html
- Error handling patterns: docs/archive/patterns/error_handling.py
- Logging guide: docs/LOGGING_GUIDE.md
