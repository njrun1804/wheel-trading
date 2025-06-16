"""Immutable audit logger for all market data fetches.

Creates an append-only audit trail of every piece of market data used
in financial decisions. This provides forensic evidence if needed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from unity_wheel.utils import get_logger

logger = get_logger(__name__)


class DataAuditLogger:
    """Creates immutable audit trail of all market data access."""

    def __init__(self, audit_dir: str = "data_audit"):
        """Initialize audit logger with directory for audit files."""
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)

        # Create daily audit file
        today = datetime.now().strftime("%Y-%m-%d")
        self.audit_file = self.audit_dir / f"data_audit_{today}.jsonl"

        # Log initialization
        self._log_event(
            {
                "event": "audit_logger_initialized",
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid(),
            }
        )

    def log_data_fetch(
        self,
        source: str,
        symbol: str,
        data_type: str,
        data: dict[str, Any],
        request_params: dict[str, Any] | None = None,
    ) -> None:
        """Log a data fetch event with full details.

        Args:
            source: Data source (e.g., "databento", "schwab")
            symbol: Symbol fetched (e.g., "U")
            data_type: Type of data (e.g., "price", "option_chain")
            data: The actual data received
            request_params: Parameters used in the request
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "symbol": symbol,
            "data_type": data_type,
            "request_params": request_params or {},
            "data_summary": self._summarize_data(data),
            "data_hash": hash(json.dumps(data, sort_keys=True, default=str)),
            "pid": os.getpid(),
        }

        self._log_event(entry)

        # Also log to structured logger for real-time monitoring
        logger.info(
            f"Data fetched from {source}",
            extra={
                "source": source,
                "symbol": symbol,
                "data_type": data_type,
                "record_count": len(data) if isinstance(data, list | dict) else 1,
            },
        )

    def log_validation_result(
        self,
        data_type: str,
        passed: bool,
        details: dict[str, Any],
    ) -> None:
        """Log results of data validation checks."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "data_validation",
            "data_type": data_type,
            "passed": passed,
            "details": details,
            "pid": os.getpid(),
        }

        self._log_event(entry)

        if not passed:
            logger.warning(f"Data validation failed for {data_type}", extra=details)

    def log_calculation(
        self,
        function: str,
        inputs: dict[str, Any],
        output: Any,
        data_sources: list[str],
    ) -> None:
        """Log calculations that use market data."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "calculation",
            "function": function,
            "inputs": self._sanitize_inputs(inputs),
            "output": str(output),
            "data_sources": data_sources,
            "pid": os.getpid(),
        }

        self._log_event(entry)

    def _log_event(self, entry: dict[str, Any]) -> None:
        """Append entry to audit file (append-only for immutability)."""
        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()  # Ensure immediate write
                os.fsync(f.fileno())  # Force write to disk
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to write audit log: {e}")
            # Don't raise - audit failures shouldn't stop trading

    def _summarize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create summary of data without storing full dataset."""
        if isinstance(data, dict):
            return {
                "keys": list(data.keys()),
                "record_count": len(data),
                "sample_values": {k: str(v)[:50] for k, v in list(data.items())[:3]},
            }
        elif isinstance(data, list):
            return {
                "record_count": len(data),
                "first_record": str(data[0])[:100] if data else None,
            }
        else:
            return {"type": type(data).__name__, "value": str(data)[:100]}

    def _sanitize_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from inputs before logging."""
        sanitized = {}
        for key, value in inputs.items():
            if key.lower() in ["api_key", "secret", "password", "token"]:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, int | float | str | bool):
                sanitized[key] = value
            else:
                sanitized[key] = type(value).__name__
        return sanitized


# Global singleton
_audit_logger: DataAuditLogger | None = None


def get_audit_logger() -> DataAuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = DataAuditLogger()
    return _audit_logger
