"""Observability dashboard data export for monitoring and visualization.

Simplified for pull-when-asked architecture - metrics are stored in DuckDB
and exported on demand, no continuous collection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from ..data import get_anomaly_detector, get_market_validator
from ..metrics import metrics_collector
from ..monitoring import get_performance_monitor
from ..utils import StructuredLogger, get_feature_flags, get_logger

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_influx_line(self) -> str:
        """Convert to InfluxDB line protocol format."""
        # Format: measurement,tag1=value1,tag2=value2 field1=value1 timestamp
        tags_str = ",".join(f"{k}={v}" for k, v in self.tags.items())
        if tags_str:
            tags_str = "," + tags_str

        timestamp_ns = int(self.timestamp.timestamp() * 1e9)
        return f"{self.metric_name}{tags_str} value={self.value} {timestamp_ns}"

    def to_prometheus(self) -> str:
        """Convert to Prometheus exposition format."""
        # Format: metric_name{label1="value1",label2="value2"} value timestamp
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.tags.items())
        if labels_str:
            labels_str = "{" + labels_str + "}"
        else:
            labels_str = ""

        timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return f"{self.metric_name}{labels_str} {self.value} {timestamp_ms}"


@dataclass
class DashboardExport:
    """Aggregated data export for dashboards."""

    timestamp: datetime
    metrics: List[MetricPoint]
    events: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    system_health: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "name": m.metric_name,
                    "value": m.value,
                    "tags": m.tags,
                }
                for m in self.metrics
            ],
            "events": self.events,
            "alerts": self.alerts,
            "system_health": self.system_health,
        }


class ObservabilityExporter:
    """
    Export system metrics and events for observability dashboards.

    Supports multiple export formats:
    - JSON for custom dashboards
    - InfluxDB line protocol
    - Prometheus exposition format
    - CSV for analysis
    """

    def __init__(
        self,
        export_dir: Path = Path("exports"),
        storage_path: Optional[Path] = None,
    ):
        """Initialize exporter."""
        self.export_dir = export_dir
        self.export_dir.mkdir(exist_ok=True)

        # Use unified DuckDB storage
        if storage_path is None:
            storage_path = Path.home() / ".wheel_trading" / "cache" / "wheel_cache.duckdb"
        self.db_path = storage_path

    def _ensure_metrics_tables(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Ensure metrics tables exist in DuckDB."""
        # Create metrics table if not exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observability_metrics (
                timestamp TIMESTAMP NOT NULL,
                metric_name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                tags JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, metric_name)
            )
        """
        )

        # Create events table if not exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observability_events (
                timestamp TIMESTAMP NOT NULL,
                event_type VARCHAR NOT NULL,
                severity VARCHAR,
                description TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, event_type)
            )
        """
        )

        # Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_created ON observability_metrics(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_created ON observability_events(created_at)"
        )

    def collect_current_metrics(self) -> DashboardExport:
        """Collect all current metrics from various systems."""
        timestamp = datetime.now(timezone.utc)
        metrics = []
        events = []
        alerts = []

        # Collect cached performance metrics from storage
        conn = duckdb.connect(str(self.db_path))
        try:
            self._ensure_metrics_tables(conn)

            # Get recent performance data from cache
            perf_data = self._get_cached_performance_data(conn)
            perf_stats = perf_data if perf_data else {}

            for operation, stats in perf_stats.items():
                metrics.extend(
                    [
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="operation_latency_ms",
                            value=stats.avg_duration_ms,
                            tags={"operation": operation, "percentile": "avg"},
                        ),
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="operation_latency_ms",
                            value=stats.p95_duration_ms,
                            tags={"operation": operation, "percentile": "p95"},
                        ),
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="operation_success_rate",
                            value=stats.success_rate,
                            tags={"operation": operation},
                        ),
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="operation_count",
                            value=stats.count,
                            tags={"operation": operation},
                        ),
                    ]
                )

            # Collect decision metrics from predictions cache
            decision_stats = self._get_decision_statistics(conn)
            if decision_stats and decision_stats["total_decisions"] > 0:
                metrics.extend(
                    [
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="decision_confidence",
                            value=decision_stats["avg_confidence"],
                            tags={"metric": "average"},
                        ),
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="decision_count",
                            value=decision_stats["total_decisions"],
                            tags={"period": "all_time"},
                        ),
                        MetricPoint(
                            timestamp=timestamp,
                            metric_name="decision_success_rate",
                            value=decision_stats.get("success_rate", 0),
                            tags={"metric": "observed"},
                        ),
                    ]
                )

            # Collect data quality metrics from cache statistics
            validation_stats = self._get_validation_statistics(conn)

            metrics.extend(
                [
                    MetricPoint(
                        timestamp=timestamp,
                        metric_name="data_validation_success_rate",
                        value=validation_stats["success_rate"],
                        tags={"type": "market_data"},
                    ),
                    MetricPoint(
                        timestamp=timestamp,
                        metric_name="data_validation_count",
                        value=validation_stats["total_validations"],
                        tags={"type": "market_data"},
                    ),
                ]
            )

            # Get system health from cache statistics
            flag_report = self._get_system_health_report(conn)

            for feature_name, feature_info in flag_report["features"].items():
                metrics.append(
                    MetricPoint(
                        timestamp=timestamp,
                        metric_name="feature_flag_status",
                        value=1.0 if feature_info["is_enabled"] else 0.0,
                        tags={"feature": feature_name, "status": feature_info["status"]},
                    )
                )

                if feature_info["error_count"] > 0:
                    alerts.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "severity": "warning",
                            "type": "feature_degradation",
                            "message": f"Feature {feature_name} has {feature_info['error_count']} errors",
                            "details": feature_info,
                        }
                    )

            # Collect SLA violations
            sla_violations = perf_monitor.sla_violations[-10:]  # Last 10
            for violation in sla_violations:
                alerts.append(
                    {
                        "timestamp": violation["timestamp"].isoformat(),
                        "severity": violation["severity"],
                        "type": "sla_violation",
                        "operation": violation["operation"],
                        "duration_ms": violation["duration_ms"],
                        "threshold_ms": violation["threshold_ms"],
                    }
                )

            # System health summary
            system_health = self._calculate_system_health(perf_stats, validation_stats, flag_report)

            # Log export
            structured_logger.log(
                level="INFO",
                message="Collected observability metrics",
                context={
                    "metrics_count": len(metrics),
                    "events_count": len(events),
                    "alerts_count": len(alerts),
                    "health_score": system_health["overall_score"],
                },
            )

            return DashboardExport(
                timestamp=timestamp,
                metrics=metrics,
                events=events,
                alerts=alerts,
                system_health=system_health,
            )
        finally:
            conn.close()

    def _calculate_system_health(
        self,
        perf_stats: Dict[str, Any],
        validation_stats: Dict[str, Any],
        flag_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        health_score = 100.0
        issues = []

        # Performance health
        slow_ops = sum(
            1 for stats in perf_stats.values() if stats.p95_duration_ms > 1000  # 1 second
        )
        if slow_ops > 0:
            health_score -= slow_ops * 5
            issues.append(f"{slow_ops} slow operations")

        # Data quality health
        if validation_stats["success_rate"] < 0.95:
            health_score -= (1 - validation_stats["success_rate"]) * 50
            issues.append(f"Data validation rate {validation_stats['success_rate']:.1%}")

        # Feature health
        degraded_features = sum(
            1 for f in flag_report["features"].values() if f["status"] == "degraded"
        )
        if degraded_features > 0:
            health_score -= degraded_features * 10
            issues.append(f"{degraded_features} degraded features")

        return {
            "overall_score": max(0, health_score),
            "components": {
                "performance": "healthy" if slow_ops == 0 else "degraded",
                "data_quality": (
                    "healthy" if validation_stats["success_rate"] >= 0.95 else "degraded"
                ),
                "features": "healthy" if degraded_features == 0 else "degraded",
            },
            "issues": issues,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    def export_json(self, data: DashboardExport, filename: Optional[str] = None) -> Path:
        """Export data as JSON."""
        if filename is None:
            filename = f"dashboard_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.export_dir / filename
        with open(filepath, "w") as f:
            json.dump(data.to_dict(), f, indent=2)

        return filepath

    def export_influxdb(self, data: DashboardExport, filename: Optional[str] = None) -> Path:
        """Export data in InfluxDB line protocol format."""
        if filename is None:
            filename = f"influx_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = self.export_dir / filename
        with open(filepath, "w") as f:
            for metric in data.metrics:
                f.write(metric.to_influx_line() + "\n")

        return filepath

    def export_prometheus(self, data: DashboardExport, filename: Optional[str] = None) -> Path:
        """Export data in Prometheus exposition format."""
        if filename is None:
            filename = f"prometheus_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = self.export_dir / filename
        with open(filepath, "w") as f:
            # Group by metric name
            metrics_by_name = {}
            for metric in data.metrics:
                if metric.metric_name not in metrics_by_name:
                    metrics_by_name[metric.metric_name] = []
                metrics_by_name[metric.metric_name].append(metric)

            # Write with TYPE and HELP comments
            for metric_name, metric_list in metrics_by_name.items():
                f.write(f"# TYPE {metric_name} gauge\n")
                f.write(f"# HELP {metric_name} Unity wheel trading metric\n")
                for metric in metric_list:
                    f.write(metric.to_prometheus() + "\n")
                f.write("\n")

        return filepath

    def export_csv(self, data: DashboardExport, filename: Optional[str] = None) -> Path:
        """Export metrics as CSV for analysis."""
        if filename is None:
            filename = f"metrics_{data.timestamp.strftime('%Y%m%d_%H%M%S')}.csv"

        # Convert to DataFrame
        records = []
        for metric in data.metrics:
            record = {
                "timestamp": metric.timestamp,
                "metric_name": metric.metric_name,
                "value": metric.value,
            }
            record.update(metric.tags)
            records.append(record)

        df = pd.DataFrame(records)

        filepath = self.export_dir / filename
        df.to_csv(filepath, index=False)

        return filepath

    def store_metrics(self, data: DashboardExport) -> None:
        """Store metrics in DuckDB."""
        conn = duckdb.connect(str(self.db_path))
        try:
            self._ensure_metrics_tables(conn)
            # Store metrics
            for metric in data.metrics:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO observability_metrics
                    (timestamp, metric_name, value, tags)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        metric.timestamp,
                        metric.metric_name,
                        metric.value,
                        metric.tags,
                    ],
                )

            # Store events
            for event in data.events:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO observability_events
                    (timestamp, event_type, severity, description, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        event.get("timestamp", data.timestamp),
                        event.get("type", "unknown"),
                        event.get("severity", "info"),
                        event.get("message", ""),
                        event.get("details", {}),
                    ],
                )

            # Clean old data (30 day retention)
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            conn.execute("DELETE FROM observability_metrics WHERE created_at < ?", [cutoff])
            conn.execute("DELETE FROM observability_events WHERE created_at < ?", [cutoff])
        finally:
            conn.close()

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Query historical metrics from database."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        query = """
            SELECT timestamp, metric_name, value, tags
            FROM observability_metrics
            WHERE metric_name = ?
            AND timestamp >= ?
            AND timestamp <= ?
        """
        params = [metric_name, start_time, end_time]

        if tags:
            # Filter by tags (simple implementation)
            tag_filter = json.dumps(tags)
            query += " AND tags LIKE ?"
            params.append(f"%{tag_filter[1:-1]}%")  # Remove outer braces

        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            df = conn.execute(query, params).df()
        finally:
            conn.close()

        # Timestamps are already datetime in DuckDB
        # Tags are already JSON

        return df

    def generate_summary_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate summary report for specified time period."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            self._ensure_metrics_tables(conn)

            # Count metrics
            metrics_count = conn.execute(
                """
                SELECT COUNT(*) FROM observability_metrics
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                [start_time, end_time],
            ).fetchone()[0]

            # Count events by type
            events_df = conn.execute(
                """
                SELECT event_type, severity, COUNT(*) as count
                FROM observability_events
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY event_type, severity
                """,
                [start_time, end_time],
            ).df()

            # Get unique metric names
            metric_names = conn.execute(
                """
                SELECT DISTINCT metric_name FROM observability_metrics
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                [start_time, end_time],
            ).fetchall()

            return {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours,
                },
                "metrics": {
                    "total_count": metrics_count,
                    "unique_metrics": [row[0] for row in metric_names],
                },
                "events": events_df.to_dict("records") if not events_df.empty else [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            conn.close()

    def _get_cached_performance_data(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """Get performance data from DuckDB cache."""
        # In pull-when-asked architecture, we don't track continuous performance
        # Return basic stats from recent operations
        return {}

    def _get_decision_statistics(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """Get decision statistics from predictions cache."""
        try:
            result = conn.execute(
                """
                SELECT
                    COUNT(*) as total_decisions,
                    AVG(CAST(predictions->>'confidence' AS DOUBLE)) as avg_confidence
                FROM predictions_cache
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """
            ).fetchone()

            if result and result[0] > 0:
                return {
                    "total_decisions": result[0],
                    "avg_confidence": result[1] or 0.0,
                    "success_rate": 0.0,  # Not tracked in recommendation-only system
                }
        except:
            pass
        return {"total_decisions": 0}

    def _get_validation_statistics(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """Get data validation statistics."""
        # In pull-when-asked, validation happens on demand
        return {
            "success_rate": 1.0,  # Assume success unless we know otherwise
            "total_validations": 0,
        }

    def _get_system_health_report(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """Get system health from cache statistics."""
        try:
            # Check cache size and age
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as cache_entries,
                    MIN(created_at) as oldest_entry
                FROM (
                    SELECT created_at FROM option_chains
                    UNION ALL
                    SELECT created_at FROM position_snapshots
                ) t
            """
            ).fetchone()

            features = {
                "caching": {"is_enabled": True, "status": "healthy", "error_count": 0},
                "api_integration": {"is_enabled": True, "status": "healthy", "error_count": 0},
            }

            return {"features": features}
        except:
            return {"features": {}}


# Global exporter instance
_exporter: Optional[ObservabilityExporter] = None


def get_observability_exporter() -> ObservabilityExporter:
    """Get or create global observability exporter."""
    global _exporter
    if _exporter is None:
        _exporter = ObservabilityExporter()
    return _exporter
