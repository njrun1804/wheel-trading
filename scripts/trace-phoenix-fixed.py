#!/usr/bin/env python3
"""Phoenix trace MCP server - Fixed version."""

import json
import os
from datetime import datetime

import requests
from mcp.server import FastMCP

mcp = FastMCP("trace-phoenix")

# Phoenix configuration
PHOENIX_BASE_URL = os.environ.get("PHOENIX_BASE_URL", "http://localhost:6006")


@mcp.tool()
def phoenix_health_check() -> str:
    """Check if Phoenix server is running and healthy."""
    try:
        response = requests.get(f"{PHOENIX_BASE_URL}/", timeout=5)
        response.raise_for_status()
        return f"✅ Phoenix server is running at {PHOENIX_BASE_URL}"
    except Exception as e:
        return f"❌ Phoenix server error: {str(e)}"


@mcp.tool()
def send_trace_to_phoenix(
    operation_name: str, duration_ms: float, attributes: dict = None, status: str = "OK"
) -> str:
    """Send a trace to Phoenix server.

    Args:
        operation_name: Name of the operation
        duration_ms: Duration in milliseconds
        attributes: Additional attributes
        status: Status (OK, ERROR)
    """
    try:
        # Phoenix accepts OpenTelemetry format traces via HTTP
        trace_data = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": "wheel-trading"},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "mcp-trace"},
                            "spans": [
                                {
                                    "traceId": os.urandom(16).hex(),
                                    "spanId": os.urandom(8).hex(),
                                    "name": operation_name,
                                    "startTimeUnixNano": int(
                                        (
                                            datetime.now().timestamp()
                                            - duration_ms / 1000
                                        )
                                        * 1e9
                                    ),
                                    "endTimeUnixNano": int(
                                        datetime.now().timestamp() * 1e9
                                    ),
                                    "status": {
                                        "code": 1 if status == "OK" else 2,
                                        "message": status,
                                    },
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in (attributes or {}).items()
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        response = requests.post(
            f"{PHOENIX_BASE_URL}/v1/traces",
            json=trace_data,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        return f"✅ Trace sent to Phoenix: {operation_name} ({duration_ms}ms)"

    except Exception as e:
        return f"❌ Error sending trace: {str(e)}"


@mcp.tool()
def query_phoenix_graphql(query: str) -> str:
    """Execute a GraphQL query against Phoenix.

    Args:
        query: GraphQL query string
    """
    try:
        # Phoenix uses GraphQL for querying traces
        response = requests.post(
            f"{PHOENIX_BASE_URL}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        data = response.json()
        if "errors" in data:
            return f"GraphQL errors: {json.dumps(data['errors'], indent=2)}"

        return json.dumps(data.get("data", {}), indent=2)

    except Exception as e:
        return f"❌ GraphQL query error: {str(e)}"


@mcp.tool()
def get_recent_spans(limit: int = 10) -> str:
    """Get recent spans from Phoenix using GraphQL.

    Args:
        limit: Number of spans to retrieve
    """
    query = f"""
    {{
      spans(first: {limit}) {{
        edges {{
          node {{
            id
            name
            statusCode
            startTime
            latencyMs
            spanKind
            attributes {{
              key
              value
            }}
          }}
        }}
      }}
    }}
    """

    return query_phoenix_graphql(query)


@mcp.tool()
def get_trace_by_id(trace_id: str) -> str:
    """Get a specific trace by ID.

    Args:
        trace_id: The trace ID to retrieve
    """
    query = f"""
    {{
      spans(traceId: "{trace_id}") {{
        edges {{
          node {{
            id
            name
            statusCode
            startTime
            latencyMs
            spanKind
            parentId
            attributes {{
              key
              value
            }}
            events {{
              name
              timestamp
              attributes {{
                key
                value
              }}
            }}
          }}
        }}
      }}
    }}
    """

    return query_phoenix_graphql(query)


@mcp.tool()
def analyze_performance_by_operation() -> str:
    """Analyze performance grouped by operation name."""
    query = """
    {
      spans(first: 1000) {
        edges {
          node {
            name
            latencyMs
            statusCode
          }
        }
      }
    }
    """

    try:
        result = query_phoenix_graphql(query)
        data = json.loads(result)

        if "spans" not in data:
            return "No span data available"

        # Group by operation name
        operations = {}
        for edge in data["spans"]["edges"]:
            span = edge["node"]
            name = span["name"]
            latency = span.get("latencyMs", 0)
            status = span.get("statusCode", "UNSET")

            if name not in operations:
                operations[name] = {
                    "count": 0,
                    "total_ms": 0,
                    "errors": 0,
                    "latencies": [],
                }

            operations[name]["count"] += 1
            operations[name]["total_ms"] += latency
            operations[name]["latencies"].append(latency)
            if status == "ERROR":
                operations[name]["errors"] += 1

        # Format results
        output = "Performance Analysis by Operation:\n\n"
        sorted_ops = sorted(
            operations.items(), key=lambda x: x[1]["total_ms"], reverse=True
        )

        for op_name, stats in sorted_ops[:10]:
            avg_latency = (
                stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0
            )
            p95_latency = (
                sorted(stats["latencies"])[int(len(stats["latencies"]) * 0.95)]
                if stats["latencies"]
                else 0
            )
            error_rate = (
                (stats["errors"] / stats["count"] * 100) if stats["count"] > 0 else 0
            )

            output += f"📊 {op_name}\n"
            output += f"   Count: {stats['count']}\n"
            output += f"   Avg Latency: {avg_latency:.1f}ms\n"
            output += f"   P95 Latency: {p95_latency:.1f}ms\n"
            output += f"   Error Rate: {error_rate:.1f}%\n\n"

        return output

    except Exception as e:
        return f"❌ Error analyzing performance: {str(e)}"


@mcp.tool()
def send_test_traces() -> str:
    """Send test traces to verify Phoenix integration."""
    test_operations = [
        ("fetch_options_data", 150, {"symbol": "SPY", "expiry": "2024-01-19"}),
        ("calculate_greeks", 45, {"option_type": "call", "strike": 450}),
        ("risk_analysis", 230, {"portfolio_size": 10, "var_confidence": 0.95}),
        ("database_query", 12, {"query_type": "select", "table": "options"}),
        ("api_request", 340, {"endpoint": "/quote", "status_code": "200"}),
    ]

    results = []
    for op_name, duration, attrs in test_operations:
        result = send_trace_to_phoenix(op_name, duration, attrs)
        results.append(result)

    return "Test traces sent:\n" + "\n".join(results)


if __name__ == "__main__":
    # Run without stdout output to avoid breaking JSON-RPC
    # Phoenix base URL: {PHOENIX_BASE_URL}
    mcp.run()
