#!/usr/bin/env python3
"""Simple Phoenix-compatible trace server."""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
import json
from datetime import datetime
import threading
from collections import deque, defaultdict
import os
import time

app = Flask(__name__)

# In-memory storage
traces = deque(maxlen=50000)
spans_by_trace = defaultdict(list)
trace_lock = threading.Lock()

# Simple HTML dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phoenix Trace Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .stats { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        .trace { border: 1px solid #ddd; padding: 10px; margin: 5px 0; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>🔥 Phoenix Trace Server</h1>
    <div class="stats">
        <h2>Statistics</h2>
        <p>Total Traces: <span id="total">0</span></p>
        <p>Error Rate: <span id="errorRate">0%</span></p>
        <p>Avg Latency: <span id="avgLatency">0ms</span></p>
    </div>
    <h2>Recent Traces</h2>
    <div id="traces"></div>
    <script>
        function refresh() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('total').textContent = data.total_traces;
                    document.getElementById('errorRate').textContent = (data.error_rate * 100).toFixed(1) + '%';
                    document.getElementById('avgLatency').textContent = data.avg_latency.toFixed(1) + 'ms';
                });
            
            fetch('/api/traces?limit=10')
                .then(r => r.json())
                .then(data => {
                    const tracesDiv = document.getElementById('traces');
                    tracesDiv.innerHTML = data.traces.map(trace => 
                        `<div class="trace ${trace.status === 'ERROR' ? 'error' : 'success'}">
                            <strong>${trace.operation || 'Unknown'}</strong> - 
                            ${trace.duration_ms}ms - 
                            ${trace.status} - 
                            ${new Date(trace.timestamp).toLocaleTimeString()}
                        </div>`
                    ).join('');
                });
        }
        setInterval(refresh, 2000);
        refresh();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(DASHBOARD_HTML)

@app.route('/health')
@app.route('/healthz')
def health():
    return jsonify({"status": "healthy", "server": "phoenix-simple"})

@app.route('/v1/traces', methods=['POST'])
def receive_traces():
    """Receive OpenTelemetry format traces."""
    try:
        data = request.json
        timestamp = datetime.now().isoformat()
        
        # Process OpenTelemetry format
        for resource_span in data.get('resourceSpans', []):
            service_name = "unknown"
            for attr in resource_span.get('resource', {}).get('attributes', []):
                if attr['key'] == 'service.name':
                    service_name = attr['value'].get('stringValue', 'unknown')
                    break
            
            for scope_span in resource_span.get('scopeSpans', []):
                for span in scope_span.get('spans', []):
                    trace_id = span.get('traceId')
                    span_id = span.get('spanId')
                    
                    # Calculate duration
                    start_ns = span.get('startTimeUnixNano', 0)
                    end_ns = span.get('endTimeUnixNano', 0)
                    duration_ms = (end_ns - start_ns) / 1_000_000
                    
                    # Extract attributes
                    attributes = {}
                    for attr in span.get('attributes', []):
                        key = attr.get('key')
                        value = attr.get('value', {})
                        attributes[key] = value.get('stringValue') or value.get('intValue') or value.get('doubleValue')
                    
                    trace_data = {
                        'trace_id': trace_id,
                        'span_id': span_id,
                        'operation': span.get('name', 'unknown'),
                        'service': service_name,
                        'duration_ms': duration_ms,
                        'status': 'ERROR' if span.get('status', {}).get('code', 1) != 1 else 'OK',
                        'timestamp': timestamp,
                        'attributes': attributes
                    }
                    
                    with trace_lock:
                        traces.append(trace_data)
                        spans_by_trace[trace_id].append(trace_data)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/traces', methods=['GET'])
def get_traces():
    """Get recent traces."""
    limit = int(request.args.get('limit', 100))
    
    with trace_lock:
        recent_traces = list(traces)[-limit:]
    
    return jsonify({
        "traces": recent_traces,
        "count": len(recent_traces)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get trace statistics."""
    with trace_lock:
        all_traces = list(traces)
    
    if not all_traces:
        return jsonify({
            "total_traces": 0,
            "error_rate": 0,
            "avg_latency": 0,
            "operations": {}
        })
    
    total = len(all_traces)
    errors = sum(1 for t in all_traces if t.get('status') == 'ERROR')
    total_latency = sum(t.get('duration_ms', 0) for t in all_traces)
    
    # Group by operation
    operations = defaultdict(lambda: {"count": 0, "total_ms": 0, "errors": 0})
    for trace in all_traces:
        op = trace.get('operation', 'unknown')
        operations[op]["count"] += 1
        operations[op]["total_ms"] += trace.get('duration_ms', 0)
        if trace.get('status') == 'ERROR':
            operations[op]["errors"] += 1
    
    return jsonify({
        "total_traces": total,
        "error_rate": errors / total if total > 0 else 0,
        "avg_latency": total_latency / total if total > 0 else 0,
        "operations": dict(operations)
    })

@app.route('/graphql', methods=['POST'])
def graphql():
    """Basic GraphQL endpoint for compatibility."""
    try:
        query = request.json.get('query', '')
        
        # Simple query parsing
        if 'spans' in query:
            limit = 100
            if 'first:' in query:
                try:
                    limit = int(query.split('first:')[1].split(')')[0].strip())
                except:
                    pass
            
            with trace_lock:
                recent_spans = list(traces)[-limit:]
            
            edges = []
            for span in recent_spans:
                edges.append({
                    "node": {
                        "id": span.get('span_id', ''),
                        "name": span.get('operation', ''),
                        "statusCode": span.get('status', 'OK'),
                        "startTime": span.get('timestamp', ''),
                        "latencyMs": span.get('duration_ms', 0),
                        "spanKind": "CLIENT",
                        "attributes": [
                            {"key": k, "value": str(v)}
                            for k, v in span.get('attributes', {}).items()
                        ]
                    }
                })
            
            return jsonify({
                "data": {
                    "spans": {
                        "edges": edges
                    }
                }
            })
        
        return jsonify({"data": {}})
    except Exception as e:
        return jsonify({"errors": [{"message": str(e)}]}), 400

if __name__ == '__main__':
    print("🔥 Starting Phoenix-compatible trace server")
    print("📊 Dashboard: http://localhost:6006")
    print("📨 Traces endpoint: http://localhost:6006/v1/traces")
    print("🔍 GraphQL: http://localhost:6006/graphql")
    app.run(host='0.0.0.0', port=6006, debug=False)