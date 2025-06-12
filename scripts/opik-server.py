#!/usr/bin/env python3
"""Local Opik server for trace collection."""

from flask import Flask, request, jsonify
import json
from datetime import datetime
import threading
import time
from collections import deque

app = Flask(__name__)

# In-memory trace storage
traces = deque(maxlen=10000)
trace_lock = threading.Lock()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/v1/traces', methods=['POST'])
def receive_trace():
    """Receive traces from MCP server."""
    try:
        trace_data = request.json
        trace_data['received_at'] = datetime.now().isoformat()
        
        with trace_lock:
            traces.append(trace_data)
        
        return jsonify({"status": "success", "trace_id": trace_data.get('trace_id')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/v1/traces', methods=['GET'])
def get_traces():
    """Get recent traces."""
    try:
        limit = int(request.args.get('limit', 100))
        service = request.args.get('service')
        
        with trace_lock:
            recent_traces = list(traces)
        
        # Filter by service if provided
        if service:
            recent_traces = [t for t in recent_traces if t.get('service') == service]
        
        # Sort by timestamp and limit
        recent_traces.sort(key=lambda x: x.get('received_at', ''), reverse=True)
        recent_traces = recent_traces[:limit]
        
        return jsonify({
            "traces": recent_traces,
            "count": len(recent_traces),
            "total": len(traces)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/v1/traces/<trace_id>', methods=['GET'])
def get_trace(trace_id):
    """Get a specific trace by ID."""
    try:
        with trace_lock:
            for trace in traces:
                if trace.get('trace_id') == trace_id:
                    return jsonify(trace)
        
        return jsonify({"status": "not_found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Get trace statistics."""
    try:
        with trace_lock:
            recent_traces = list(traces)
        
        if not recent_traces:
            return jsonify({"status": "no_data"})
        
        # Calculate statistics
        operations = {}
        total_duration = 0
        error_count = 0
        
        for trace in recent_traces:
            op = trace.get('operation', 'unknown')
            duration = trace.get('duration_ms', 0)
            status = trace.get('status', 'OK')
            
            if op not in operations:
                operations[op] = {
                    'count': 0,
                    'total_duration': 0,
                    'errors': 0
                }
            
            operations[op]['count'] += 1
            operations[op]['total_duration'] += duration
            total_duration += duration
            
            if status != 'OK':
                operations[op]['errors'] += 1
                error_count += 1
        
        # Calculate averages
        for op in operations:
            count = operations[op]['count']
            operations[op]['avg_duration'] = operations[op]['total_duration'] / count if count > 0 else 0
            operations[op]['error_rate'] = operations[op]['errors'] / count if count > 0 else 0
        
        return jsonify({
            "total_traces": len(recent_traces),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "error_rate": error_count / len(recent_traces) if recent_traces else 0,
            "operations": operations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Opik trace server on http://localhost:5173")
    print("📊 Dashboard: http://localhost:5173/api/v1/stats")
    print("🔍 Traces: http://localhost:5173/api/v1/traces")
    app.run(host='0.0.0.0', port=5173, debug=False)