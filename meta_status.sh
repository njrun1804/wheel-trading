#!/bin/bash
if [[ -f "meta_system.pid" ]]; then
    PID=$(cat meta_system.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Meta system running (PID: $PID)"
        echo "📊 Check meta_evolution.db for activity logs"
    else
        echo "❌ Meta system not running"
        rm -f meta_system.pid
    fi
else
    echo "❌ Meta system not running"
fi
