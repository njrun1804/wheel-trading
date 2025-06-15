#!/bin/bash
if [[ -f "meta_system.pid" ]]; then
    PID=$(cat meta_system.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "🛑 Stopping meta system (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "⚠️ Force stopping..."
            kill -9 $PID
        fi
        rm -f meta_system.pid
        echo "✅ Meta system stopped"
    else
        echo "❌ Meta system not running"
        rm -f meta_system.pid
    fi
else
    echo "❌ Meta system not running"
fi
