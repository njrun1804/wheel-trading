#!/bin/bash

# Set CPU affinity for processes

# Performance cores: 0-7 (M4 Pro)
# Efficiency cores: 8-11 (M4 Pro)

set_performance_affinity() {
    local pid=$1
    # Use performance cores (0-7)
    taskpolicy -c background -s 0 -p $pid 2>/dev/null || true
}

set_efficiency_affinity() {
    local pid=$1
    # Use efficiency cores (8-11)
    taskpolicy -c background -s 1 -p $pid 2>/dev/null || true
}

echo "Setting CPU affinity..."

# MCP/Claude processes → Performance cores
for pid in $(pgrep -f "claude|mcp|wheel.*py"); do
    set_performance_affinity $pid
    echo "  Performance → PID $pid ($(ps -p $pid -o comm=))"
done

# Docker → Efficiency cores
for pid in $(pgrep -f "docker|containerd"); do
    set_efficiency_affinity $pid
    echo "  Efficiency → PID $pid ($(ps -p $pid -o comm=))"
done

echo "Done!"
