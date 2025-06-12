#!/bin/bash

# Real-time resource monitoring

while true; do
    clear
    echo "=== M4 Pro Resource Monitor ==="
    echo ""
    
    # CPU usage by core type
    echo "CPU Usage:"
    echo "  Performance cores (0-7): $(ps aux | grep -E "claude|mcp|wheel" | awk '{sum+=$3} END {print sum}')%"
    echo "  Efficiency cores (8-11): $(ps aux | grep -E "docker|containerd" | awk '{sum+=$3} END {print sum}')%"
    echo ""
    
    # Memory usage
    echo "Memory Usage (24GB Total):"
    echo "  MCP/Claude: $(ps aux | grep -E "claude|mcp|wheel" | awk '{sum+=$6} END {printf "%.1f GB", sum/1024/1024}')"
    echo "  Docker: $(ps aux | grep -E "docker" | awk '{sum+=$6} END {printf "%.1f GB", sum/1024/1024}')"
    echo "  Available: $(vm_stat | grep "Pages free" | awk '{print $3*4096/1024/1024/1024 " GB"}')"
    echo ""
    
    # Top processes
    echo "Top Processes:"
    ps aux | sort -nrk 3,3 | head -5 | awk '{printf "  %-20s %5s%% %6.1fGB %s\n", $11, $3, $6/1024/1024, $2}'
    
    sleep 2
done
