#!/bin/bash

# Optimized Claude Code launcher for M4 Pro with 24GB RAM
# Prevents RangeError: Invalid string length

# Node.js Memory Optimizations (18.6GB of 24GB available)
export NODE_OPTIONS="--max-old-space-size=19200 --max-semi-space-size=512 --max-buffer-size=16777216"

# Claude Code Buffer Limits
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=256000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000
export CLAUDE_CODE_THINKING_BUDGET_TOKENS=200000
export CLAUDE_CODE_STREAMING_ENABLED=true
export CLAUDE_CODE_CHUNK_SIZE=8192

# System Resource Limits
ulimit -n 16384    # File descriptors
ulimit -u 4096     # User processes

# Hardware Acceleration
export UV_THREADPOOL_SIZE=12  # Match M4 Pro cores
export PYTHONUNBUFFERED=1

echo "Starting Claude Code with M4 Pro optimizations..."
echo "Node.js heap: 19.2GB | Streaming: enabled | Chunk size: 8KB"

claude "$@"