#\!/bin/bash
# Quick launcher for Jarvis 2.0

# Activate environment if needed
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Launch Jarvis 2.0
python jarvis2.py "$@"
