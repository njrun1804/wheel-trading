#!/bin/bash
# Orchestrator cleanup script

echo "Unity Wheel Orchestrator Cleanup"
echo "================================"

# Create directories if they don't exist
echo "Creating directories..."
mkdir -p tests/orchestrator
mkdir -p logs/archive

# Move test files
echo -e "\nMoving test files to tests/orchestrator/..."
for f in test_*.py; do
    if [ -f "$f" ]; then
        echo "  Moving $f"
        mv "$f" tests/orchestrator/
    fi
done

# Move log files
echo -e "\nArchiving log files..."
for f in *.log; do
    if [ -f "$f" ]; then
        echo "  Archiving $f"
        mv "$f" logs/archive/
    fi
done

# List backup directories
echo -e "\nBackup directories found:"
find . -type d -name "*backup*" | grep -v __pycache__ | while read dir; do
    echo "  $dir"
done

# List duplicate config files
echo -e "\nPotential duplicate configs:"
find . -name "*.yaml" -o -name "*.yml" | grep -E "(config|unified)" | grep -v __pycache__ | sort

# List untracked files
echo -e "\nUntracked files (consider adding to .gitignore or removing):"
git status --porcelain | grep "^??" | awk '{print "  " $2}'

echo -e "\nCleanup recommendations completed!"
echo "To remove backup directories, run:"
echo "  rm -rf src/unity_wheel/orchestrator_backup_20250613_135149"

echo -e "\nTo test the orchestrator, run:"
echo "  python orchestrate.py 'test basic functionality'"