#!/bin/bash
# Sync Codex changes back to main src/ directory

echo "🔄 SYNCING CODEX CHANGES TO SRC"
echo "==============================="

# Backup current src/
echo "📦 Creating backup of current src/..."
cp -r src/ src_backup_$(date +%Y%m%d_%H%M%S)/
echo "✅ Backup created"

# Copy optimized code back to src/
echo "📥 Copying optimized code to src/..."

# Copy main codebase
cp -r src/unity_wheel/* src/unity_wheel/

# Copy config and patterns
cp -r data_pipeline/config/* src/config/
cp -r data_pipeline/patterns/* src/patterns/

echo "✅ Code synchronized to src/"

# Show what changed
echo "📊 Changes summary:"
git status --porcelain src/ | head -10

# Validate by running a quick test
echo "🧪 Quick validation..."
python -c "from src.unity_wheel.strategy.wheel import WheelStrategy; print('✅ Import successful')" || echo "❌ Import failed"

echo ""
echo "🎯 READY TO COMMIT"
echo "Run: git add src/ && git commit -m 'Codex optimizations: [describe changes]'"
