#!/bin/bash
# Setup script for Claude Thought Stream Integration

echo "🚀 Setting up Claude Thought Stream Integration"
echo "==============================================="

# Check for required environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY environment variable not set"
    echo "   Please set your Anthropic API key:"
    echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✅ ANTHROPIC_API_KEY found"

# Install required packages
echo "📦 Installing required packages..."
pip install -r requirements_claude_integration.txt

# Check for M4 Pro optimizations
echo "🔍 Checking for M4 Pro optimizations..."
if python -c "import mlx.core" 2>/dev/null; then
    echo "✅ MLX found - M4 Pro hardware acceleration enabled"
else
    echo "⚠️  MLX not found - installing for M4 Pro optimization..."
    pip install mlx
fi

# Test the integration
echo "🧪 Testing Claude integration..."
python claude_stream_integration.py --message "Test the thought stream integration system"

echo ""
echo "🎯 Setup complete! You can now:"
echo "   • Test with sample requests: python claude_stream_integration.py --test"
echo "   • Process single message: python claude_stream_integration.py --message 'your message'"
echo "   • Monitor continuous stream in your applications"
echo ""
echo "🧠 The meta system will now capture Claude's real-time thinking patterns!"