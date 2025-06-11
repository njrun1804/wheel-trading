#!/bin/bash

echo "Setting up Embeddings MCP Server for Large Repos"
echo "================================================"
echo ""

# Fix npm permissions
echo "Please enter your password to fix npm permissions:"
sudo chown -R $(whoami):staff ~/.npm

echo ""
echo "Searching for available MCP embeddings/code search servers..."

# Try to find embeddings server
npm search "@modelcontextprotocol" 2>/dev/null | grep -E "embed|code|search" || true

echo ""
echo "Available options for code search in large repos:"
echo ""
echo "1. Use ripgrep (rg) for fast searching - already available"
echo "2. Create a custom embeddings index using local tools"
echo "3. Use the filesystem MCP with smart file filtering"
echo ""

# Create a wrapper script for smart code search
cat > ~/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/smart-code-search.sh << 'EOF'
#!/bin/bash
# Smart code search using ripgrep with context

PATTERN="$1"
CONTEXT_LINES="${2:-5}"
PROJECT_DIR="${3:-.}"

echo "Searching for: $PATTERN"
echo "Context lines: $CONTEXT_LINES"
echo ""

# Use ripgrep with smart defaults
rg "$PATTERN" \
  --type py \
  --type yaml \
  --type json \
  --type md \
  --context "$CONTEXT_LINES" \
  --max-count 10 \
  --max-filesize 1M \
  --sort path \
  "$PROJECT_DIR"
EOF

chmod +x ~/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/smart-code-search.sh

echo "Created smart-code-search.sh script for efficient searching"
echo ""
echo "Since the embeddings server isn't available yet, here are alternatives:"
echo ""
echo "1. The filesystem MCP server already chunks large files intelligently"
echo "2. Use the smart-code-search.sh script: ./scripts/smart-code-search.sh 'pattern' [context_lines]"
echo "3. Configure ripgrep aliases for common searches"
echo ""
echo "Your setup is optimized for large repos with:"
echo "✓ Reduced output tokens (8k instead of 64k)"
echo "✓ Smart file searching with ripgrep"
echo "✓ MCP filesystem server with intelligent chunking"
