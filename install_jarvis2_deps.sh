#!/bin/bash
# Install dependencies for Jarvis2 real code generation

echo "🔧 Installing Jarvis2 dependencies for real code generation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Essential dependencies for code understanding
echo ""
echo "📦 Installing code parsing and AST tools..."
pip install libcst>=1.4.0 ast-comments>=1.2.2 parso>=0.8.4

echo ""
echo "📦 Installing code formatting and analysis..."
pip install black>=24.8.0 radon>=6.0.1 pylint>=3.2.0

echo ""
echo "📦 Installing code embeddings..."
# Start with lightweight option
pip install sentence-transformers>=3.0.0

echo ""
echo "📦 Installing MLX language models for M4 Pro..."
pip install mlx-lm>=0.18.0

echo ""
echo "📦 Installing code generation helpers..."
pip install jinja2>=3.1.4 docstring-parser>=0.16

echo ""
echo "📦 Installing testing tools for validation..."
pip install hypothesis>=6.111.0

echo ""
echo "📦 Installing development tools..."
pip install rich>=13.7.1 tqdm>=4.66.4

# Optional: For more advanced code understanding
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Essential dependencies installed!"
echo ""
echo "Optional: For advanced features, you can also install:"
echo "  - transformers (for CodeBERT): pip install transformers>=4.44.0"
echo "  - tree-sitter (for multi-lang): pip install tree-sitter tree-sitter-python"
echo "  - Type checking: pip install mypy>=1.11.0"
echo ""
echo "Ready to implement real code generation in Jarvis2! 🚀"