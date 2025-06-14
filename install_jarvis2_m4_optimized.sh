#!/bin/bash
# M4 Pro optimized installation for Jarvis2

echo "🚀 Installing M4 Pro optimized dependencies for Jarvis2..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Core parsing - all native/fast on M4
echo ""
echo "📦 Installing code parsing (M4 native speed)..."
pip install libcst>=1.4.0 ast-comments>=1.2.2 parso>=0.8.4

# Code tools - pure Python, fast on ARM
echo ""
echo "📦 Installing code analysis tools..."
pip install black>=24.8.0 radon>=6.0.1

# MLX for Apple Silicon ML
echo ""
echo "📦 Installing MLX language models (Metal accelerated)..."
pip install mlx-lm>=0.18.0

# Lightweight embeddings using numpy/sklearn (M4 optimized)
echo ""
echo "📦 Installing lightweight embeddings..."
pip install scikit-learn>=1.5.0  # Already in requirements.txt

# Templates and utils
echo ""
echo "📦 Installing generation helpers..."
pip install jinja2>=3.1.4 docstring-parser>=0.16

# Dev tools
echo ""
echo "📦 Installing development tools..."
pip install rich>=13.7.1 tqdm>=4.66.4

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ M4 Pro optimized stack installed!"
echo ""
echo "Using:"
echo "  • MLX for ML operations (Metal GPU)"
echo "  • Native Python parsers (fast on ARM)"  
echo "  • Scikit-learn for embeddings (uses Accelerate framework)"
echo "  • No heavy PyTorch transformers needed!"
echo ""
echo "Your M4 Pro is ready for efficient code generation! 🏎️"