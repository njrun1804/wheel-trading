# BOB Quick Start Guide

## 🚀 5-Minute Quick Start

Get up and running with BOB (Bolt Orchestrator Bootstrap) in under 5 minutes. This guide will have you solving complex problems with AI-powered multi-agent orchestration immediately.

## Prerequisites Check

```bash
# Verify you have the requirements
python --version          # Python 3.11+
uname -m                  # Should show "arm64" for M4 Pro
system_profiler SPHardwareDataType | grep "Chip"  # Should show M4 Pro
```

## Step 1: Installation (30 seconds)

```bash
# You're already in the wheel-trading directory, so just verify
ls bob_unified.py         # Should exist
ls config.yaml           # Should exist

# Install core dependencies (optional but recommended)
pip install pyyaml psutil rich numpy

# Make BOB executable
chmod +x bob_unified.py
```

## Step 2: Verify Installation (15 seconds)

```bash
# Test basic functionality
python bob_unified.py --version

# Expected output:
# BOB (Bolt Orchestrator Bootstrap) v1.0.0
# 8-agent hardware-accelerated problem solving
# M4 Pro optimized • 12 cores • 24GB unified memory
```

## Step 3: First Command (30 seconds)

```bash
# Your first BOB command - system health check
python bob_unified.py "analyze system health and performance"

# Expected behavior:
# ✅ Initializing BOB system...
# 🔍 Analyzing system components...
# 📊 Generating health report...
# ✅ System health: EXCELLENT
```

## Step 4: Interactive Mode (60 seconds)

```bash
# Start interactive mode for guided experience
python bob_unified.py --interactive
```

Interactive session:
```
🤖 BOB Interactive Mode - Hardware-Accelerated Problem Solving
Type 'help' for commands, 'workflow' for guided tasks, 'exit' to quit

bob> help
📚 BOB Help System
==================

Essential Commands:
• analyze <target>     - Analyze code, system, or trading data
• fix <problem>        - Fix issues and bugs
• optimize <component> - Optimize performance
• create <feature>     - Generate new code/features
• workflow            - Guided multi-step workflows
• status              - System status and metrics
• exit                - Exit interactive mode

bob> workflow
🔄 Available Workflows:
1. System Analysis & Health Check
2. Code Quality Review
3. Trading Performance Analysis
4. Development Task
5. Problem Solving

Select workflow (1-5): 1

🔍 System Analysis Workflow Started
✅ Step 1: Hardware detection complete
✅ Step 2: Component health verified  
✅ Step 3: Performance metrics collected
📊 Results: System running optimally at 95% efficiency

bob> exit
👋 BOB session complete. System ready for production use.
```

## Step 5: Essential Commands (90 seconds)

Try these essential commands to explore BOB's capabilities:

### Code Analysis
```bash
# Analyze the entire codebase structure
python bob_unified.py "analyze the wheel trading codebase architecture"

# Find potential performance issues
python bob_unified.py "identify performance bottlenecks in the system"

# Security review
python bob_unified.py "review authentication and security patterns"
```

### Trading Analysis
```bash
# Analyze current trading setup
python bob_unified.py "analyze Unity wheel trading strategy configuration"

# Risk assessment
python bob_unified.py "evaluate current portfolio risk exposure"

# Performance review
python bob_unified.py "analyze trading performance metrics"
```

### System Operations
```bash
# System status
python bob_unified.py status

# Performance report
python bob_unified.py "generate detailed performance report"

# Configuration review
python bob_unified.py "validate current system configuration"
```

## Step 6: Advanced Features (60 seconds)

### Natural Language Problem Solving
```bash
# Complex multi-step problems
python bob_unified.py "optimize the entire trading system for better performance and lower risk"

# Development tasks
python bob_unified.py "create comprehensive unit tests for the options pricing module"

# System improvements
python bob_unified.py "implement better error handling across all trading components"
```

### Workflow System
```bash
# Start development workflow
python bob_unified.py workflow development

# Start trading analysis workflow  
python bob_unified.py workflow trading

# Custom problem-solving workflow
python bob_unified.py workflow problem-solving
```

## Configuration Quick Setup

### Basic Configuration
The system works out-of-the-box, but you can customize it:

```yaml
# config.yaml - Basic customization
bob:
  # Adjust based on your hardware
  agents:
    count: 8                    # 8 agents optimal for M4 Pro
  
  hardware:
    cpu_cores: 12              # Use all 12 cores
    memory_limit_gb: 20        # Leave 4GB for system
  
  # Trading settings
  trading:
    symbol: "U"                # Unity stock
    max_position_size: 100000  # $100k max position
```

### Environment Variables (Optional)
```bash
# Performance tuning
export BOB_CPU_CORES=12
export BOB_MEMORY_LIMIT_GB=20
export BOB_LOG_LEVEL=INFO

# Trading settings
export BOB_TRADING_SYMBOL=U
export BOB_RISK_MAX_POSITION=100000
```

## Common Use Cases

### 1. Daily Development Tasks

```bash
# Morning health check
python bob_unified.py "perform comprehensive system health check"

# Code review
python bob_unified.py "review recent code changes for quality and security"

# Performance monitoring
python bob_unified.py "analyze system performance and identify optimizations"
```

### 2. Trading Operations

```bash
# Market analysis
python bob_unified.py "analyze Unity options market conditions"

# Position review
python bob_unified.py "evaluate current positions and suggest adjustments"

# Risk check
python bob_unified.py "validate all risk limits and exposure"
```

### 3. Development Workflows

```bash
# Feature development
python bob_unified.py "create a new risk management dashboard with real-time updates"

# Bug fixing
python bob_unified.py "find and fix all authentication timeout issues"

# Testing
python bob_unified.py "generate comprehensive test suite for wheel strategy"
```

### 4. System Maintenance

```bash
# Performance optimization
python bob_unified.py "optimize database queries and system performance"

# Cleanup
python bob_unified.py "clean up unused code and optimize imports"

# Documentation
python bob_unified.py "update documentation for all trading functions"
```

## Performance Tips

### Optimal Usage Patterns

1. **Use Interactive Mode**: For multiple related tasks
   ```bash
   python bob_unified.py --interactive
   ```

2. **Leverage Context**: Be specific in your requests
   ```bash
   # ✅ Good - specific and actionable
   python bob_unified.py "fix memory leak in agent pool"
   
   # ❌ Avoid - too vague
   python bob_unified.py "make it better"
   ```

3. **Use Workflows**: For complex multi-step tasks
   ```bash
   python bob_unified.py workflow development
   ```

### Hardware Optimization

The system automatically optimizes for your M4 Pro, but you can monitor performance:

```bash
# Check resource usage
python bob_unified.py "show current CPU, GPU, and memory utilization"

# Performance benchmarks
python bob_unified.py "run performance benchmarks and show results"

# Thermal monitoring
python bob_unified.py "check thermal status and performance throttling"
```

## Troubleshooting

### Common Issues

**Command Not Found**:
```bash
# Make sure you're in the right directory
cd /path/to/wheel-trading
python bob_unified.py --version
```

**Slow Performance**:
```bash
# Check system resources
python bob_unified.py status
python bob_unified.py "analyze system performance bottlenecks"
```

**Memory Issues**:
```bash
# Check memory usage
python bob_unified.py "analyze memory usage and suggest optimizations"

# Reduce memory if needed
export BOB_MEMORY_LIMIT_GB=16
```

### Getting Help

```bash
# Built-in help system
python bob_unified.py help
python bob_unified.py help commands
python bob_unified.py help troubleshooting

# System diagnostics
python bob_unified.py "diagnose any system issues"
python bob_unified.py "validate system configuration"
```

## Next Steps

### Explore Advanced Features

1. **API Integration**: Learn to use BOB programmatically
2. **Custom Workflows**: Create your own workflow definitions
3. **Configuration Tuning**: Optimize for your specific use cases
4. **Trading Integration**: Deep dive into trading system capabilities

### Learn More

- **BOB_UNIFIED_SYSTEM_README.md**: Complete system documentation
- **BOB_CLI_REFERENCE.md**: Comprehensive command reference
- **BOB_ARCHITECTURE_GUIDE.md**: Technical architecture details
- **config.yaml**: Configuration options and examples

### Join the Workflow

```bash
# Start with a guided workflow
python bob_unified.py --interactive
bob> workflow

# Try natural language commands
python bob_unified.py "help me learn BOB's capabilities"

# Explore trading features
python bob_unified.py "show me what trading analysis you can do"
```

## Quick Reference Card

```bash
# Essential commands
python bob_unified.py --version                    # Version info
python bob_unified.py status                       # System status
python bob_unified.py --interactive                # Interactive mode
python bob_unified.py help                         # Help system

# Analysis commands
python bob_unified.py "analyze [target]"           # Analyze anything
python bob_unified.py "fix [problem]"              # Fix issues
python bob_unified.py "optimize [component]"       # Optimize performance
python bob_unified.py "create [feature]"           # Generate new code

# Workflows
python bob_unified.py workflow                     # List workflows
python bob_unified.py workflow development         # Dev workflow
python bob_unified.py workflow trading             # Trading workflow

# System management
python bob_unified.py "system health check"        # Health check
python bob_unified.py "performance report"         # Performance metrics
python bob_unified.py "validate configuration"     # Config check
```

---

## 🎉 You're Ready!

Congratulations! You now have BOB running and can solve complex problems using natural language commands with 8-agent parallel processing. The system is hardware-optimized for your M4 Pro and ready for both development and trading tasks.

**What you can do now**:
- ✅ Analyze code and systems with natural language
- ✅ Fix bugs and optimize performance automatically  
- ✅ Generate new features and documentation
- ✅ Perform trading analysis and risk management
- ✅ Use interactive workflows for complex tasks
- ✅ Monitor system performance in real-time

**Time to completion**: ~5 minutes  
**System status**: 🚀 Production ready  
**Next step**: Try `python bob_unified.py "what can you help me with today?"`