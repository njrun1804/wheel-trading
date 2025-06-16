# BOB CLI Reference Guide

## Overview

The BOB CLI provides a unified natural language interface for all system operations, replacing dozens of legacy scripts with a single, intelligent command interface. The CLI intelligently routes queries to the appropriate system components (Einstein search, BOLT agents, or direct execution) based on query complexity and context.

## Command Structure

```bash
./bob [OPTIONS] [COMMAND] [QUERY]
```

## Basic Usage Patterns

### Natural Language Commands

```bash
# Simple form - direct natural language
./bob "fix authentication issues in storage.py"
./bob "optimize wheel trading parameters"
./bob "analyze system performance"

# Explicit command form
./bob solve "complex multi-step problem"
./bob analyze "component or system"
./bob optimize "performance target"
```

### Interactive Mode

```bash
# Start interactive session
./bob --interactive
./bob -i

# Interactive with specific context
./bob --interactive --context="trading"
./bob -i -c trading
```

## Command Categories

### 1. Analysis Commands

#### Code Analysis
```bash
# General code analysis
./bob "analyze error handling patterns across the codebase"
./bob "find performance bottlenecks in wheel trading system"
./bob "review security vulnerabilities in authentication code"

# Specific file/module analysis
./bob analyze src/unity_wheel/strategy/wheel.py
./bob "analyze all risk management modules"
./bob "review test coverage in trading components"

# Architecture analysis
./bob "analyze coupling between components"
./bob "identify circular dependencies"
./bob "review API consistency across modules"
```

#### Trading Analysis
```bash
# Position analysis
./bob "analyze current Unity wheel positions"
./bob "review portfolio risk exposure"
./bob "calculate optimal position sizing for current market"

# Performance analysis
./bob "analyze trading performance last 30 days"
./bob "review wheel strategy effectiveness"
./bob "compare actual vs expected returns"

# Market analysis
./bob "analyze Unity options flow patterns"
./bob "review implied volatility trends"
./bob "identify arbitrage opportunities"
```

#### System Analysis
```bash
# Performance analysis
./bob "analyze system resource utilization"
./bob "identify memory leaks and performance issues"
./bob "review database query performance"

# Health analysis
./bob "analyze system health metrics"
./bob "review error rates and failure patterns"
./bob "assess hardware utilization efficiency"
```

### 2. Code Generation Commands

#### Feature Development
```bash
# New features
./bob "create real-time position monitoring dashboard"
./bob "implement advanced risk management alerts"
./bob "add options Greeks calculation module"

# API development
./bob "create RESTful API for trading operations"
./bob "implement authentication middleware"
./bob "add rate limiting to API endpoints"

# UI components
./bob "create trading dashboard with live updates"
./bob "implement position management interface"
./bob "add risk visualization components"
```

#### Testing and Documentation
```bash
# Test generation
./bob "generate unit tests for options pricing module"
./bob "create integration tests for trading workflow"
./bob "add property-based tests for risk calculations"

# Documentation
./bob "generate API documentation for trading functions"
./bob "create user guide for wheel strategy"
./bob "add inline documentation to risk module"
```

### 3. Optimization Commands

#### Performance Optimization
```bash
# System optimization
./bob "optimize database query performance"
./bob "improve memory usage in agent pool"
./bob "accelerate options pricing calculations"

# Trading optimization
./bob "optimize wheel strategy parameters for Unity"
./bob "improve position sizing algorithm"
./bob "optimize risk calculation performance"

# Hardware optimization
./bob "optimize GPU utilization for ML operations"
./bob "improve CPU cache efficiency"
./bob "optimize memory allocation patterns"
```

#### Configuration Optimization
```bash
# System configuration
./bob "optimize agent pool configuration"
./bob "tune hardware resource allocation"
./bob "optimize search index parameters"

# Trading configuration
./bob "optimize risk limits for current market"
./bob "tune strategy parameters for volatility"
./bob "adjust position sizing for portfolio size"
```

### 4. Fix and Maintenance Commands

#### Bug Fixes
```bash
# Error fixes
./bob "fix authentication timeout issues"
./bob "resolve database connection problems"
./bob "fix memory leaks in agent pool"

# Logic fixes
./bob "fix Greeks calculation errors"
./bob "resolve position sizing edge cases"
./bob "fix risk limit validation logic"

# Integration fixes
./bob "fix API endpoint authentication"
./bob "resolve data synchronization issues"
./bob "fix real-time update problems"
```

#### System Maintenance
```bash
# Database maintenance
./bob "clean up old trading data"
./bob "optimize database indexes"
./bob "repair database integrity issues"

# Cache maintenance
./bob "clear stale cache entries"
./bob "rebuild search index"
./bob "optimize cache hit rates"

# Log maintenance
./bob "rotate and archive log files"
./bob "clean up debug artifacts"
./bob "optimize log storage"
```

### 5. System Management Commands

#### Status and Health
```bash
# System status
./bob status
./bob health-check
./bob system-info

# Component status
./bob status --agents
./bob status --tools
./bob status --trading

# Performance status
./bob performance-report
./bob resource-usage
./bob thermal-status
```

#### Configuration Management
```bash
# View configuration
./bob config show
./bob config validate
./bob config backup

# Update configuration
./bob config set agents.count 8
./bob config set hardware.memory_limit_gb 20
./bob config reload
```

#### Service Management
```bash
# Start/stop services
./bob start
./bob stop
./bob restart

# Service-specific control
./bob start --agents-only
./bob restart --trading-system
./bob stop --graceful
```

## System Routing Intelligence

BOB automatically routes queries to the most appropriate system component:

### Einstein Search Routing
Queries routed to Einstein semantic search:
- Code understanding: "explain how options pricing works"
- Pattern finding: "find all error handling patterns"
- Architecture queries: "show dependencies between modules"
- Documentation requests: "find examples of wheel strategy usage"

### BOLT Agent Routing  
Queries routed to multi-agent system:
- Complex analysis: "analyze and optimize entire trading system"
- Multi-step tasks: "create feature with tests and documentation"
- Cross-cutting concerns: "improve error handling across all modules"
- Large refactoring: "modernize entire authentication system"

### Direct Execution Routing
Queries executed directly:
- Simple operations: "show system status"
- Configuration changes: "update memory limits"
- Quick fixes: "restart failed service"
- Data queries: "show current positions"

### Hybrid Routing
Complex queries may use multiple systems:
- Einstein for context gathering
- BOLT agents for parallel execution
- Direct tools for specific operations
- Trading system for domain logic

## Interactive Mode Reference

### Starting Interactive Mode

```bash
# Basic interactive mode
./bob --interactive

# Interactive with context
./bob -i --context="trading"
./bob -i --context="development"
./bob -i --context="analysis"
```

### Interactive Commands

```
bob> help                    # Show help
bob> help <topic>           # Topic-specific help
bob> context                # Show current context
bob> workflow               # Show available workflows
bob> status                 # System status
bob> config                 # Configuration commands
bob> history               # Command history
bob> clear                 # Clear screen
bob> exit                  # Exit interactive mode
```

### Context Management

```
bob> context trading        # Switch to trading context
bob> context development    # Switch to development context  
bob> context analysis       # Switch to analysis context
bob> context clear          # Clear context
bob> context show           # Show current context
```

### Workflow System

```
bob> workflow
🔄 Available Workflows:
1. Fix Code Issues
2. Optimize Performance
3. Generate New Features
4. Review Security
5. Trading Analysis
6. System Maintenance

bob> workflow 1             # Start workflow 1
bob> workflow trading       # Start trading workflow
bob> workflow cancel        # Cancel current workflow
```

## Advanced Usage

### Query Modifiers

```bash
# Analysis modifiers
./bob "analyze system performance" --deep
./bob "review code quality" --comprehensive
./bob "find security issues" --thorough

# Execution modifiers
./bob "optimize performance" --aggressive
./bob "fix authentication" --safe-mode
./bob "deploy changes" --dry-run

# Output modifiers
./bob "system status" --json
./bob "performance report" --detailed
./bob "error analysis" --summary
```

### Chained Commands

```bash
# Sequential execution
./bob "analyze performance; optimize bottlenecks; validate improvements"

# Conditional execution
./bob "check system health && deploy changes"

# Pipeline operations
./bob "analyze code | generate report | send notification"
```

### Batch Operations

```bash
# Batch file processing
./bob batch --file commands.txt
./bob batch --commands "cmd1,cmd2,cmd3"

# Scheduled operations
./bob schedule "daily health check" --cron "0 9 * * *"
./bob schedule "weekly optimization" --interval 7d
```

## Configuration Options

### Command-Line Options

```bash
# Global options
--config PATH              # Configuration file path
--log-level LEVEL         # Logging level (DEBUG, INFO, WARN, ERROR)
--verbose, -v             # Verbose output
--quiet, -q               # Quiet mode
--dry-run                 # Preview without execution
--trace                   # Enable execution tracing

# Performance options
--agents N                # Number of agents to use
--memory-limit GB         # Memory limit
--cpu-cores N             # CPU cores to use
--gpu-backend BACKEND     # GPU backend (metal, mlx, cpu)

# Output options
--json                    # JSON output format
--yaml                    # YAML output format
--format FORMAT           # Custom output format
--output FILE             # Output to file
```

### Environment Variables

```bash
# Core configuration
export BOB_CONFIG_PATH="config.yaml"
export BOB_LOG_LEVEL="INFO"
export BOB_INTERACTIVE_HISTORY="true"

# Performance tuning
export BOB_DEFAULT_AGENTS=8
export BOB_MEMORY_LIMIT_GB=20
export BOB_CPU_CORES=12

# Behavior configuration
export BOB_AUTO_CONFIRM="false"
export BOB_COLOR_OUTPUT="true"
export BOB_PROGRESS_BAR="true"
```

## Error Handling

### Error Categories

1. **Syntax Errors**: Invalid command syntax
2. **Configuration Errors**: Invalid configuration
3. **Resource Errors**: Insufficient resources
4. **Service Errors**: Service unavailable
5. **Trading Errors**: Trading system issues

### Error Recovery

```bash
# Automatic recovery
./bob "fix system errors" --auto-recover

# Manual recovery
./bob diagnose             # Diagnose issues
./bob fix-config           # Fix configuration
./bob restart-services     # Restart services
./bob restore-backup       # Restore from backup
```

### Debug Mode

```bash
# Enable debug mode
./bob --debug "your command"
./bob --trace "your command"
./bob --profile "your command"

# Debug specific components
./bob --debug-agents "your command"
./bob --debug-tools "your command" 
./bob --debug-trading "your command"
```

## Performance Optimization

### Query Optimization

```bash
# Fast queries (prefer specific over general)
./bob "status agents"              # ✅ Specific
./bob "show me everything"         # ❌ Too general

# Efficient context usage
./bob -c trading "analyze positions"    # ✅ With context
./bob "analyze Unity trading positions" # ✅ Explicit context
```

### Resource Management

```bash
# Limit resource usage
./bob --agents 4 "resource intensive task"
./bob --memory-limit 10 "memory intensive operation"

# Monitor resource usage
./bob monitor --resources
./bob status --performance
```

## Integration Examples

### Trading Workflow

```bash
# Complete trading analysis workflow
./bob workflow trading

# Or step by step:
./bob "analyze current Unity positions"
./bob "identify market opportunities"
./bob "calculate optimal position sizes"
./bob "generate trade recommendations"
./bob "validate risk parameters"
./bob "execute approved trades"
```

### Development Workflow

```bash
# Complete development workflow
./bob workflow development

# Or step by step:
./bob "analyze code quality"
./bob "identify improvement opportunities"
./bob "generate optimized implementations"
./bob "create comprehensive tests"
./bob "update documentation"
./bob "validate changes"
```

### System Maintenance Workflow

```bash
# Complete maintenance workflow
./bob workflow maintenance

# Or step by step:
./bob "analyze system health"
./bob "identify maintenance needs"
./bob "optimize performance"
./bob "clean up resources"
./bob "update configurations"
./bob "validate system state"
```

## Tips and Best Practices

### Query Formulation

1. **Be Specific**: "fix authentication timeout in storage.py" vs "fix auth"
2. **Provide Context**: Include relevant file names, modules, or components
3. **Use Action Words**: start, stop, analyze, optimize, fix, create
4. **Specify Scope**: "entire system" vs "trading module" vs "specific file"

### Efficient Usage  

1. **Use Interactive Mode**: For multiple related commands
2. **Leverage Context**: Set context once, use multiple commands
3. **Use Workflows**: For common multi-step operations
4. **Monitor Resources**: Check system status before heavy operations

### Error Prevention

1. **Validate First**: Use --dry-run for destructive operations
2. **Check Status**: Verify system health before complex operations
3. **Use Appropriate Resources**: Don't over-allocate agents/memory
4. **Monitor Progress**: Use --verbose for long-running operations

---

## Quick Reference Card

```bash
# Essential commands
./bob status                    # System status
./bob help                     # General help
./bob --interactive            # Interactive mode

# Common operations
./bob "analyze [target]"       # Analyze code/system
./bob "fix [problem]"          # Fix issues
./bob "optimize [component]"   # Optimize performance
./bob "create [feature]"       # Generate new code

# System management
./bob restart                  # Restart system
./bob config show             # Show configuration
./bob performance-report      # Performance metrics

# Trading operations
./bob "analyze positions"      # Position analysis
./bob "optimize strategy"      # Strategy optimization
./bob "check risk limits"      # Risk validation
```

For detailed help on any command: `./bob help <command>`