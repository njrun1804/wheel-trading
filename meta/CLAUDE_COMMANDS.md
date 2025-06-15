# Meta Commands for Claude Code CLI

## Overview
This document provides specific commands for Claude Code CLI to interact with the meta system effectively.

## Essential Meta Commands

### Development Start
```bash
# Start meta-aware development session
python meta_coordinator.py --dev-mode &
```
This command:
- Initializes meta system observation
- Begins learning from your code patterns
- Enables real-time code evolution
- Starts file watching for all meta components

### Build Process
```bash
# Meta-enhanced build
python meta_coordinator.py --build-mode
python -m pytest tests/ -v
python meta_auditor.py --validate-build
```

### Testing with Meta
```bash
# Run tests with meta intelligence
python meta_coordinator.py --test-mode &
python -m pytest tests/ -v --tb=short
python meta_auditor.py --test-analysis
```

### Code Quality Validation
```bash
# Validate using Claude's proven patterns
python meta_auditor.py --claude-patterns
python meta_auditor.py --trading-domain-specific
```

## Meta System Status

### Check Meta Health
```bash
# Verify meta system is operational
python meta_prime.py --health-check
python meta_coordinator.py --status
python meta_watcher.py --status
```

### View Meta Intelligence
```bash
# See what meta system has learned
python meta_coordinator.py --learning-report
python meta_auditor.py --pattern-analysis
```

## Code Integration Commands

### Initialize Meta for New Files
```bash
# Add meta awareness to new Python files
python meta_coordinator.py --init-file <filepath>
```

### Validate Meta Integration
```bash
# Check if file has proper meta integration
python meta_auditor.py --check-meta-integration <filepath>
```

## Evolution Commands

### Manual Evolution Trigger
```bash
# Trigger evolution based on current observations
python meta_coordinator.py --evolve-now
```

### View Evolution History
```bash
# See what the system has evolved
python meta_coordinator.py --evolution-history
```

### Rollback Evolution
```bash
# Rollback last evolution if needed
python meta_coordinator.py --rollback-last
```

## Performance Commands

### Performance Analysis
```bash
# Analyze system performance with meta intelligence
python meta_coordinator.py --performance-analysis
```

### Hardware Optimization
```bash
# Optimize for M4 Pro with meta awareness
python meta_coordinator.py --optimize-m4-pro
```

## Debugging Commands

### Meta-Aware Debugging
```bash
# Debug with meta intelligence
python meta_coordinator.py --debug-mode
python meta_auditor.py --error-analysis
```

### Trace Meta Operations
```bash
# See what meta system is doing
python meta_coordinator.py --trace-operations
```

## Maintenance Commands

### Daily Maintenance
```bash
# Run daily meta maintenance
python meta_coordinator.py --daily-maintenance
```

### Clean Meta Data
```bash
# Clean up old meta observations
python meta_coordinator.py --cleanup-data --older-than 7d
```

### Backup Meta State
```bash
# Backup meta system state
python meta_coordinator.py --backup --location meta_backups/
```

## Emergency Commands

### Stop All Meta Processes
```bash
# Emergency stop
pkill -f meta_
```

### Restart Meta System
```bash
# Restart meta system
python meta_coordinator.py --restart
```

### Recovery Mode
```bash
# Start in recovery mode
python meta_coordinator.py --recovery-mode
```

## Integration with Existing Workflows

### With Existing Startup
```bash
# Integrate with existing startup.sh
python meta_coordinator.py --integrate-startup
source startup.sh
```

### With Testing
```bash
# Meta-aware testing
python meta_coordinator.py --test-integration
python -m pytest tests/ -v
```

### With Deployment
```bash
# Meta-aware deployment preparation
python meta_coordinator.py --deployment-prep
python meta_auditor.py --deployment-readiness
```

## Configuration Commands

### Set Meta Configuration
```bash
# Configure meta system behavior
python meta_coordinator.py --config observation_level=detailed
python meta_coordinator.py --config evolution_enabled=true
python meta_coordinator.py --config hardware_optimization=m4_pro
```

### View Meta Configuration
```bash
# See current meta configuration
python meta_coordinator.py --show-config
```

## Reporting Commands

### Generate Meta Report
```bash
# Comprehensive meta system report
python meta_coordinator.py --full-report
```

### Performance Report
```bash
# Performance improvement report
python meta_coordinator.py --performance-report
```

### Learning Report
```bash
# What the system has learned
python meta_coordinator.py --learning-report
```

## Advanced Commands

### Custom Evolution
```bash
# Run custom evolution strategy
python meta_coordinator.py --custom-evolution --strategy performance_optimization
```

### Meta-Analysis
```bash
# Deep analysis of meta system behavior
python meta_coordinator.py --meta-analysis
```

### Export Meta Data
```bash
# Export meta observations for analysis
python meta_coordinator.py --export-data --format json --output meta_export.json
```

## Command Combinations

### Full Development Session
```bash
# Complete meta-aware development session
python meta_coordinator.py --dev-mode &
python meta_watcher.py --watch-all &
python meta_auditor.py --continuous-validation &
```

### Complete Build and Test
```bash
# Full build with meta intelligence
python meta_coordinator.py --build-mode
python -m pytest tests/ -v
python meta_auditor.py --comprehensive-audit
python meta_coordinator.py --evolution-check
```

### Production Deployment
```bash
# Production-ready deployment with meta
python meta_coordinator.py --production-mode
python meta_auditor.py --production-validation
python meta_coordinator.py --deployment-report
```

## Usage Notes

1. **Always start with**: `python meta_coordinator.py --dev-mode &`
2. **Before building**: `python meta_auditor.py --validate`
3. **After major changes**: `python meta_coordinator.py --evolution-check`
4. **For debugging**: `python meta_coordinator.py --debug-mode`
5. **End of session**: `python meta_coordinator.py --session-summary`

## Success Indicators

Commands are working correctly when:
- Meta system shows "healthy" status
- File watching is active
- Evolution readiness is appropriate
- Performance metrics are improving
- Code quality scores are increasing

## Error Handling

If meta commands fail:
1. Check meta system health: `python meta_prime.py --health-check`
2. Restart if needed: `python meta_coordinator.py --restart`
3. Check logs: `python meta_coordinator.py --show-logs`
4. Recovery mode: `python meta_coordinator.py --recovery-mode`

These commands ensure Claude Code CLI can effectively interact with and leverage the meta system's capabilities.