# Meta Build Instructions for Claude Code CLI

## Overview
This document provides build instructions for the meta system components in the wheel-trading codebase. The meta system provides development workflow automation including file monitoring, quality checking, and template-based code improvements.

## Build Commands

### Primary Build Command
```bash
# Standard build with meta integration
python -m pytest tests/ -v && python run.py --validate
```

### Meta-Enhanced Build
```bash
# Build with meta system active
python meta_coordinator.py --build-mode &
python -m pytest tests/ -v --tb=short
python run.py --diagnose
```

## Meta Integration Workflow

### 1. Pre-Development Setup
```bash
# Initialize meta system (run once per session)
python meta_prime.py &
python meta_watcher.py &
```

### 2. Development Loop
```bash
# For each major code change:
python meta_coordinator.py --observe-changes
# Make your code changes
python meta_auditor.py --validate-changes
# Commit if validation passes
```

### 3. Testing with Meta
```bash
# Run tests with meta observation
python meta_coordinator.py --test-mode &
python -m pytest tests/ -v
python meta_auditor.py --test-report
```

## Meta-Aware Code Patterns

### When Creating New Files
Every new Python file should include meta-awareness:

```python
# At the top of new files
from meta_prime import MetaPrime
from typing import Dict, Any

# Initialize meta observation for this module
meta = MetaPrime()
meta.observe("module_creation", {"module": __name__})

# Your code here
```

### When Modifying Existing Files
Before major modifications:

```python
# Add to existing files being modified
meta.observe("code_modification", {
    "file": __file__,
    "modification_type": "enhancement",  # or "bugfix", "refactor"
    "purpose": "describe what you're changing"
})
```

## Build Validation

### Code Quality Checks
```bash
# Meta-driven quality validation
python meta_auditor.py --check-quality --file <filepath>
```

### Performance Validation
```bash
# Meta-driven performance check
python meta_coordinator.py --performance-check
```

## Hardware Optimization

### M4 Pro Specific Commands
```bash
# Enable M4 Pro optimizations
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "from src.unity_wheel.accelerated_tools.sequential_thinking_mac_optimized import get_sequential_thinking_turbo; print('M4 Pro optimizations active')"
```

### Accelerated Tools Integration
```bash
# Test accelerated tools with meta
python test_all_accelerated_tools.py
python meta_coordinator.py --validate-acceleration
```

## Component-Specific Build Instructions

### Trading Components (src/unity_wheel/strategy/)
```bash
# Build trading components with meta
python meta_coordinator.py --component strategy
python -m pytest tests/test_wheel.py -v
python meta_auditor.py --domain trading
```

### Risk Management (src/unity_wheel/risk/)
```bash
# Build risk components with meta
python meta_coordinator.py --component risk
python -m pytest tests/test_risk*.py -v
```

### API Layer (src/unity_wheel/api/)
```bash
# Build API with meta
python meta_coordinator.py --component api
python -m pytest tests/test_*api*.py -v
python run.py --validate-api
```

## Testing Strategy

### Meta-Enhanced Testing
```bash
# Run tests with meta intelligence
python meta_coordinator.py --test-intelligence &
python -m pytest tests/ -v --tb=short
python meta_auditor.py --test-analysis
```

### Performance Testing
```bash
# Test with performance monitoring
python meta_coordinator.py --performance-monitor &
python -m pytest tests/test_performance_*.py -v
```

## Error Handling and Debugging

### Meta-Aware Debugging
```bash
# Debug with meta intelligence
python meta_coordinator.py --debug-mode
python meta_auditor.py --error-analysis
```

### Meta System Health Check
```bash
# Verify meta system is working
python meta_prime.py --health-check
python meta_coordinator.py --status
```

## Deployment Preparation

### Pre-Deployment Validation
```bash
# Complete meta validation before deployment
python meta_auditor.py --comprehensive-audit
python meta_coordinator.py --deployment-readiness
```

### Production Build
```bash
# Build for production with meta
python meta_coordinator.py --production-mode
python -m pytest tests/ -v --tb=no
python run.py --production-validate
```

## Continuous Integration

### CI/CD Pipeline Integration
```bash
# For automated builds
python meta_prime.py --ci-mode &
python meta_coordinator.py --ci-validate
python -m pytest tests/ -v --junitxml=results.xml
python meta_auditor.py --ci-report
```

## Meta System Maintenance

### Regular Maintenance Tasks
```bash
# Clean up meta data (run weekly)
python meta_coordinator.py --cleanup-old-data

# Update meta intelligence (run after major changes)
python meta_coordinator.py --update-intelligence

# Backup meta state
python meta_coordinator.py --backup-state
```

### Meta System Updates
```bash
# Update meta capabilities
python meta_generator.py --self-update
python meta_coordinator.py --validate-update
```

## Environment Setup

### Required Environment Variables
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export META_ENABLED=true
export HARDWARE_ACCELERATION=m4_pro
```

### Development Environment
```bash
# Complete development setup
python meta_coordinator.py --setup-dev-env
source startup.sh
```

## Integration with Existing CLAUDE.md

This meta build system integrates with the existing CLAUDE.md workflow:

1. **Accelerated Tools**: Meta system works with all accelerated tools
2. **Hardware Optimization**: Meta system uses M4 Pro optimizations
3. **Testing**: Meta system enhances existing test workflows
4. **Development**: Meta system observes and improves development patterns

## Success Indicators

The build is successful when:
- All tests pass with meta observation active
- Meta auditor reports no critical issues
- Performance metrics show expected M4 Pro utilization
- Meta system reports healthy evolution state

## Troubleshooting

### Common Issues
1. **Meta system not responding**: Restart with `python meta_prime.py`
2. **Performance degradation**: Check with `python meta_coordinator.py --performance-check`
3. **Build failures**: Validate with `python meta_auditor.py --build-diagnosis`

### Emergency Procedures
```bash
# Stop all meta processes
pkill -f meta_

# Restart meta system
python meta_prime.py &
python meta_coordinator.py --recovery-mode
```

## Final Notes

The meta system is designed to be:
- **Non-intrusive**: Works alongside existing workflows
- **Self-improving**: Gets better with each build
- **Hardware-optimized**: Leverages M4 Pro capabilities
- **Quality-focused**: Enhances code quality automatically

Always run meta validation before committing code changes.