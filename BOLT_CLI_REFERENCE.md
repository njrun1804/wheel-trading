# Bolt CLI Reference

## Command Overview

Bolt provides a simple yet powerful command-line interface for accessing the 8-agent hardware-accelerated problem-solving system.

## Basic Syntax

```bash
bolt solve "query" [OPTIONS]
```

## Primary Command

### `bolt solve`

Execute a problem-solving query using the 8-agent system.

**Syntax:**
```bash
bolt solve "your query here" [--analyze-only] [--help]
```

**Arguments:**
- `query` (required): The problem description or task you want Bolt to solve

**Options:**
- `--analyze-only`: Only analyze the query without making any changes
- `--help`: Show help message and exit

## Command Examples

### Basic Usage
```bash
# Simple optimization request
bolt solve "optimize database performance"

# Analysis without changes
bolt solve "find performance bottlenecks" --analyze-only

# Complex multi-part query
bolt solve "refactor wheel strategy module for better maintainability and add error handling"
```

### Direct Python Script Execution
```bash
# Run via Python script
python bolt_cli.py "analyze code quality issues"

# With full path
python /path/to/wheel-trading/bolt_cli.py "debug memory leaks"
```

## Option Details

### `--analyze-only`

**Purpose**: Analyze the query and show what would be done without making any changes

**Use Cases**:
- Understanding scope before making changes
- Exploring possibilities without risk
- Learning what Bolt would do
- Validating query interpretation

**Examples**:
```bash
# Safe exploration
bolt solve "refactor entire codebase" --analyze-only

# Impact assessment
bolt solve "migrate to new database" --analyze-only

# Understanding complexity
bolt solve "add real-time monitoring" --analyze-only
```

**Output with --analyze-only**:
```bash
=== Query Analysis ===
Query: optimize database performance
Relevant files: 8
Planned tasks: 5
Estimated agents: 5

=== Planned Tasks ===
1. analyze_scope: optimize database performance (Priority: CRITICAL)
2. profile_performance: identify bottlenecks (Priority: HIGH)
3. analyze_memory: check memory usage patterns (Priority: NORMAL)
4. suggest_optimizations: generate optimization plan (Priority: HIGH)
5. create_implementation_plan: detailed steps (Priority: NORMAL)
```

## Environment Variables

### Core Configuration
```bash
# Memory limit for GPU operations (default: 18GB)
export PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=19327352832

# Allow duplicate OpenMP libraries
export KMP_DUPLICATE_LIB_OK=TRUE

# Force CPU-only mode (disables GPU acceleration)
export MLX_FORCE_CPU=1

# Enable debug logging
export BOLT_LOG_LEVEL=DEBUG

# Set custom temp directory
export BOLT_TEMP_DIR="/tmp/bolt"
```

### Performance Tuning
```bash
# CPU affinity (P-cores only)
export BOLT_CPU_AFFINITY="0,1,2,3,4,5,6,7"

# Memory allocation strategy
export BOLT_MEMORY_STRATEGY="aggressive"

# GPU optimization
export MLX_METAL_CACHE_ENABLE=1
export MLX_GPU_CORES=20
```

### Development Options
```bash
# Enable profiling
export BOLT_PROFILING=1

# Enable performance monitoring
export BOLT_MONITORING=1

# Set recursion limit (default: 1)
export BOLT_MAX_RECURSION=1

# Agent count override (default: 8)
export BOLT_AGENT_COUNT=8
```

## Configuration Files

### User Configuration (`~/.bolt/config.yaml`)
```yaml
# User-specific Bolt configuration
default:
  analyze_only: false
  log_level: INFO
  temp_dir: /tmp/bolt

agents:
  count: 8
  timeout: 300

hardware:
  memory_limit_gb: 18
  gpu_enabled: true
  
tools:
  concurrency_limits:
    semantic_search: 3
    pattern_search: 4
    code_analysis: 2
```

### Project Configuration (`bolt.yaml`)
```yaml
# Project-specific configuration
project:
  name: "wheel-trading"
  type: "python"
  
optimization:
  focus_areas:
    - "src/unity_wheel/strategy/"
    - "src/unity_wheel/math/"
  
  exclude_patterns:
    - "test_*"
    - "*_backup.py"
    - "venv/"
```

## Exit Codes

Bolt returns specific exit codes to indicate execution status:

- `0`: Success - operation completed successfully
- `1`: Error - operation failed due to error
- `2`: Invalid arguments - command syntax error
- `3`: System error - hardware or system resource issue
- `4`: Timeout - operation exceeded time limit
- `5`: Memory error - insufficient memory for operation

**Example Usage**:
```bash
# Check exit code in scripts
bolt solve "optimize code"
if [ $? -eq 0 ]; then
    echo "Optimization successful"
else
    echo "Optimization failed with code $?"
fi
```

## Integration with Shell Scripts

### Basic Script Integration
```bash
#!/bin/bash
# bolt_optimization.sh

echo "Starting code optimization..."

# Run analysis first
bolt solve "analyze performance issues" --analyze-only
if [ $? -ne 0 ]; then
    echo "Analysis failed, exiting"
    exit 1
fi

# Run actual optimization
bolt solve "optimize identified performance issues"
echo "Optimization complete"
```

### Advanced Script with Error Handling
```bash
#!/bin/bash
# advanced_bolt_workflow.sh

set -e  # Exit on any error

# Configuration
BOLT_TIMEOUT=300
BOLT_LOG_FILE="/tmp/bolt_$(date +%s).log"

# Function to run Bolt with logging
run_bolt() {
    local query="$1"
    local options="${2:-}"
    
    echo "Running: bolt solve \"$query\" $options"
    
    # Run with timeout and logging
    timeout $BOLT_TIMEOUT bolt solve "$query" $options 2>&1 | tee -a "$BOLT_LOG_FILE"
    
    local exit_code=$?
    
    case $exit_code in
        0) echo "✅ Success: $query" ;;
        1) echo "❌ Error: $query" ;;
        2) echo "❌ Invalid arguments: $query" ;;
        3) echo "❌ System error: $query" ;;
        4) echo "⏰ Timeout: $query" ;;
        5) echo "💾 Memory error: $query" ;;
        124) echo "⏰ Command timeout: $query" ;;
        *) echo "❓ Unknown error ($exit_code): $query" ;;
    esac
    
    return $exit_code
}

# Main workflow
main() {
    echo "🚀 Starting Bolt workflow..."
    
    # Step 1: Analysis
    run_bolt "analyze project for optimization opportunities" "--analyze-only" || {
        echo "Analysis failed, stopping workflow"
        exit 1
    }
    
    # Step 2: Performance optimization
    run_bolt "optimize performance bottlenecks identified in analysis" || {
        echo "Performance optimization failed"
        exit 1
    }
    
    # Step 3: Code quality
    run_bolt "improve code quality issues" || {
        echo "Code quality improvement failed"
        exit 1
    }
    
    # Step 4: Verification
    run_bolt "verify no regressions after optimization" "--analyze-only" || {
        echo "Verification failed"
        exit 1
    }
    
    echo "🎉 Workflow completed successfully"
    echo "Log file: $BOLT_LOG_FILE"
}

# Run main function
main "$@"
```

## Integration with Git Hooks

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running Bolt pre-commit analysis..."

# Check for code quality issues
bolt solve "analyze staged changes for quality issues" --analyze-only

if [ $? -ne 0 ]; then
    echo "❌ Code quality issues found, commit blocked"
    echo "Run 'bolt solve \"fix code quality issues\"' to address them"
    exit 1
fi

echo "✅ Code quality check passed"
```

### Pre-push Hook
```bash
#!/bin/bash
# .git/hooks/pre-push

echo "Running comprehensive analysis before push..."

# Check for performance regressions
bolt solve "verify no performance regressions in recent changes" --analyze-only

if [ $? -ne 0 ]; then
    echo "❌ Performance issues detected, push blocked"
    exit 1
fi

echo "✅ Performance check passed"
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/bolt-analysis.yml
name: Bolt Code Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: macos-latest  # Required for Metal GPU support
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Bolt
      run: |
        pip install -r requirements.txt
        python install_bolt.py
    
    - name: Run Bolt Analysis
      run: |
        bolt solve "analyze code quality and performance" --analyze-only
        
    - name: Upload Results
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: bolt-analysis-results
        path: /tmp/bolt_*.log
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent {
        label 'macos'  // Mac agent required for Metal GPU
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'python install_bolt.py'
            }
        }
        
        stage('Analysis') {
            steps {
                sh '''
                    bolt solve "comprehensive code analysis" --analyze-only || true
                '''
            }
        }
        
        stage('Optimization') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    bolt solve "apply safe optimizations"
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '/tmp/bolt_*.log', allowEmptyArchive: true
        }
    }
}
```

## Advanced Usage Patterns

### Batch Processing
```bash
# Process multiple queries from file
while IFS= read -r query; do
    echo "Processing: $query"
    bolt solve "$query" --analyze-only
done < queries.txt
```

### Parallel Execution
```bash
# Run multiple Bolt instances (not recommended for resource-intensive queries)
bolt solve "optimize module A" &
bolt solve "optimize module B" &
bolt solve "optimize module C" &
wait  # Wait for all to complete
```

### Conditional Execution
```bash
# Only run if specific conditions are met
if [ -f "src/unity_wheel/strategy/wheel.py" ]; then
    bolt solve "optimize wheel strategy implementation"
else
    echo "Wheel strategy module not found"
fi
```

## Troubleshooting Commands

### System Check
```bash
# Verify installation
bolt solve "test system functionality" --analyze-only

# Check hardware detection
python -c "
from bolt.hardware_state import get_hardware_state
hw = get_hardware_state()
print(f'System: {hw.cpu.p_cores}P + {hw.cpu.e_cores}E cores, {hw.memory.total_gb:.1f}GB RAM')
"
```

### Debug Mode
```bash
# Enable debug logging
export BOLT_LOG_LEVEL=DEBUG
bolt solve "debug query" --analyze-only 2>&1 | tee debug.log
```

### Performance Testing
```bash
# Quick performance test
time bolt solve "analyze small code section" --analyze-only

# Memory usage monitoring
/usr/bin/time -l bolt solve "memory intensive query" --analyze-only
```

## Tips and Best Practices

### Query Writing
- Be specific and actionable
- Include context when possible
- Use domain-specific terminology
- Specify scope boundaries when needed

### Performance Optimization
- Use `--analyze-only` for exploration
- Start with focused queries before broad ones
- Monitor system resources during execution
- Consider breaking complex queries into steps

### Error Handling
- Always check exit codes in scripts
- Use appropriate timeouts
- Log output for debugging
- Have fallback strategies

### Integration
- Use configuration files for consistent settings
- Implement proper error handling in scripts
- Monitor performance in CI/CD pipelines
- Archive logs for troubleshooting

This CLI reference provides comprehensive information for effectively using Bolt's command-line interface in various contexts and workflows.