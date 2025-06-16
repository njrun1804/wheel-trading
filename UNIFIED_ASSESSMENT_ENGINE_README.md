# Unified Assessment Engine (UAE)

A single coherent path for processing any natural language command in the wheel trading system. The UAE combines Einstein semantic search, intelligent intent analysis, sophisticated action planning, and optimized execution routing to provide a seamless command processing experience.

## 🎯 Overview

The Unified Assessment Engine creates a single entry point for all natural language commands by:

1. **Context Gathering** - Einstein-powered semantic understanding and code analysis
2. **Intent Analysis** - Multi-model intent classification with context awareness
3. **Action Planning** - Task decomposition and resource optimization
4. **Execution Routing** - Bolt multi-agent and direct tool coordination

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from unity_wheel.unified_assessment import UnifiedAssessmentEngine

async def main():
    # Initialize the engine
    engine = UnifiedAssessmentEngine()
    await engine.initialize()
    
    try:
        # Process a command
        result = await engine.process_command("fix authentication issue")
        
        print(f"Success: {result.success}")
        print(f"Summary: {result.summary}")
        print(f"Findings: {result.findings}")
        print(f"Recommendations: {result.recommendations}")
        
    finally:
        await engine.shutdown()

asyncio.run(main())
```

### Run the Demo

```bash
# Run comprehensive demo suite
python unified_assessment_demo.py

# Run interactive demo
python unified_assessment_demo.py --interactive
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Natural Language Command                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Unified Assessment Engine                       │
│                                                                  │
│  Context Gathering → Intent Analysis → Action Planning          │
│                              ↓                                   │
│                    Execution Routing                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 1. Context Gathering Layer
- **Einstein Semantic Search**: Sub-100ms semantic code understanding
- **Code Structure Analysis**: AST parsing and pattern detection
- **Dependency Graph Navigation**: Relationship mapping
- **Historical Context**: Previous command analysis

#### 2. Intent Analysis Layer
- **Pattern-based Classification**: Rule-based intent detection
- **Context-aware Disambiguation**: Resolve ambiguities using context
- **Confidence Scoring**: Multi-level confidence assessment
- **Parameter Extraction**: Extract targets and constraints

#### 3. Action Planning Layer
- **Task Decomposition**: Break complex intents into atomic tasks
- **Dependency Resolution**: Analyze task dependencies
- **Resource Allocation**: Optimize for M4 Pro hardware
- **Execution Strategy Selection**: Choose optimal execution approach

#### 4. Execution Routing Layer
- **Bolt Multi-Agent**: Complex multi-file operations
- **Direct Tools**: Simple, fast operations
- **Hybrid Execution**: Mixed workload optimization
- **Progress Monitoring**: Real-time execution tracking

## 📝 Supported Commands

### Intent Categories

The UAE supports these primary intent categories:

#### FIX Commands
```
"fix authentication issue"
"resolve the memory leak in options pricing"
"debug the failing tests"
```

#### CREATE Commands
```
"create new trading strategy for Unity stock"
"add a new risk management component"
"build a backtesting framework"
```

#### OPTIMIZE Commands
```
"optimize performance of the wheel strategy"
"improve memory usage in data processing"
"speed up the options chain analysis"
```

#### ANALYZE Commands
```
"analyze risk management components"
"review error handling patterns"
"examine the trading calendar integration"
```

#### REFACTOR Commands
```
"refactor the position sizing logic"
"clean up the authentication code"
"restructure the data providers"
```

#### QUERY Commands
```
"show me all authentication related files"
"find Unity trading strategies"
"list all database connections"
```

#### TEST Commands
```
"test the backtesting system"
"run integration tests"
"validate the risk calculations"
```

### Command Examples by Complexity

#### Simple Commands (Direct Tools)
- "show Unity options data"
- "find authentication files"
- "list trading strategies"

#### Medium Commands (Hybrid Execution)
- "analyze wheel strategy performance"
- "optimize database queries"
- "review error handling"

#### Complex Commands (Bolt Multi-Agent)
- "fix authentication issue across all components"
- "create comprehensive trading dashboard"
- "optimize entire trading pipeline performance"

## ⚙️ Configuration

### Engine Configuration

```python
config = {
    "context": {
        "max_files": 50,           # Max files to analyze
        "search_depth": 3,         # Semantic search depth
        "include_dependencies": True,
        "include_history": True,
        "optimization_target": "balanced"  # speed, accuracy, balanced
    },
    "intent": {
        "confidence_threshold": 0.6,
        "enable_clarification": True,
        "max_alternatives": 3
    },
    "planning": {
        "max_parallel_tasks": 8,
        "resource_optimization": True,
        "hardware_profile": "m4_pro"
    },
    "routing": {
        "prefer_bolt_for_complex": True,
        "bolt_agent_count": 8,
        "enable_fallbacks": True
    }
}

engine = UnifiedAssessmentEngine(config)
```

### Optimization Targets

- **speed**: Prioritize fast response times
- **accuracy**: Prioritize comprehensive analysis
- **balanced**: Balance speed and accuracy (default)

## 📊 Performance Metrics

### Latency Targets
- **Context Gathering**: <200ms (parallel Einstein queries)
- **Intent Analysis**: <50ms (cached patterns)
- **Action Planning**: <100ms (pre-computed strategies)
- **Total Processing**: <1s simple, <5s complex

### Hardware Optimization (M4 Pro)
- **CPU Cores**: 8P + 4E cores utilized
- **GPU**: 20 Metal cores for ML inference
- **Memory**: 24GB unified memory optimization
- **Parallelization**: Up to 12 concurrent operations

### Example Performance

```
Command: "fix authentication issue"
├─ Context Gathering: 89ms (Einstein search)
├─ Intent Analysis: 23ms (pattern matching)
├─ Action Planning: 45ms (task decomposition)
├─ Execution: 1,234ms (Bolt 4-agent workflow)
└─ Total: 1,391ms
```

## 🔍 Result Analysis

### CommandResult Structure

```python
@dataclass
class CommandResult:
    # Status and success
    status: CommandStatus
    success: bool
    
    # Results
    summary: str
    findings: List[str]
    recommendations: List[str]
    actions_taken: List[str]
    
    # Files and changes
    files_affected: List[str]
    changes_made: List[Dict[str, Any]]
    
    # Performance metrics
    metrics: CommandMetrics
    
    # Error handling
    errors: List[CommandError]
    warnings: List[str]
```

### Key Metrics

```python
# Access performance metrics
result.metrics.total_duration_ms
result.metrics.context_confidence
result.metrics.intent_confidence
result.metrics.execution_success_rate

# Check results
result.success
result.files_affected
result.recommendations
```

## 🛡️ Error Handling

### Graceful Degradation

The UAE implements multiple fallback mechanisms:

1. **Context Gathering Failures** → Use minimal context
2. **Intent Ambiguity** → Request clarification
3. **Planning Conflicts** → Use conservative approach
4. **Execution Errors** → Rollback and retry

### Error Categories

- **Processing Errors**: System-level failures
- **Intent Errors**: Ambiguous or unclear commands
- **Execution Errors**: Task execution failures
- **Resource Errors**: Hardware/memory limitations

### Example Error Handling

```python
result = await engine.process_command("ambiguous command")

if not result.success:
    for error in result.errors:
        print(f"Error: {error.error_type} - {error.error_message}")
        
    # Check for clarification needs
    if result.warnings:
        print("Suggestions:")
        for warning in result.warnings:
            print(f"  • {warning}")
```

## 🔧 Advanced Usage

### Custom Context Hints

```python
result = await engine.process_command(
    "optimize trading performance",
    context_hints={
        "search_depth": 5,
        "max_files": 100,
        "optimization_target": "speed",
        "focus_areas": ["trading", "performance", "optimization"]
    }
)
```

### Batch Processing

```python
commands = [
    "analyze authentication system",
    "review error handling patterns",
    "optimize database queries"
]

results = []
for command in commands:
    result = await engine.process_command(command)
    results.append(result)
```

### Real-time Monitoring

```python
# Get engine statistics
stats = await engine.get_engine_stats()
print(f"Commands processed: {stats['total_commands_processed']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average time: {stats['average_processing_time_ms']:.1f}ms")
```

## 🎮 Interactive Mode

Run the interactive demo to experiment with commands:

```bash
python unified_assessment_demo.py --interactive
```

```
🎯 Enter command: fix authentication issue
✅ Status: completed
🎯 Success: True
⏱️ Duration: 1.23s
📊 Intent Confidence: 0.89
🔍 Context Confidence: 0.76

📋 Summary: Successfully identified and fixed authentication issue

🔍 Findings:
  1. Found authentication module in src/unity_wheel/auth/
  2. Identified token validation issue
  3. Located failing tests in tests/test_auth.py

💡 Recommendations:
  1. Add integration tests for auth flow
  2. Implement better error messages
  3. Consider adding rate limiting
```

## 🚀 Production Deployment

### Initialization

```python
# Production configuration
production_config = {
    "context": {
        "max_files": 100,
        "search_depth": 5,
        "optimization_target": "accuracy"
    },
    "routing": {
        "bolt_agent_count": 12,
        "enable_monitoring": True
    }
}

engine = UnifiedAssessmentEngine(production_config)
await engine.initialize()
```

### Health Monitoring

```python
# Monitor engine health
stats = await engine.get_engine_stats()
if stats['success_rate'] < 0.9:
    logger.warning("Engine success rate below threshold")

if stats['average_processing_time_ms'] > 5000:
    logger.warning("Engine response time degraded")
```

### Graceful Shutdown

```python
# Always clean up resources
try:
    # Process commands
    result = await engine.process_command(command)
finally:
    await engine.shutdown()
```

## 📚 Integration Examples

### With Existing Systems

```python
# Integration with wheel trading system
from unity_wheel.api.advisor import TradingAdvisor

advisor = TradingAdvisor()
engine = UnifiedAssessmentEngine()

# Use UAE for analysis
result = await engine.process_command(
    "analyze Unity position sizing for $100k portfolio"
)

# Apply recommendations to trading system
if result.success:
    for recommendation in result.recommendations:
        # Process recommendations with advisor
        pass
```

### With CLI Tools

```python
#!/usr/bin/env python3
"""UAE CLI wrapper"""
import click
from unity_wheel.unified_assessment import UnifiedAssessmentEngine

@click.command()
@click.argument('command')
@click.option('--format', default='text', help='Output format: text, json')
async def uae_cli(command, format):
    engine = UnifiedAssessmentEngine()
    await engine.initialize()
    
    try:
        result = await engine.process_command(command)
        
        if format == 'json':
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"Result: {result.summary}")
            
    finally:
        await engine.shutdown()
```

## 🤝 Contributing

The UAE is designed to be extensible. Key extension points:

### Custom Intent Categories

```python
# Add custom intent patterns
from unity_wheel.unified_assessment.core.intent import IntentPatternMatcher

matcher = IntentPatternMatcher()
matcher.intent_patterns[CustomCategory.DEPLOY] = [
    (r'\b(deploy|release|publish)\b', 0.9),
    (r'\b(production|staging)\b', 0.7)
]
```

### Custom Task Types

```python
# Add custom task decomposition
from unity_wheel.unified_assessment.core.planning import TaskDecomposer

class CustomDecomposer(TaskDecomposer):
    def _create_custom_tasks(self, intent, context):
        # Custom task creation logic
        return tasks
```

### Custom Execution Engines

```python
# Add custom execution engine
from unity_wheel.unified_assessment.core.routing import ExecutionRouter

class CustomExecutor:
    async def execute_task(self, task, context):
        # Custom execution logic
        return ExecutionResult(...)
```

## 📈 Performance Tuning

### For Speed Optimization

```python
config = {
    "context": {"max_files": 20, "search_depth": 2},
    "planning": {"max_parallel_tasks": 12},
    "routing": {"prefer_direct_tools": True}
}
```

### For Accuracy Optimization

```python
config = {
    "context": {"max_files": 100, "search_depth": 5},
    "intent": {"confidence_threshold": 0.8},
    "routing": {"prefer_bolt_for_complex": True}
}
```

### For Resource Optimization

```python
config = {
    "planning": {"hardware_profile": "m4_pro"},
    "routing": {"bolt_agent_count": 8},
    "context": {"optimization_target": "balanced"}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Slow Context Gathering**
   - Reduce `max_files` in config
   - Lower `search_depth`
   - Check Einstein index health

2. **Low Intent Confidence**
   - Use more specific commands
   - Provide context hints
   - Check command phrasing

3. **Execution Failures**
   - Check Bolt system status
   - Verify tool availability
   - Review error logs

4. **Memory Issues**
   - Reduce parallel task count
   - Lower cache sizes
   - Monitor system resources

### Debug Mode

```python
import logging
logging.getLogger('unity_wheel.unified_assessment').setLevel(logging.DEBUG)

# Detailed execution logging
result = await engine.process_command(command)
for log_entry in result.execution_log:
    print(f"{log_entry['timestamp']}: {log_entry['step']}")
```

## 📄 License

Part of the Unity Wheel Trading System. See main project LICENSE file for details.

---

The Unified Assessment Engine represents a significant advancement in natural language command processing for trading systems, providing a seamless bridge between human intent and system execution through intelligent context gathering, intent analysis, and optimized routing.