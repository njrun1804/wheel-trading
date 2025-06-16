# BOB CLI - Natural Language Interface for Wheel Trading

BOB CLI provides a natural language interface to the wheel trading system, allowing developers to execute complex operations using simple English commands.

## Features

- **Natural Language Processing**: Understands commands like "fix the authentication issue" or "create a new trading strategy"
- **Interactive Mode**: REPL with tab completion, command history, and guided workflows
- **Context Awareness**: Maintains context between commands in interactive mode
- **Integration**: Seamlessly integrates with Einstein search and Bolt multi-agent system
- **Help System**: Comprehensive help with examples and suggestions

## Installation

The BOB CLI is already integrated into the wheel trading system. No additional installation needed.

## Usage

### Basic Commands

```bash
# Execute a single command
python bob_cli.py "fix the authentication issue in storage.py"
python bob_cli.py "create a new trading strategy for Unity"
python bob_cli.py "optimize wheel performance parameters"

# Interactive mode
python bob_cli.py --interactive

# Get help
python bob_cli.py help
python bob_cli.py help strategies
```

### Command Types

BOB understands these types of commands:

- **fix**: Fix issues, bugs, and errors
- **create**: Create new components and strategies
- **optimize**: Optimize performance and parameters
- **analyze**: Analyze code and trading patterns
- **run**: Execute operations and workflows
- **test**: Run tests and validate functionality
- **deploy**: Deploy changes and releases
- **monitor**: Monitor performance and health

### Interactive Mode

Interactive mode provides a REPL with advanced features:

```bash
bob> fix authentication error
✅ Searching codebase for authentication error

bob> workflow
🔄 Available Workflows:
1. Fix Code Issues
2. Create New Component
3. Optimize Performance
4. Run Analysis
5. Deploy Changes

bob> context
📋 Current Context:
current_file: storage.py
recent_commands:
  - fix authentication error

bob> help interactive
📚 Interactive Mode
==================
Using BOB's interactive REPL...
```

### Guided Workflows

Interactive mode includes guided workflows for complex tasks:

1. **Fix Code Issues**: Step-by-step issue resolution
2. **Create New Component**: Template-based component creation
3. **Optimize Performance**: Performance analysis and tuning
4. **Run Analysis**: Comprehensive system analysis
5. **Deploy Changes**: Safe deployment with validation

### Integration with Wheel Trading

BOB CLI integrates with the existing wheel trading infrastructure:

- Uses Einstein semantic search for code understanding
- Leverages Bolt multi-agent system for complex tasks
- Integrates with existing run.py functionality
- Respects system configuration and risk limits

## Examples

### Fixing Issues
```bash
bob "fix the database connection timeout"
bob "resolve import errors in risk module"
bob "repair broken unit tests"
```

### Creating Components
```bash
bob "create a new options pricing model"
bob "build a risk management dashboard"
bob "generate tests for wheel strategy"
```

### Optimization
```bash
bob "optimize Unity position sizing"
bob "improve query performance"
bob "tune risk parameters for aggressive trading"
```

### Analysis
```bash
bob "analyze Unity trading patterns"
bob "review last month's performance"
bob "check system health metrics"
```

## Advanced Features

### Dry Run Mode
Preview what would happen without executing:
```bash
python bob_cli.py --dry-run "deploy latest changes"
```

### Verbose Output
Enable detailed logging:
```bash
python bob_cli.py --verbose "analyze system performance"
```

### Command Suggestions
BOB provides intelligent command suggestions based on:
- Partial command matching
- Context from previous commands
- Common workflow patterns

## Architecture

The BOB CLI consists of:

1. **bob_cli.py**: Main entry point and CLI interface
2. **bob/cli/processor.py**: Natural language processing and command routing
3. **bob/cli/interactive.py**: Interactive REPL implementation
4. **bob/cli/help.py**: Help system and command suggestions

## Testing

Run the test suite to verify functionality:
```bash
python test_bob_cli.py
```

## Future Enhancements

- Direct integration with Claude API for enhanced NLP
- Machine learning for command pattern recognition
- Custom workflow definitions
- Plugin system for extending commands