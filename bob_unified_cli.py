#!/usr/bin/env python3
"""
BOB Unified CLI - Natural Language Interface for Wheel Trading System

A comprehensive command-line interface that integrates Einstein semantic search,
BOLT multi-agent orchestration, and the Unity wheel trading system into a
unified natural language interface.

Usage:
    bob                                    # Interactive mode
    bob "optimize trading performance"     # Direct command
    bob analyze "portfolio risk"           # Categorized command
    bob --help                            # Show help
"""

import argparse
import asyncio
import cmd
import json
import os
import readline
import shlex
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core imports with fallback handling
try:
    from bolt.core.integration import BoltIntegration
    HAS_BOLT = True
except ImportError as e:
    print(f"Warning: BOLT integration not available: {e}")
    HAS_BOLT = False

try:
    from einstein.unified_index import EinsteinIndexHub
    HAS_EINSTEIN = True
except ImportError as e:
    print(f"Warning: Einstein search not available: {e}")
    HAS_EINSTEIN = False

try:
    from unity_wheel.cli.run import main as trading_main
    HAS_TRADING = True
except ImportError as e:
    print(f"Warning: Trading system not available: {e}")
    HAS_TRADING = False


class BobConfig:
    """Configuration management for BOB CLI."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".bob"
        self.config_file = self.config_dir / "config.yaml"
        self.session_dir = self.config_dir / "sessions"
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults."""
        if not self.config_file.exists():
            return self.create_default_config()
            
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Error loading config: {e}")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        config = {
            'system': {
                'performance_mode': 'maximum',
                'cpu_cores': 12,
                'memory_limit_gb': 20,
                'gpu_acceleration': True,
            },
            'einstein': {
                'cache_size_mb': 2048,
                'max_results': 50,
                'semantic_threshold': 0.7,
            },
            'bolt': {
                'agents': 8,
                'task_batch_size': 16,
                'work_stealing': True,
                'priority_scheduling': True,
            },
            'trading': {
                'default_symbol': 'U',
                'max_position': 100000,
                'max_delta': 0.30,
                'risk_check_enabled': True,
                'paper_trade_default': False,
            },
            'ui': {
                'color_output': True,
                'progress_bars': True,
                'notifications': True,
                'auto_save_session': True,
            }
        }
        
        # Ensure directories exist and save config
        self.config_dir.mkdir(exist_ok=True)
        self.session_dir.mkdir(exist_ok=True)
        self.save_config(config)
        return config
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
            
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
        self.save_config()


class CommandProcessor:
    """Natural language command processor."""
    
    def __init__(self, config: BobConfig):
        self.config = config
        self.context = {}
        self.session_id = f"bob-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    def parse_command(self, command: str) -> Tuple[str, str, Dict[str, Any]]:
        """Parse natural language command into action, target, and options."""
        command = command.strip()
        
        # Handle quoted commands
        if command.startswith('"') and command.endswith('"'):
            command = command[1:-1]
        
        # Extract action words
        action_words = {
            'analyze': ['analyze', 'analysis', 'review', 'examine', 'check', 'investigate'],
            'optimize': ['optimize', 'improve', 'enhance', 'tune', 'speed up', 'accelerate'],
            'fix': ['fix', 'repair', 'resolve', 'debug', 'solve', 'correct'],
            'create': ['create', 'generate', 'build', 'make', 'add', 'implement'],
            'test': ['test', 'validate', 'verify', 'check'],
            'monitor': ['monitor', 'watch', 'track', 'observe'],
            'deploy': ['deploy', 'release', 'publish', 'launch'],
            'help': ['help', 'explain', 'how', 'what', 'why'],
        }
        
        # Find action
        action = 'analyze'  # default
        for act, words in action_words.items():
            if any(word in command.lower() for word in words):
                action = act
                break
        
        # Extract target (everything after action words)
        target = command
        for word in action_words.get(action, []):
            if word in command.lower():
                idx = command.lower().find(word)
                if idx >= 0:
                    target = command[idx + len(word):].strip()
                    break
        
        # Extract options (basic pattern matching)
        options = {
            'verbose': any(word in command.lower() for word in ['verbose', 'detailed', 'full']),
            'quick': any(word in command.lower() for word in ['quick', 'fast', 'brief']),
            'dry_run': any(word in command.lower() for word in ['dry run', 'preview', 'simulate']),
        }
        
        return action, target, options


class SessionManager:
    """Session and context management."""
    
    def __init__(self, config: BobConfig):
        self.config = config
        self.session_dir = config.session_dir
        self.current_session = None
        self.context = {}
        
    def save_session(self, name: str, context: Dict[str, Any]):
        """Save current session to file."""
        session_file = self.session_dir / f"{name}.json"
        
        session_data = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'config_snapshot': self.config.config,
        }
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self, name: str) -> Dict[str, Any]:
        """Load session from file."""
        session_file = self.session_dir / f"{name}.json"
        
        if not session_file.exists():
            return {}
            
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}
    
    def list_sessions(self) -> List[str]:
        """List available sessions."""
        if not self.session_dir.exists():
            return []
            
        sessions = []
        for file in self.session_dir.glob("*.json"):
            sessions.append(file.stem)
        
        return sorted(sessions, reverse=True)


class ProgressReporter:
    """Progress reporting and user feedback."""
    
    def __init__(self, config: BobConfig):
        self.config = config
        self.use_colors = config.get('ui.color_output', True)
        self.show_progress = config.get('ui.progress_bars', True)
        
    def info(self, message: str):
        """Display info message."""
        if self.use_colors:
            print(f"\033[36mℹ️  {message}\033[0m")
        else:
            print(f"ℹ️  {message}")
    
    def success(self, message: str):
        """Display success message."""
        if self.use_colors:
            print(f"\033[32m✅ {message}\033[0m")
        else:
            print(f"✅ {message}")
    
    def warning(self, message: str):
        """Display warning message."""
        if self.use_colors:
            print(f"\033[33m⚠️  {message}\033[0m")
        else:
            print(f"⚠️  {message}")
    
    def error(self, message: str):
        """Display error message."""
        if self.use_colors:
            print(f"\033[31m❌ {message}\033[0m")
        else:
            print(f"❌ {message}")
    
    def progress(self, message: str, percent: int = None):
        """Display progress message."""
        if percent is not None and self.show_progress:
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r🔄 {message} [{bar}] {percent}%", end="", flush=True)
        else:
            print(f"🔄 {message}")


class BobCLI(cmd.Cmd):
    """Interactive CLI for BOB system."""
    
    intro = """
🤖 BOB (Bolt Orchestrator Bootstrap) CLI v2.0
Hardware: M4 Pro (12 cores) | Memory: 24GB | GPU: 20 cores
Einstein: Ready | BOLT: 8 agents | Trading: Connected

Type 'help' for commands or 'tutorial' for guided tour.
Use 'exit' or Ctrl+D to quit.
"""
    
    prompt = "bob> "
    
    def __init__(self):
        super().__init__()
        self.config = BobConfig()
        self.processor = CommandProcessor(self.config)
        self.session_manager = SessionManager(self.config)
        self.reporter = ProgressReporter(self.config)
        self.context = {}
        
        # Initialize components
        self.integration = None
        self.einstein_index = None
        
    async def async_init(self):
        """Async initialization of components."""
        try:
            if HAS_BOLT:
                self.integration = BoltIntegration(
                    num_agents=self.config.get('bolt.agents', 8)
                )
                await self.integration.initialize()
                
            if HAS_EINSTEIN:
                self.einstein_index = EinsteinIndexHub()
                await self.einstein_index.initialize()
                
        except Exception as e:
            self.reporter.warning(f"Component initialization failed: {e}")
    
    def do_analyze(self, arg):
        """Analyze code, trading, or system components."""
        if not arg:
            print("Usage: analyze <target>")
            return
            
        asyncio.run(self._analyze(arg))
    
    async def _analyze(self, target: str):
        """Perform analysis."""
        self.reporter.info(f"Analyzing: {target}")
        
        # Use Einstein search if available
        if self.einstein_index:
            self.reporter.progress("Einstein search in progress...", 25)
            results = await self.einstein_index.search(target, max_results=10)
            self.reporter.progress("Einstein search complete", 100)
            print()
            
            if results:
                print(f"📊 Found {len(results)} relevant files:")
                for i, result in enumerate(results[:5], 1):
                    print(f"  {i}. {result.file_path} (score: {result.score:.3f})")
            else:
                print("🔍 No results found")
        
        # Use BOLT for deeper analysis
        if self.integration:
            self.reporter.progress("BOLT analysis in progress...", 50)
            try:
                result = await self.integration.analyze_query(target)
                self.reporter.progress("BOLT analysis complete", 100)
                print()
                
                if result.get('tasks'):
                    print(f"📋 Planned tasks: {len(result['tasks'])}")
                    for i, task in enumerate(result['tasks'][:3], 1):
                        print(f"  {i}. {task.get('description', 'Unknown task')}")
                        
            except Exception as e:
                self.reporter.error(f"BOLT analysis failed: {e}")
    
    def do_optimize(self, arg):
        """Optimize performance, trading, or system components."""
        if not arg:
            print("Usage: optimize <target>")
            return
            
        asyncio.run(self._optimize(arg))
    
    async def _optimize(self, target: str):
        """Perform optimization."""
        self.reporter.info(f"Optimizing: {target}")
        
        if self.integration:
            try:
                result = await self.integration.solve(
                    f"optimize {target}", 
                    analyze_only=False
                )
                
                if result.get('success'):
                    self.reporter.success("Optimization completed")
                    
                    synthesis = result.get('results', {})
                    if synthesis.get('recommendations'):
                        print("\n💡 Recommendations:")
                        for rec in synthesis['recommendations'][:3]:
                            print(f"  • {rec}")
                else:
                    self.reporter.error("Optimization failed")
                    
            except Exception as e:
                self.reporter.error(f"Optimization error: {e}")
    
    def do_fix(self, arg):
        """Fix issues, bugs, or problems."""
        if not arg:
            print("Usage: fix <issue>")
            return
            
        asyncio.run(self._fix(arg))
    
    async def _fix(self, issue: str):
        """Fix issues."""
        self.reporter.info(f"Fixing: {issue}")
        
        if self.integration:
            try:
                result = await self.integration.solve(
                    f"fix {issue}",
                    analyze_only=False
                )
                
                if result.get('success'):
                    self.reporter.success("Fix applied successfully")
                else:
                    self.reporter.error("Fix failed")
                    
            except Exception as e:
                self.reporter.error(f"Fix error: {e}")
    
    def do_create(self, arg):
        """Create new components or features."""
        if not arg:
            print("Usage: create <component>")
            return
            
        asyncio.run(self._create(arg))
    
    async def _create(self, component: str):
        """Create components."""
        self.reporter.info(f"Creating: {component}")
        
        if self.integration:
            try:
                result = await self.integration.solve(
                    f"create {component}",
                    analyze_only=False
                )
                
                if result.get('success'):
                    self.reporter.success("Component created successfully")
                else:
                    self.reporter.error("Creation failed")
                    
            except Exception as e:
                self.reporter.error(f"Creation error: {e}")
    
    def do_config(self, arg):
        """Configuration management."""
        if not arg:
            print("Usage: config <show|set|get> [key] [value]")
            return
            
        parts = shlex.split(arg)
        command = parts[0]
        
        if command == "show":
            print("📋 Current Configuration:")
            print(yaml.dump(self.config.config, default_flow_style=False, indent=2))
            
        elif command == "get" and len(parts) > 1:
            key = parts[1]
            value = self.config.get(key)
            print(f"{key}: {value}")
            
        elif command == "set" and len(parts) > 2:
            key = parts[1]
            value = parts[2]
            
            # Try to parse as appropriate type
            try:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
            except:
                pass  # Keep as string
                
            self.config.set(key, value)
            print(f"✅ Set {key} = {value}")
            
        else:
            print("Usage: config <show|set|get> [key] [value]")
    
    def do_session(self, arg):
        """Session management."""
        if not arg:
            print("Usage: session <save|load|list|clear> [name]")
            return
            
        parts = shlex.split(arg)
        command = parts[0]
        
        if command == "save" and len(parts) > 1:
            name = parts[1]
            if self.session_manager.save_session(name, self.context):
                self.reporter.success(f"Session saved: {name}")
            else:
                self.reporter.error("Failed to save session")
                
        elif command == "load" and len(parts) > 1:
            name = parts[1]
            session_data = self.session_manager.load_session(name)
            if session_data:
                self.context = session_data.get('context', {})
                self.reporter.success(f"Session loaded: {name}")
            else:
                self.reporter.error("Failed to load session")
                
        elif command == "list":
            sessions = self.session_manager.list_sessions()
            if sessions:
                print("📚 Available Sessions:")
                for i, session in enumerate(sessions, 1):
                    print(f"  {i}. {session}")
            else:
                print("No sessions found")
                
        elif command == "clear":
            self.context.clear()
            self.reporter.success("Context cleared")
            
        else:
            print("Usage: session <save|load|list|clear> [name]")
    
    def do_context(self, arg):
        """Show current context."""
        print("📋 Current Context:")
        print(f"  Session ID: {self.processor.session_id}")
        print(f"  Active Files: {len(self.context.get('files', []))}")
        print(f"  Recent Commands: {len(self.context.get('history', []))}")
        
        if self.context.get('focus'):
            print(f"  Focus: {self.context['focus']}")
    
    def do_tutorial(self, arg):
        """Start interactive tutorial."""
        print("""
📚 BOB CLI Tutorial

BOB understands natural language commands. Here are some examples:

🔍 Analysis:
  analyze "trading performance for Unity"
  analyze "memory usage in options pricing"

⚡ Optimization:
  optimize "database query performance"
  optimize "wheel strategy parameters"

🔧 Fixing Issues:
  fix "authentication errors"
  fix "slow startup times"

🏗️  Creating Components:
  create "new risk monitoring dashboard"
  create "volatility prediction model"

🎛️  Configuration:
  config show
  config set trading.max_position 150000

💾 Session Management:
  session save "my-analysis-session"
  session load "my-analysis-session"
  
Try typing: analyze "options pricing accuracy"
""")
    
    def default(self, line):
        """Handle natural language commands."""
        if line.strip():
            asyncio.run(self._process_natural_language(line))
    
    async def _process_natural_language(self, command: str):
        """Process natural language command."""
        action, target, options = self.processor.parse_command(command)
        
        self.reporter.info(f"Processing: {command}")
        
        # Route to appropriate handler
        if action == 'analyze':
            await self._analyze(target)
        elif action == 'optimize':
            await self._optimize(target)
        elif action == 'fix':
            await self._fix(target)
        elif action == 'create':
            await self._create(target)
        elif action == 'help':
            self.do_help("")
        else:
            # Generic processing
            if self.integration:
                try:
                    result = await self.integration.solve(command, analyze_only=True)
                    
                    synthesis = result.get('results', {})
                    if synthesis.get('summary'):
                        print(f"\n📝 Summary: {synthesis['summary']}")
                        
                    if synthesis.get('recommendations'):
                        print("\n💡 Recommendations:")
                        for rec in synthesis['recommendations'][:3]:
                            print(f"  • {rec}")
                            
                except Exception as e:
                    self.reporter.error(f"Processing error: {e}")
            else:
                self.reporter.warning("Advanced processing not available")
    
    def do_exit(self, arg):
        """Exit BOB CLI."""
        print("👋 Goodbye!")
        return True
    
    def do_EOF(self, arg):
        """Handle Ctrl+D."""
        print("\n👋 Goodbye!")
        return True


async def execute_command(command: str, options: Dict[str, Any]) -> bool:
    """Execute a single command non-interactively."""
    config = BobConfig()
    reporter = ProgressReporter(config)
    
    try:
        # Initialize integration
        integration = None
        if HAS_BOLT:
            integration = BoltIntegration(num_agents=config.get('bolt.agents', 8))
            await integration.initialize()
        
        # Process command
        if integration:
            reporter.info(f"Executing: {command}")
            
            result = await integration.solve(
                command, 
                analyze_only=options.get('dry_run', False)
            )
            
            if result.get('success'):
                reporter.success("Command completed successfully")
                
                synthesis = result.get('results', {})
                if synthesis.get('summary'):
                    print(f"\n📝 Summary: {synthesis['summary']}")
                    
                if synthesis.get('recommendations'):
                    print("\n💡 Recommendations:")
                    for rec in synthesis['recommendations'][:5]:
                        print(f"  • {rec}")
                        
                return True
            else:
                reporter.error("Command failed")
                return False
        else:
            reporter.error("BOLT integration not available")
            return False
            
    except Exception as e:
        reporter.error(f"Command execution failed: {e}")
        if options.get('debug'):
            traceback.print_exc()
        return False
    
    finally:
        if integration:
            await integration.shutdown()


def main():
    """Main entry point for BOB CLI."""
    parser = argparse.ArgumentParser(
        description="BOB - Unified Natural Language CLI for Wheel Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bob                                    # Interactive mode
  bob "optimize trading performance"     # Direct command
  bob analyze "portfolio risk"           # Categorized command
  bob --interactive                      # Force interactive mode
  bob --config /path/to/config.yaml     # Custom config
  
Interactive Commands:
  analyze "Unity wheel strategy performance"
  optimize "database query speed"
  fix "authentication timeout issues"
  create "new volatility surface model"
  config show
  session save "my-session"
  tutorial
        """
    )
    
    parser.add_argument(
        'command', 
        nargs='?', 
        help='Natural language command to execute'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Show what would be done without executing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    
    args = parser.parse_args()
    
    # Handle version
    if args.version:
        print("BOB (Bolt Orchestrator Bootstrap) CLI v2.0")
        print("Natural Language Interface for Wheel Trading System")
        print("Features: Einstein Search, BOLT Orchestration, M4 Pro Acceleration")
        return 0
    
    # Handle interactive mode or no arguments
    if args.interactive or not args.command:
        try:
            cli = BobCLI()
            # Initialize async components
            asyncio.run(cli.async_init())
            cli.cmdloop()
            return 0
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"Error starting interactive mode: {e}")
            return 1
    
    # Handle direct command execution
    if args.command:
        try:
            options = {
                'dry_run': args.dry_run,
                'verbose': args.verbose,
                'debug': args.debug,
            }
            
            success = asyncio.run(execute_command(args.command, options))
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print("\n⚠️  Command cancelled by user")
            return 130
        except Exception as e:
            print(f"Error executing command: {e}")
            if args.debug:
                traceback.print_exc()
            return 1
    
    # Fallback - show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())