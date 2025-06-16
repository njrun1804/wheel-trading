#!/usr/bin/env python3
"""
BOB CLI - Natural Language Interface for Wheel Trading System

A developer-friendly CLI that understands natural language commands
and executes appropriate actions in the wheel trading ecosystem.
"""

import argparse
import sys
import os
from typing import Optional, List
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bob.cli.processor import CommandProcessor
from bob.cli.interactive import InteractiveMode
from bob.cli.help import HelpSystem
from src.unity_wheel.utils.logging import get_logger

logger = get_logger(__name__)


class BobCLI:
    """Main CLI interface for BOB commands"""
    
    def __init__(self):
        self.processor = CommandProcessor()
        self.help_system = HelpSystem()
        self.interactive = InteractiveMode(self.processor, self.help_system)
        
    def execute_command(self, command: str) -> int:
        """Execute a single natural language command"""
        try:
            # Preprocess and validate command
            processed_command = self.processor.preprocess(command)
            
            if not processed_command:
                print("❌ Could not understand the command. Try 'bob help' for examples.")
                return 1
                
            # Execute the command
            result = self.processor.execute(processed_command)
            
            if result.success:
                print(f"✅ {result.message}")
                if result.details:
                    print(f"\n{result.details}")
                return 0
            else:
                print(f"❌ {result.message}")
                if result.suggestions:
                    print("\n💡 Suggestions:")
                    for suggestion in result.suggestions:
                        print(f"   - {suggestion}")
                return 1
                
        except KeyboardInterrupt:
            print("\n👋 Command cancelled.")
            return 130
        except Exception as e:
            logger.error(f"Command execution failed: {e}", exc_info=True)
            print(f"❌ Unexpected error: {str(e)}")
            return 1
            
    def run_interactive(self) -> int:
        """Run in interactive mode"""
        print("🤖 BOB Interactive Mode")
        print("   Type 'help' for commands, 'exit' to quit")
        print("-" * 50)
        
        try:
            return self.interactive.run()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
            
    def show_help(self, topic: Optional[str] = None) -> int:
        """Show help information"""
        if topic:
            help_text = self.help_system.get_topic_help(topic)
        else:
            help_text = self.help_system.get_general_help()
            
        print(help_text)
        return 0


def main():
    """Main entry point for BOB CLI"""
    parser = argparse.ArgumentParser(
        description="BOB - Natural language interface for wheel trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bob "fix the authentication issue in storage.py"
  bob "create a new trading strategy for Unity"
  bob "optimize wheel performance parameters"
  bob --interactive
  bob help strategies
        """
    )
    
    parser.add_argument(
        "command",
        nargs="*",
        help="Natural language command to execute"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Configure environment
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    if args.dry_run:
        os.environ["BOB_DRY_RUN"] = "1"
        
    # Initialize CLI
    cli = BobCLI()
    
    # Handle different modes
    if args.interactive:
        return cli.run_interactive()
    elif args.command:
        # Join command parts into single string
        command = " ".join(args.command)
        
        # Special handling for help
        if command.lower().startswith("help"):
            parts = command.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else None
            return cli.show_help(topic)
            
        return cli.execute_command(command)
    else:
        # No command provided, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())