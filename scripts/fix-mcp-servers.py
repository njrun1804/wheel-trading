#!/usr/bin/env python3
"""
Systematic MCP Server Fixer

This script categorizes MCP servers and applies common fixes based on failure patterns.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

class MCPServerFixer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.load_config()
        
    def load_config(self):
        """Load MCP server configuration."""
        with open(self.config_path) as f:
            self.config = json.load(f)
        self.servers = self.config.get('mcpServers', {})
        
    def categorize_servers(self) -> Dict[str, List[str]]:
        """Categorize servers by type."""
        categories = {
            'NPX servers': [],
            'Python modules': [],
            'Python scripts': [],
            'Binary/Other': []
        }
        
        for name, server_config in self.servers.items():
            command = server_config.get('command', '')
            args = server_config.get('args', [])
            
            if command == 'npx':
                categories['NPX servers'].append(name)
            elif command.endswith('python3') and '-m' in args:
                categories['Python modules'].append(name)
            elif command.endswith('python3') and args and args[0].endswith('.py'):
                categories['Python scripts'].append(name)
            else:
                categories['Binary/Other'].append(name)
        
        return categories
    
    def fix_npx_servers(self, servers: List[str]) -> List[Tuple[str, bool, str]]:
        """Fix NPX-based servers."""
        results = []
        
        print(f"\n{BLUE}Fixing NPX servers...{NC}")
        
        for server in servers:
            config = self.servers[server]
            args = config.get('args', [])
            
            # Common fixes for NPX servers
            fixes_applied = []
            
            # 1. Ensure -y flag for auto-install
            if '-y' not in args:
                args.insert(0, '-y')
                fixes_applied.append("Added -y flag")
            
            # 2. Ensure @latest tag
            for i, arg in enumerate(args):
                if arg.startswith('@modelcontextprotocol/') and not arg.endswith('@latest'):
                    args[i] = arg + '@latest'
                    fixes_applied.append("Added @latest tag")
            
            # 3. Check if npx is available
            if not self._command_exists('npx'):
                results.append((server, False, "npx not found - install Node.js"))
                continue
            
            if fixes_applied:
                self.servers[server]['args'] = args
                results.append((server, True, f"Fixed: {', '.join(fixes_applied)}"))
            else:
                results.append((server, True, "No fixes needed"))
        
        return results
    
    def fix_python_modules(self, servers: List[str]) -> List[Tuple[str, bool, str]]:
        """Fix Python module-based servers."""
        results = []
        
        print(f"\n{BLUE}Fixing Python module servers...{NC}")
        
        for server in servers:
            config = self.servers[server]
            args = config.get('args', [])
            
            # Find module name
            module_idx = args.index('-m') + 1 if '-m' in args else -1
            if module_idx <= 0 or module_idx >= len(args):
                results.append((server, False, "Invalid module configuration"))
                continue
            
            module_name = args[module_idx]
            
            # Special case: duckdb server
            if server == 'duckdb' and module_name == 'mcp_server_duckdb':
                # Check if console script exists
                duckdb_cmd = self._find_command('mcp-server-duckdb')
                if duckdb_cmd:
                    # Update to use console script
                    self.servers[server]['command'] = duckdb_cmd
                    # Fix arguments - need --db-path flag
                    if len(args) > module_idx + 1:
                        db_path = args[module_idx + 1]
                        self.servers[server]['args'] = ['--db-path', db_path]
                    else:
                        self.servers[server]['args'] = []
                    results.append((server, True, "Fixed: Using console script with --db-path"))
                else:
                    results.append((server, False, "mcp-server-duckdb not installed"))
                continue
            
            # Check if module is installed
            if self._check_python_module(module_name):
                results.append((server, True, "Module installed"))
            else:
                # Try to install
                install_name = module_name.replace('_', '-')
                results.append((server, False, f"Module not found - run: pip install {install_name}"))
        
        return results
    
    def fix_python_scripts(self, servers: List[str]) -> List[Tuple[str, bool, str]]:
        """Fix Python script-based servers."""
        results = []
        
        print(f"\n{BLUE}Fixing Python script servers...{NC}")
        
        for server in servers:
            config = self.servers[server]
            args = config.get('args', [])
            
            if not args:
                results.append((server, False, "No script specified"))
                continue
            
            script_path = Path(args[0])
            
            # Check if script exists
            if not script_path.exists():
                results.append((server, False, f"Script not found: {script_path}"))
                continue
            
            # Check if script is executable (has shebang)
            with open(script_path, 'r') as f:
                first_line = f.readline()
                if not first_line.startswith('#!'):
                    # Add shebang if missing
                    content = first_line + f.read()
                    with open(script_path, 'w') as fw:
                        fw.write('#!/usr/bin/env python3\n' + content)
                    results.append((server, True, "Added shebang to script"))
                else:
                    results.append((server, True, "Script OK"))
        
        return results
    
    def fix_binary_servers(self, servers: List[str]) -> List[Tuple[str, bool, str]]:
        """Fix binary/other servers."""
        results = []
        
        print(f"\n{BLUE}Fixing binary/other servers...{NC}")
        
        for server in servers:
            config = self.servers[server]
            command = config.get('command', '')
            
            # Check if command exists
            if self._command_exists(command):
                results.append((server, True, "Command found"))
            else:
                # Try to find it
                found_cmd = self._find_command(os.path.basename(command))
                if found_cmd:
                    self.servers[server]['command'] = found_cmd
                    results.append((server, True, f"Fixed: Updated path to {found_cmd}"))
                else:
                    results.append((server, False, f"Command not found: {command}"))
        
        return results
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists."""
        return subprocess.run(['which', command], 
                            capture_output=True, 
                            text=True).returncode == 0
    
    def _find_command(self, command: str) -> str:
        """Find command in PATH."""
        result = subprocess.run(['which', command], 
                              capture_output=True, 
                              text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    
    def _check_python_module(self, module: str) -> bool:
        """Check if a Python module is installed."""
        result = subprocess.run(
            [sys.executable, '-c', f'import {module}'],
            capture_output=True
        )
        return result.returncode == 0
    
    def save_config(self):
        """Save updated configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\n{GREEN}Configuration saved to {self.config_path}{NC}")
    
    def run(self):
        """Run the fixer."""
        print(f"{GREEN}=== MCP Server Fixer ==={NC}")
        print(f"Analyzing {len(self.servers)} servers...")
        
        categories = self.categorize_servers()
        all_results = []
        
        # Fix each category
        for category, servers in categories.items():
            if not servers:
                continue
            
            if category == 'NPX servers':
                results = self.fix_npx_servers(servers)
            elif category == 'Python modules':
                results = self.fix_python_modules(servers)
            elif category == 'Python scripts':
                results = self.fix_python_scripts(servers)
            else:
                results = self.fix_binary_servers(servers)
            
            all_results.extend(results)
        
        # Summary
        print(f"\n{YELLOW}=== Summary ==={NC}")
        fixed = sum(1 for _, success, _ in all_results if success)
        failed = len(all_results) - fixed
        
        print(f"Total servers: {len(all_results)}")
        print(f"{GREEN}Fixed/OK: {fixed}{NC}")
        print(f"{RED}Failed: {failed}{NC}")
        
        if failed > 0:
            print(f"\n{RED}Failed servers:{NC}")
            for server, success, message in all_results:
                if not success:
                    print(f"  - {server}: {message}")
        
        # Save if any changes were made
        if any("Fixed" in msg for _, _, msg in all_results):
            self.save_config()
        
        return failed == 0


def main():
    """Main entry point."""
    config_path = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
    
    fixer = MCPServerFixer(config_path)
    success = fixer.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()