#!/usr/bin/env python3
"""
Check what configurations will persist after computer restart.
"""

import os
import subprocess
import json
from pathlib import Path

class PersistenceChecker:
    def __init__(self):
        self.home = Path.home()
        self.results = {
            'shell_config': {},
            'system_limits': {},
            'launchers': {},
            'mcp_config': {},
            'recommendations': []
        }
    
    def check_shell_configuration(self):
        """Check if environment variables are in shell config."""
        zshrc = self.home / '.zshrc'
        bashrc = self.home / '.bashrc'
        
        for config_file in [zshrc, bashrc]:
            if config_file.exists():
                content = config_file.read_text()
                
                # Check for our optimizations
                has_node_options = 'NODE_OPTIONS' in content and 'max-old-space-size' in content
                has_claude_config = 'CLAUDE_CODE_MAX_OUTPUT_TOKENS' in content
                
                self.results['shell_config'][str(config_file)] = {
                    'exists': True,
                    'has_node_optimizations': has_node_options,
                    'has_claude_config': has_claude_config,
                    'persistent': has_node_options and has_claude_config
                }
            else:
                self.results['shell_config'][str(config_file)] = {
                    'exists': False,
                    'persistent': False
                }
    
    def check_system_limits(self):
        """Check system-level limits configuration."""
        launchd_conf = Path('/etc/launchd.conf')
        
        if launchd_conf.exists():
            try:
                content = launchd_conf.read_text()
                has_maxfiles = 'maxfiles' in content
                has_maxproc = 'maxproc' in content
                
                self.results['system_limits']['launchd_conf'] = {
                    'exists': True,
                    'has_file_limits': has_maxfiles,
                    'has_process_limits': has_maxproc,
                    'persistent': has_maxfiles and has_maxproc
                }
            except PermissionError:
                self.results['system_limits']['launchd_conf'] = {
                    'exists': True,
                    'error': 'Permission denied - run as admin to check'
                }
        else:
            self.results['system_limits']['launchd_conf'] = {
                'exists': False,
                'persistent': False
            }
    
    def check_launchers(self):
        """Check for persistent Claude launchers."""
        local_bin = self.home / '.local/bin'
        launchers = ['claude-persistent', 'claude-optimized']
        
        for launcher in launchers:
            launcher_path = local_bin / launcher
            if launcher_path.exists():
                # Check if it has memory optimizations
                content = launcher_path.read_text()
                has_optimizations = 'NODE_OPTIONS' in content and 'max-old-space-size' in content
                
                self.results['launchers'][launcher] = {
                    'exists': True,
                    'path': str(launcher_path),
                    'executable': os.access(launcher_path, os.X_OK),
                    'has_optimizations': has_optimizations,
                    'persistent': has_optimizations
                }
            else:
                self.results['launchers'][launcher] = {
                    'exists': False,
                    'persistent': False
                }
    
    def check_mcp_configuration(self):
        """Check MCP server configurations."""
        mcp_config = Path('mcp-servers.json')
        
        if mcp_config.exists():
            try:
                with open(mcp_config) as f:
                    config = json.load(f)
                
                # Check if filesystem server has memory optimizations
                filesystem_config = config.get('mcpServers', {}).get('filesystem', {})
                env_config = filesystem_config.get('env', {})
                
                has_node_options = 'NODE_OPTIONS' in env_config
                has_memory_config = any('max-old-space-size' in str(v) for v in env_config.values())
                
                self.results['mcp_config']['mcp-servers.json'] = {
                    'exists': True,
                    'has_memory_config': has_memory_config,
                    'persistent': has_memory_config
                }
            except Exception as e:
                self.results['mcp_config']['mcp-servers.json'] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            self.results['mcp_config']['mcp-servers.json'] = {
                'exists': False,
                'persistent': False
            }
    
    def check_current_environment(self):
        """Check current session environment."""
        current_node_options = os.environ.get('NODE_OPTIONS', '')
        current_claude_tokens = os.environ.get('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '')
        
        self.results['current_session'] = {
            'node_options': current_node_options,
            'claude_tokens': current_claude_tokens,
            'has_memory_optimizations': 'max-old-space-size' in current_node_options,
            'note': 'Current session only - will be lost on restart unless made permanent'
        }
    
    def generate_recommendations(self):
        """Generate recommendations for ensuring persistence."""
        recommendations = []
        
        # Check shell config
        shell_persistent = any(
            config.get('persistent', False) 
            for config in self.results['shell_config'].values()
        )
        
        if not shell_persistent:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Add environment variables to ~/.zshrc',
                'command': './scripts/make-permanent.sh',
                'reason': 'Environment variables will be lost on restart'
            })
        
        # Check launchers
        launcher_persistent = any(
            config.get('persistent', False)
            for config in self.results['launchers'].values()
        )
        
        if not launcher_persistent:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Create persistent Claude launcher',
                'command': 'Run make-permanent.sh to create claude-persistent',
                'reason': 'No optimized Claude launcher found'
            })
        
        # Check system limits
        if not self.results['system_limits'].get('launchd_conf', {}).get('persistent', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Configure system limits',
                'command': 'sudo ./scripts/make-permanent.sh',
                'reason': 'System limits need admin configuration'
            })
        
        self.results['recommendations'] = recommendations
    
    def run_check(self):
        """Run all persistence checks."""
        print("🔍 Checking configuration persistence...")
        
        self.check_shell_configuration()
        self.check_system_limits()
        self.check_launchers()
        self.check_mcp_configuration()
        self.check_current_environment()
        self.generate_recommendations()
        
        return self.results
    
    def print_results(self):
        """Print human-readable results."""
        print("\n" + "="*60)
        print("📋 PERSISTENCE CHECK RESULTS")
        print("="*60)
        
        # Shell Configuration
        print("\n🐚 Shell Configuration:")
        for config_file, status in self.results['shell_config'].items():
            if status.get('exists'):
                persistent = "✅ PERSISTENT" if status.get('persistent') else "❌ NOT PERSISTENT"
                print(f"  {Path(config_file).name}: {persistent}")
            else:
                print(f"  {Path(config_file).name}: ❌ MISSING")
        
        # Launchers
        print("\n🚀 Claude Launchers:")
        for launcher, status in self.results['launchers'].items():
            if status.get('exists'):
                persistent = "✅ PERSISTENT" if status.get('persistent') else "❌ NOT OPTIMIZED"
                print(f"  {launcher}: {persistent}")
            else:
                print(f"  {launcher}: ❌ MISSING")
        
        # Current Session
        print("\n💻 Current Session:")
        current = self.results['current_session']
        if current['has_memory_optimizations']:
            print("  ✅ Memory optimizations active (temporary)")
        else:
            print("  ❌ No memory optimizations active")
        
        # Recommendations
        if self.results['recommendations']:
            print("\n⚠️  ACTIONS NEEDED FOR PERSISTENCE:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"  {i}. [{rec['priority']}] {rec['action']}")
                print(f"     Command: {rec['command']}")
                print(f"     Reason: {rec['reason']}\n")
        else:
            print("\n✅ ALL CONFIGURATIONS ARE PERSISTENT!")
        
        print("="*60)

def main():
    checker = PersistenceChecker()
    results = checker.run_check()
    checker.print_results()
    
    # Return exit code based on persistence status
    has_persistent_config = any(
        config.get('persistent', False)
        for config in results['shell_config'].values()
    )
    
    return 0 if has_persistent_config else 1

if __name__ == '__main__':
    exit(main())