#!/usr/bin/env python3
"""
WezTerm Key Binding Conflict Analysis
Analyzes .wezterm.lua for conflicts with macOS, Claude Code CLI, and terminal applications
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class KeyBindingAnalyzer:
    def __init__(self):
        self.wezterm_bindings = {}
        self.conflicts = {
            'macos_system': [],
            'claude_code_cli': [],
            'terminal_apps': [],
            'internal_wezterm': [],
            'accessibility': []
        }
        
        # Define known conflict sources
        self.macos_system_shortcuts = {
            'CMD+T': 'New Tab (Safari, Chrome, most apps)',
            'CMD+W': 'Close Window/Tab (System-wide)',
            'CMD+N': 'New Window (System-wide)',
            'CMD+Q': 'Quit Application (System-wide)',
            'CMD+M': 'Minimize Window (System-wide)',
            'CMD+H': 'Hide Application (System-wide)',
            'CMD+,': 'Preferences (System-wide)',
            'CMD+Space': 'Spotlight Search (System)',
            'CMD+Tab': 'Application Switcher (System)',
            'CMD+`': 'Cycle Windows (System)',
            'CMD+1-9': 'Tab switching (many apps)',
            'CMD+A': 'Select All (System-wide)',
            'CMD+C': 'Copy (System-wide)',
            'CMD+V': 'Paste (System-wide)',
            'CMD+X': 'Cut (System-wide)',
            'CMD+Z': 'Undo (System-wide)',
            'CMD+Y': 'Redo (System-wide)',
            'CMD+F': 'Find (System-wide)',
            'CMD+G': 'Find Next (System-wide)',
            'CMD+SHIFT+G': 'Find Previous (System-wide)',
            'CMD+R': 'Refresh (many apps)',
            'CMD+S': 'Save (System-wide)',
            'CMD+O': 'Open (System-wide)',
            'CMD+P': 'Print (System-wide)',
            'CMD+L': 'Address Bar (browsers)',
            'CMD+ENTER': 'Fullscreen (many apps)',
            'CMD+SHIFT+Enter': 'New line without send (messaging apps)',
        }
        
        self.claude_code_cli_shortcuts = {
            'CTRL+C': 'Interrupt/Cancel',
            'CTRL+D': 'EOF/Exit',
            'CTRL+Z': 'Suspend process',
            'CTRL+L': 'Clear screen',
            'CTRL+R': 'Reverse search',
            'CTRL+A': 'Beginning of line',
            'CTRL+E': 'End of line',
            'CTRL+K': 'Kill to end of line',
            'CTRL+U': 'Kill to beginning of line',
            'CTRL+W': 'Kill word backward',
            'CTRL+Y': 'Yank (paste killed text)',
            'CTRL+T': 'Transpose characters',
            'CTRL+F': 'Forward character',
            'CTRL+B': 'Backward character',
            'CTRL+N': 'Next line',
            'CTRL+P': 'Previous line',
            'ESC': 'Various escape sequences',
            'TAB': 'Completion',
            'SHIFT+TAB': 'Reverse completion',
        }
        
        self.terminal_app_shortcuts = {
            # Vim shortcuts
            'CTRL+[': 'Escape in Vim',
            'CTRL+]': 'Jump to tag in Vim',
            'CTRL+O': 'Jump back in Vim',
            'CTRL+I': 'Jump forward in Vim',
            'CTRL+W': 'Window commands in Vim',
            'CTRL+G': 'File info in Vim',
            
            # Emacs shortcuts
            'CTRL+X': 'Command prefix in Emacs',
            'ALT+X': 'Execute command in Emacs',
            'CTRL+SPACE': 'Set mark in Emacs',
            
            # Bash/Zsh shortcuts
            'CTRL+SHIFT+C': 'Copy (terminal)',
            'CTRL+SHIFT+V': 'Paste (terminal)',
            'ALT+B': 'Backward word',
            'ALT+F': 'Forward word',
            'ALT+D': 'Delete word forward',
            'ALT+BACKSPACE': 'Delete word backward',
            'ALT+.': 'Insert last argument',
            
            # tmux shortcuts (default prefix CTRL+B)
            'CTRL+B': 'tmux prefix key',
            
            # screen shortcuts (default prefix CTRL+A)
            'CTRL+A': 'screen prefix key (conflicts with line beginning)',
        }
        
    def parse_wezterm_config(self, config_path: Path) -> Dict:
        """Parse WezTerm Lua config to extract key bindings"""
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract key bindings from the Lua config
        bindings = {}
        in_keys_section = False
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            if 'config.keys = {' in line:
                in_keys_section = True
                continue
            elif in_keys_section and line == '}':
                in_keys_section = False
                continue
            elif in_keys_section and '{ key =' in line:
                # Parse key binding line
                try:
                    # Extract key, mods, and action
                    if "key = '" in line and "mods = '" in line:
                        key_part = line.split("key = '")[1].split("'")[0]
                        mods_part = line.split("mods = '")[1].split("'")[0]
                        action_part = line.split("action = ")[1].split("}")[0].strip().rstrip(',')
                        
                        key_combo = f"{mods_part}+{key_part}" if mods_part else key_part
                        bindings[key_combo] = {
                            'action': action_part,
                            'line': line_num,
                            'raw': line
                        }
                except Exception as e:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    
        return bindings
    
    def analyze_conflicts(self):
        """Analyze all potential conflicts"""
        
        # Check macOS system conflicts
        for wez_key, wez_info in self.wezterm_bindings.items():
            # Normalize key format
            normalized_key = wez_key.upper().replace('CMD', 'CMD').replace('CTRL', 'CTRL')
            
            if normalized_key in self.macos_system_shortcuts:
                self.conflicts['macos_system'].append({
                    'wezterm_key': wez_key,
                    'wezterm_action': wez_info['action'],
                    'system_function': self.macos_system_shortcuts[normalized_key],
                    'severity': 'HIGH' if 'CMD' in normalized_key else 'MEDIUM',
                    'line': wez_info['line']
                })
        
        # Check Claude Code CLI conflicts
        for wez_key, wez_info in self.wezterm_bindings.items():
            normalized_key = wez_key.upper()
            
            if normalized_key in self.claude_code_cli_shortcuts:
                self.conflicts['claude_code_cli'].append({
                    'wezterm_key': wez_key,
                    'wezterm_action': wez_info['action'],
                    'cli_function': self.claude_code_cli_shortcuts[normalized_key],
                    'severity': 'HIGH' if 'CTRL' in normalized_key else 'MEDIUM',
                    'line': wez_info['line']
                })
        
        # Check terminal application conflicts
        for wez_key, wez_info in self.wezterm_bindings.items():
            normalized_key = wez_key.upper()
            
            if normalized_key in self.terminal_app_shortcuts:
                self.conflicts['terminal_apps'].append({
                    'wezterm_key': wez_key,
                    'wezterm_action': wez_info['action'],
                    'app_function': self.terminal_app_shortcuts[normalized_key],
                    'severity': 'MEDIUM',
                    'line': wez_info['line']
                })
        
        # Check for internal WezTerm conflicts (duplicate bindings)
        key_counts = {}
        for key in self.wezterm_bindings.keys():
            key_counts[key] = key_counts.get(key, 0) + 1
        
        for key, count in key_counts.items():
            if count > 1:
                self.conflicts['internal_wezterm'].append({
                    'key': key,
                    'count': count,
                    'severity': 'HIGH'
                })
    
    def test_specific_keys(self):
        """Test specific keys mentioned in the request"""
        test_keys = ['CMD+T', 'CMD+W', 'CMD+1', 'CMD+2', 'CMD+3', 'CMD+4']
        results = {}
        
        for key in test_keys:
            results[key] = {
                'wezterm_bound': key in self.wezterm_bindings,
                'wezterm_action': self.wezterm_bindings.get(key, {}).get('action', 'Not bound'),
                'macos_conflict': key in self.macos_system_shortcuts,
                'macos_function': self.macos_system_shortcuts.get(key, 'No conflict'),
                'risk_level': self.assess_risk_level(key)
            }
        
        return results
    
    def assess_risk_level(self, key: str) -> str:
        """Assess the risk level of a key binding conflict"""
        if key in ['CMD+W', 'CMD+Q', 'CMD+M']:
            return 'CRITICAL'  # These can close applications unexpectedly
        elif key in ['CMD+T', 'CMD+N', 'CMD+R']:
            return 'HIGH'  # These are very commonly used
        elif 'CMD+' in key and any(c.isdigit() for c in key):
            return 'MEDIUM'  # Tab switching is common but less critical
        else:
            return 'LOW'
    
    def generate_report(self) -> str:
        """Generate comprehensive conflict analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("WEZTERM KEY BINDING CONFLICT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_conflicts = sum(len(conflicts) for conflicts in self.conflicts.values())
        report.append(f"SUMMARY:")
        report.append(f"  Total WezTerm bindings analyzed: {len(self.wezterm_bindings)}")
        report.append(f"  Total conflicts found: {total_conflicts}")
        report.append("")
        
        # Specific key test results
        report.append("SPECIFIC KEY TESTS (CMD+T, CMD+W, CMD+1-4):")
        report.append("-" * 50)
        test_results = self.test_specific_keys()
        
        for key, info in test_results.items():
            report.append(f"  {key}:")
            report.append(f"    WezTerm Binding: {'YES' if info['wezterm_bound'] else 'NO'}")
            if info['wezterm_bound']:
                report.append(f"    WezTerm Action: {info['wezterm_action']}")
            report.append(f"    macOS Conflict: {'YES' if info['macos_conflict'] else 'NO'}")
            if info['macos_conflict']:
                report.append(f"    macOS Function: {info['macos_function']}")
            report.append(f"    Risk Level: {info['risk_level']}")
            report.append("")
        
        # Detailed conflict analysis
        for conflict_type, conflicts in self.conflicts.items():
            if conflicts:
                report.append(f"{conflict_type.upper().replace('_', ' ')} CONFLICTS:")
                report.append("-" * 50)
                
                for conflict in conflicts:
                    if conflict_type == 'macos_system':
                        report.append(f"  {conflict['wezterm_key']} (Line {conflict['line']}):")
                        report.append(f"    WezTerm Action: {conflict['wezterm_action']}")
                        report.append(f"    macOS Function: {conflict['system_function']}")
                        report.append(f"    Severity: {conflict['severity']}")
                        
                    elif conflict_type == 'claude_code_cli':
                        report.append(f"  {conflict['wezterm_key']} (Line {conflict['line']}):")
                        report.append(f"    WezTerm Action: {conflict['wezterm_action']}")
                        report.append(f"    CLI Function: {conflict['cli_function']}")
                        report.append(f"    Severity: {conflict['severity']}")
                        
                    elif conflict_type == 'terminal_apps':
                        report.append(f"  {conflict['wezterm_key']} (Line {conflict['line']}):")
                        report.append(f"    WezTerm Action: {conflict['wezterm_action']}")
                        report.append(f"    App Function: {conflict['app_function']}")
                        report.append(f"    Severity: {conflict['severity']}")
                        
                    elif conflict_type == 'internal_wezterm':
                        report.append(f"  {conflict['key']}: Bound {conflict['count']} times")
                        report.append(f"    Severity: {conflict['severity']}")
                    
                    report.append("")
        
        # Current WezTerm bindings
        report.append("CURRENT WEZTERM KEY BINDINGS:")
        report.append("-" * 50)
        for key, info in sorted(self.wezterm_bindings.items()):
            report.append(f"  {key}: {info['action']} (Line {info['line']})")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)
        
        # Critical recommendations
        critical_conflicts = [c for conflicts in self.conflicts.values() 
                            for c in conflicts if c.get('severity') == 'HIGH']
        
        if any('CMD+W' in str(c) for c in critical_conflicts):
            report.append("  🚨 CRITICAL: CMD+W conflicts with system 'Close Window'")
            report.append("     Consider using CMD+SHIFT+W or disabling this binding")
            report.append("")
        
        if any('CMD+T' in str(c) for c in critical_conflicts):
            report.append("  ⚠️  HIGH: CMD+T conflicts with system 'New Tab'")
            report.append("     Most users expect this to open new tabs in apps")
            report.append("")
        
        # General recommendations
        report.append("  General Recommendations:")
        report.append("  • Use CTRL+SHIFT combinations for terminal-specific actions")
        report.append("  • Avoid overriding common CMD+ shortcuts on macOS")
        report.append("  • Test key bindings in different contexts (vim, emacs, etc.)")
        report.append("  • Consider using F-keys or less common modifier combinations")
        report.append("  • Provide escape sequences for disabled system shortcuts")
        report.append("")
        
        # Safe alternatives
        report.append("SUGGESTED SAFE ALTERNATIVES:")
        report.append("-" * 50)
        report.append("  Instead of CMD+W: CMD+SHIFT+W or CTRL+SHIFT+W")
        report.append("  Instead of CMD+T: CMD+SHIFT+T or CTRL+SHIFT+T")
        report.append("  For pane navigation: ALT+hjkl or CMD+ALT+arrows")
        report.append("  For quick actions: F1-F12 keys")
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self, config_path: Path):
        """Run complete analysis"""
        print("Parsing WezTerm configuration...")
        self.wezterm_bindings = self.parse_wezterm_config(config_path)
        
        print("Analyzing conflicts...")
        self.analyze_conflicts()
        
        print("Generating report...")
        return self.generate_report()

def main():
    analyzer = KeyBindingAnalyzer()
    config_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/.wezterm.lua")
    
    if not config_path.exists():
        print(f"Error: WezTerm config not found at {config_path}")
        return 1
    
    try:
        report = analyzer.run_analysis(config_path)
        print(report)
        
        # Save report to file
        report_path = config_path.parent / "wezterm_conflict_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())