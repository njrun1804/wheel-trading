#!/usr/bin/env python3
"""
Claude Code Production Hooks
Integrates meta improvement system directly with Claude Code CLI
"""

import os
import sys
import time
import json
import threading
from typing import Any, Dict, Optional

# Production system integration
from production_meta_improvement_system import get_production_system


class ClaudeCodeProductionHooks:
    """Production hooks for Claude Code integration"""
    
    def __init__(self):
        self.system = get_production_system()
        self.hook_lock = threading.RLock()
        self.hooks_installed = False
        
        print("🎣 Claude Code Production Hooks initialized")
    
    def install_hooks(self):
        """Install production hooks into Claude Code environment"""
        
        if self.hooks_installed:
            return
        
        print("🔧 Installing Claude Code production hooks...")
        
        # Hook into environment variables to detect Claude Code operations
        self._hook_environment_monitoring()
        
        # Hook into file operations
        self._hook_file_operations()
        
        # Hook into response processing
        self._hook_response_processing()
        
        self.hooks_installed = True
        print("✅ Production hooks installed")
    
    def _hook_environment_monitoring(self):
        """Monitor Claude Code environment for activity"""
        
        def monitor_environment():
            """Background thread to monitor Claude Code activity"""
            
            last_check = time.time()
            
            while True:
                try:
                    # Check for Claude Code activity indicators
                    current_time = time.time()
                    
                    # Monitor for new environment variables
                    claude_vars = {k: v for k, v in os.environ.items() if 'CLAUDE' in k}
                    
                    if claude_vars and current_time - last_check > 60:
                        # Record Claude Code activity
                        self.system.capture_user_request(
                            "Claude Code environment activity detected",
                            {"environment_vars": list(claude_vars.keys())}
                        )
                        last_check = current_time
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    print(f"⚠️ Environment monitoring error: {e}")
                    time.sleep(60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_environment, daemon=True)
        monitor_thread.start()
    
    def _hook_file_operations(self):
        """Hook into file write operations to improve code"""
        
        # Override built-in open function for write operations
        original_open = __builtins__['open']
        
        def improved_open(file, mode='r', *args, **kwargs):
            """Enhanced open that improves code files before writing"""
            
            # Check if this is a write operation on a code file
            if 'w' in mode and self._is_code_file(str(file)):
                # Return a wrapper that improves content before writing
                return CodeFileWrapper(original_open(file, mode, *args, **kwargs), self.system)
            else:
                return original_open(file, mode, *args, **kwargs)
        
        # Replace the built-in open function
        __builtins__['open'] = improved_open
    
    def _hook_response_processing(self):
        """Hook into Claude response processing"""
        
        # This would integrate with Claude Code's response handling
        # For now, we'll create a monitoring mechanism
        
        def process_claude_response(response_text: str, context: Dict[str, Any] = None):
            """Process Claude response for learning"""
            
            with self.hook_lock:
                # Capture Claude's response for learning
                self.system.capture_claude_response(response_text, context or {})
                
                # If response contains code, improve it
                if self._contains_code(response_text):
                    # Extract and improve code blocks
                    improved_response = self._improve_code_in_response(response_text)
                    return improved_response
                
                return response_text
        
        # Store the function for external use
        self.process_claude_response = process_claude_response
    
    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is a code file"""
        code_extensions = {'.py', '.js', '.ts', '.sh', '.go', '.java', '.cpp', '.c', '.rs', '.rb'}
        return any(filepath.endswith(ext) for ext in code_extensions)
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks"""
        code_indicators = ['```', 'def ', 'function ', 'class ', 'import ', '#include', 'const ']
        return any(indicator in text for indicator in code_indicators)
    
    def _improve_code_in_response(self, response_text: str) -> str:
        """Improve code blocks within a response"""
        
        lines = response_text.split('\n')
        improved_lines = []
        in_code_block = False
        code_buffer = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block - improve the code
                    if code_buffer:
                        code = '\n'.join(code_buffer)
                        improved_code, _ = self.system.intercept_and_improve_code(
                            code, {"source": "claude_response"}
                        )
                        improved_lines.extend(improved_code.split('\n'))
                    improved_lines.append(line)
                    code_buffer = []
                    in_code_block = False
                else:
                    # Start of code block
                    improved_lines.append(line)
                    in_code_block = True
            elif in_code_block:
                code_buffer.append(line)
            else:
                improved_lines.append(line)
        
        return '\n'.join(improved_lines)


class CodeFileWrapper:
    """Wrapper for file objects that improves code before writing"""
    
    def __init__(self, file_obj, meta_system):
        self.file_obj = file_obj
        self.meta_system = meta_system
        self.written_content = []
    
    def write(self, content):
        """Improve content before writing"""
        
        # Accumulate content for improvement
        self.written_content.append(content)
        
        # If we have enough content, improve it
        if len(''.join(self.written_content)) > 100:  # Minimum content threshold
            full_content = ''.join(self.written_content)
            improved_content, _ = self.meta_system.intercept_and_improve_code(
                full_content, {"operation": "file_write"}
            )
            
            # Write improved content
            result = self.file_obj.write(improved_content)
            self.written_content = []  # Reset buffer
            return result
        else:
            # Not enough content yet, just write as-is
            return self.file_obj.write(content)
    
    def __getattr__(self, name):
        """Delegate all other methods to the original file object"""
        return getattr(self.file_obj, name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Improve any remaining content before closing
        if self.written_content:
            full_content = ''.join(self.written_content)
            improved_content, _ = self.meta_system.intercept_and_improve_code(
                full_content, {"operation": "file_write_final"}
            )
            self.file_obj.write(improved_content)
        
        return self.file_obj.__exit__(exc_type, exc_val, exc_tb)


# Global hooks instance
_production_hooks = None


def get_production_hooks() -> ClaudeCodeProductionHooks:
    """Get or create global production hooks"""
    global _production_hooks
    
    if _production_hooks is None:
        _production_hooks = ClaudeCodeProductionHooks()
    
    return _production_hooks


def activate_claude_code_hooks():
    """Activate production hooks for Claude Code"""
    
    hooks = get_production_hooks()
    hooks.install_hooks()
    
    print("🎯 Claude Code production hooks activated")
    print("🔧 All file operations now improved automatically")
    
    return hooks


# Auto-activation for production use
# META COMPLETELY DISABLED FOR EINSTEIN TESTING
# if os.getenv('CLAUDECODE') or os.getenv('ACTIVATE_META_IMPROVEMENTS'):
#     print("🚀 Auto-activating Claude Code production hooks...")
#     activate_claude_code_hooks()
print("🔪 Meta auto-activation DISABLED for clean Einstein testing")


# Test function for verification
def test_production_hooks():
    """Test the production hooks"""
    
    print("🧪 Testing production hooks...")
    
    hooks = get_production_hooks()
    hooks.install_hooks()
    
    # Test code improvement
    test_code = "def unsafe_read(f): return open(f).read()"
    improved = hooks.process_claude_response(f"```python\n{test_code}\n```")
    
    print("✅ Production hooks test completed")
    return hooks


if __name__ == "__main__":
    test_production_hooks()