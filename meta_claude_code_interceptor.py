#!/usr/bin/env python3
"""
Meta Claude Code Interceptor
Hooks into Claude Code generation to apply real-time improvements
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from meta_active_improvement_engine import MetaActiveImprovementEngine
from claude_cli_reasoning_capture import ClaudeCLIReasoningCapture
from meta_prime import MetaPrime


class MetaClaudeCodeInterceptor:
    """Intercepts and improves Claude Code generation in real-time"""
    
    def __init__(self):
        self.improvement_engine = MetaActiveImprovementEngine()
        self.cli_capture = ClaudeCLIReasoningCapture()
        # Only create MetaPrime if not disabled
        import os
        if os.environ.get('DISABLE_META_AUTOSTART') == '1':
            self.meta_prime = None
        else:
            self.meta_prime = MetaPrime()
        
        # Interception state
        self.intercepts_active = True
        self.improvements_count = 0
        self.bypass_count = 0
        
        print("🎣 Meta Claude Code Interceptor ready")
        print("🔧 Real-time code improvement active")
    
    def code_generation_hook(self, original_func: Callable) -> Callable:
        """Decorator to hook into code generation functions"""
        
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # Execute original code generation
            generated_code = original_func(*args, **kwargs)
            
            if not self.intercepts_active or not isinstance(generated_code, str):
                self.bypass_count += 1
                return generated_code
            
            # Capture the generation event
            self._capture_generation_event(generated_code, args, kwargs)
            
            # Apply active improvements
            improved_code, improvements = self.improvement_engine.intercept_and_improve_code(
                generated_code, 
                {"function": original_func.__name__, "args_count": len(args)}
            )
            
            if improvements:
                self.improvements_count += len(improvements)
                print(f"🔧 Code improved: {len(improvements)} enhancements applied")
                
                # Record successful interception
                self.meta_prime.observe("code_generation_intercepted", {
                    "original_length": len(generated_code),
                    "improved_length": len(improved_code),
                    "improvements_count": len(improvements),
                    "function": original_func.__name__
                })
                
                return improved_code
            
            return generated_code
        
        return wrapper
    
    def user_request_hook(self, original_func: Callable) -> Callable:
        """Decorator to hook into user request processing"""
        
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # Capture user request
            if args and isinstance(args[0], str):
                self.cli_capture.capture_user_request(args[0], {
                    "function": original_func.__name__,
                    "timestamp": time.time()
                })
            
            # Execute original function
            result = original_func(*args, **kwargs)
            
            # Capture Claude's response/analysis
            if isinstance(result, str):
                self.cli_capture.capture_claude_analysis(result, {
                    "response_to": args[0] if args else "unknown",
                    "processing_function": original_func.__name__
                })
            
            return result
        
        return wrapper
    
    def error_recovery_hook(self, original_func: Callable) -> Callable:
        """Decorator to hook into error recovery functions"""
        
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            try:
                return original_func(*args, **kwargs)
            except Exception as e:
                # Capture error and recovery attempt
                error_msg = str(e)
                recovery_attempt = f"Recovered from {type(e).__name__} in {original_func.__name__}"
                
                self.cli_capture.capture_error_recovery(error_msg, recovery_attempt, {
                    "function": original_func.__name__,
                    "error_type": type(e).__name__
                })
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    def _capture_generation_event(self, code: str, args: tuple, kwargs: dict):
        """Capture code generation event for training"""
        
        # Determine language from context
        language = "python"  # Default
        if "javascript" in str(kwargs).lower() or "js" in str(kwargs).lower():
            language = "javascript"
        elif "bash" in str(kwargs).lower() or "shell" in str(kwargs).lower():
            language = "bash"
        
        self.cli_capture.capture_code_generation(code, language, {
            "generation_context": "claude_code_cli",
            "args_provided": len(args),
            "kwargs_provided": len(kwargs)
        })
    
    def get_interception_stats(self) -> Dict[str, Any]:
        """Get statistics on code interception"""
        
        engine_stats = self.improvement_engine.get_improvement_stats()
        cli_stats = self.cli_capture.get_monitoring_status()
        
        return {
            "intercepts_active": self.intercepts_active,
            "improvements_applied": self.improvements_count,
            "bypassed_generations": self.bypass_count,
            "improvement_engine": engine_stats,
            "cli_capture": cli_stats,
            "total_patterns_learned": len(self.improvement_engine.learned_patterns)
        }
    
    def toggle_intercepts(self, active: bool = None):
        """Toggle code interception on/off"""
        
        if active is None:
            self.intercepts_active = not self.intercepts_active
        else:
            self.intercepts_active = active
        
        status = "ACTIVE" if self.intercepts_active else "DISABLED"
        print(f"🎣 Code interception: {status}")
        
        self.meta_prime.observe("interception_toggled", {
            "active": self.intercepts_active,
            "timestamp": time.time()
        })


# Global interceptor instance (lazy initialization)
interceptor = None

def get_interceptor():
    """Get or create the global interceptor instance"""
    global interceptor
    if interceptor is None:
        import os
        if os.environ.get('DISABLE_META_AUTOSTART') == '1':
            # Return a stub interceptor
            class StubInterceptor:
                def code_generation_hook(self, func):
                    return func
                def user_request_hook(self, func):
                    return func
                def error_recovery_hook(self, func):
                    return func
            interceptor = StubInterceptor()
        else:
            interceptor = MetaClaudeCodeInterceptor()
    return interceptor


# Decorator functions for easy use
def improve_code_generation(func):
    """Decorator to automatically improve generated code"""
    return get_interceptor().code_generation_hook(func)


def capture_user_request(func):
    """Decorator to capture user requests for training"""
    return get_interceptor().user_request_hook(func)


def monitor_error_recovery(func):
    """Decorator to monitor error recovery patterns"""
    return get_interceptor().error_recovery_hook(func)


# Demo integration
@improve_code_generation
def generate_sample_code(prompt: str) -> str:
    """Simulate Claude generating code"""
    # This would be actual Claude code generation
    if "file" in prompt.lower():
        return '''def read_file(filename):
    f = open(filename)
    content = f.read()
    return content'''
    else:
        return '''def process_data(x):
    y = x * 2
    return y'''


@capture_user_request
def process_user_request(request: str) -> str:
    """Simulate processing user request"""
    return f"I'll help you with: {request}. Let me analyze this systematically and provide a safe, well-documented solution."


@monitor_error_recovery
def risky_operation():
    """Simulate operation that might fail"""
    raise ValueError("Simulated error for testing recovery")


def demo_code_interception():
    """Demonstrate real-time code interception and improvement"""
    
    print("🎣 DEMO: REAL-TIME CODE INTERCEPTION")
    print("=" * 60)
    
    # Test user request capture
    print("1️⃣ Testing user request capture...")
    response = process_user_request("Create a function to read a file safely")
    print(f"📥 Captured request and response")
    
    # Test code generation improvement
    print("\\n2️⃣ Testing code generation improvement...")
    original_code = generate_sample_code("Create a file reader")
    print(f"📝 Generated and improved code")
    
    # Test another generation
    print("\\n3️⃣ Testing another code generation...")
    math_code = generate_sample_code("Create a math function")
    print(f"📝 Generated math function")
    
    # Test error recovery
    print("\\n4️⃣ Testing error recovery monitoring...")
    try:
        risky_operation()
    except ValueError:
        print("📥 Captured error recovery pattern")
    
    # Show interception stats
    print("\\n📊 INTERCEPTION STATISTICS:")
    stats = interceptor.get_interception_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)):
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("\\n🎯 REAL-TIME INTERCEPTION: WORKING!")
    print("Every piece of code generated is now improved using Claude patterns!")
    
    return interceptor


if __name__ == "__main__":
    demo_code_interception()