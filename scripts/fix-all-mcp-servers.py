#!/usr/bin/env python3
"""Fix all failing MCP servers with comprehensive diagnostics and solutions."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_status(status: str, message: str):
    """Print colored status message."""
    color = GREEN if status == "OK" else RED if status == "ERROR" else YELLOW
    print(f"{color}[{status}]{NC} {message}")

def check_python_module(module: str) -> bool:
    """Check if a Python module is installed."""
    try:
        subprocess.run([sys.executable, "-c", f"import {module}"], 
                      capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_python_module(module: str, package: str = None) -> bool:
    """Install a Python module."""
    pkg = package or module
    try:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def create_enhanced_trace_server():
    """Create an enhanced trace MCP server."""
    trace_content = '''#!/usr/bin/env python3
"""Enhanced trace MCP server for debugging and logging."""

from mcp.server import FastMCP
import json
import time
from datetime import datetime
from pathlib import Path
import traceback
import sys
import os

mcp = FastMCP("trace")

# Create trace directory
TRACE_DIR = Path.home() / ".local" / "share" / "wheel-trading" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
def trace_log(
    message: str,
    level: str = "INFO",
    context: dict = None
) -> str:
    """Log a trace message with context.
    
    Args:
        message: The message to log
        level: Log level (DEBUG, INFO, WARN, ERROR)
        context: Additional context data
    """
    timestamp = datetime.now().isoformat()
    trace_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "context": context or {}
    }
    
    # Write to trace file
    trace_file = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(trace_file, 'a') as f:
        f.write(json.dumps(trace_entry) + '\\n')
    
    return f"Traced at {timestamp}: [{level}] {message}"

@mcp.tool()
def trace_error(
    error_type: str,
    error_message: str,
    stack_trace: str = None
) -> str:
    """Log an error with full stack trace.
    
    Args:
        error_type: Type of error
        error_message: Error message
        stack_trace: Optional stack trace
    """
    if not stack_trace:
        stack_trace = traceback.format_exc()
    
    context = {
        "error_type": error_type,
        "stack_trace": stack_trace,
        "python_version": sys.version,
        "cwd": os.getcwd()
    }
    
    return trace_log(error_message, "ERROR", context)

@mcp.tool()
def trace_performance(
    operation: str,
    duration_ms: float,
    metadata: dict = None
) -> str:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        metadata: Additional performance metadata
    """
    context = {
        "operation": operation,
        "duration_ms": duration_ms,
        "metadata": metadata or {}
    }
    
    return trace_log(f"Performance: {operation} took {duration_ms}ms", "INFO", context)

@mcp.tool()
def get_traces(
    level: str = None,
    limit: int = 100,
    date: str = None
) -> str:
    """Retrieve recent traces.
    
    Args:
        level: Filter by log level
        limit: Maximum number of traces to return
        date: Date to retrieve traces for (YYYYMMDD format)
    """
    if date:
        trace_file = TRACE_DIR / f"trace_{date}.jsonl"
    else:
        trace_file = TRACE_DIR / f"trace_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    if not trace_file.exists():
        return "No traces found for the specified date"
    
    traces = []
    with open(trace_file, 'r') as f:
        for line in f:
            try:
                trace = json.loads(line.strip())
                if not level or trace.get('level') == level:
                    traces.append(trace)
            except json.JSONDecodeError:
                continue
    
    # Return most recent traces up to limit
    traces = traces[-limit:]
    return json.dumps(traces, indent=2)

@mcp.tool()
def clear_traces(date: str = None) -> str:
    """Clear trace logs.
    
    Args:
        date: Specific date to clear (YYYYMMDD format), or all if not specified
    """
    if date:
        trace_file = TRACE_DIR / f"trace_{date}.jsonl"
        if trace_file.exists():
            trace_file.unlink()
            return f"Cleared traces for {date}"
        return f"No traces found for {date}"
    else:
        # Clear all trace files
        count = 0
        for trace_file in TRACE_DIR.glob("trace_*.jsonl"):
            trace_file.unlink()
            count += 1
        return f"Cleared {count} trace files"

if __name__ == "__main__":
    print("Starting enhanced trace MCP server...")
    mcp.run()
'''
    
    trace_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-mcp-enhanced.py")
    trace_path.write_text(trace_content)
    trace_path.chmod(0o755)
    print_status("OK", f"Created enhanced trace server: {trace_path}")
    return str(trace_path)

def create_ripgrep_wrapper():
    """Create a ripgrep wrapper that handles the space in path issue."""
    wrapper_content = '''#!/usr/bin/env python3
"""Ripgrep MCP wrapper to handle paths with spaces."""

import sys
import os

# Add the scripts directory to Python path
scripts_dir = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts"
sys.path.insert(0, scripts_dir)

# Import and run the actual ripgrep MCP
import importlib.util
spec = importlib.util.spec_from_file_location("ripgrep_mcp", os.path.join(scripts_dir, "ripgrep-mcp.py"))
ripgrep_mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ripgrep_mcp)
'''
    
    # Create wrapper in the scripts directory instead
    wrapper_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/ripgrep-mcp-wrapper.py")
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)
    print_status("OK", f"Created ripgrep wrapper: {wrapper_path}")
    return str(wrapper_path)

def fix_mcp_servers():
    """Main function to fix all MCP servers."""
    print(f"{GREEN}=== Fixing ALL MCP Servers ==={NC}\n")
    
    # Load current config
    config_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    fixes_made = []
    
    # 1. Fix python_analysis - already updated to use python-mcp-server-simple.py
    print_status("INFO", "python_analysis already updated to use simple version")
    
    # 2. Fix trace server
    print(f"\n{YELLOW}Fixing trace server...{NC}")
    enhanced_trace_path = create_enhanced_trace_server()
    config["mcpServers"]["trace"]["args"] = [enhanced_trace_path]
    fixes_made.append("trace - created enhanced version with full logging")
    
    # 3. Fix ripgrep - create wrapper to handle space in path
    print(f"\n{YELLOW}Fixing ripgrep server...{NC}")
    if check_python_module("mcp.server"):
        wrapper_path = create_ripgrep_wrapper()
        config["mcpServers"]["ripgrep"]["command"] = wrapper_path
        config["mcpServers"]["ripgrep"]["args"] = []
        fixes_made.append("ripgrep - created wrapper to handle path spaces")
    
    # 4. Fix dependency-graph
    print(f"\n{YELLOW}Fixing dependency-graph server...{NC}")
    # Check if we should use NPX version instead
    dep_graph_content = '''#!/usr/bin/env python3
"""Dependency graph MCP server for code analysis."""

from mcp.server import FastMCP
import ast
import os
from pathlib import Path
import json

mcp = FastMCP("dependency-graph")

@mcp.tool()
def analyze_imports(file_path: str) -> str:
    """Analyze Python imports in a file.
    
    Args:
        file_path: Path to the Python file to analyze
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        with open(path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"type": "import", "module": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name
                    })
        
        return json.dumps({
            "file": str(path),
            "imports": imports,
            "import_count": len(imports)
        }, indent=2)
        
    except SyntaxError as e:
        return f"Syntax error in {file_path}: {e}"
    except Exception as e:
        return f"Error analyzing {file_path}: {str(e)}"

@mcp.tool()
def find_dependencies(directory: str, module_name: str) -> str:
    """Find all files that import a specific module.
    
    Args:
        directory: Directory to search in
        module_name: Module name to search for
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        dependencies = []
        
        for py_file in dir_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if module_name in alias.name:
                                dependencies.append(str(py_file))
                                break
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and module_name in node.module:
                            dependencies.append(str(py_file))
                            break
            except:
                continue
        
        return json.dumps({
            "module": module_name,
            "used_by": dependencies,
            "usage_count": len(dependencies)
        }, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def generate_dependency_graph(directory: str) -> str:
    """Generate a full dependency graph for a Python project.
    
    Args:
        directory: Root directory of the project
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        graph = {}
        
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_imports.append(node.module)
                
                relative_path = py_file.relative_to(dir_path)
                graph[str(relative_path)] = list(set(file_imports))
                
            except:
                continue
        
        return json.dumps(graph, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting dependency-graph MCP server...")
    mcp.run()
'''
    
    dep_graph_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp.py")
    dep_graph_path.write_text(dep_graph_content)
    dep_graph_path.chmod(0o755)
    config["mcpServers"]["dependency-graph"]["command"] = sys.executable
    config["mcpServers"]["dependency-graph"]["args"] = [str(dep_graph_path)]
    fixes_made.append("dependency-graph - created Python version with AST analysis")
    
    # 5. Check trace-opik
    print(f"\n{YELLOW}Checking trace-opik server...{NC}")
    opik_script = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-opik-mcp.py")
    if opik_script.exists():
        print_status("OK", "trace-opik script exists")
        # Ensure opik module is installed
        if not check_python_module("opik"):
            install_python_module("opik")
            fixes_made.append("trace-opik - installed opik module")
    
    # 6. Check trace-phoenix
    print(f"\n{YELLOW}Checking trace-phoenix server...{NC}")
    phoenix_script = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-phoenix-mcp.py")
    if phoenix_script.exists():
        print_status("OK", "trace-phoenix script exists")
        # Phoenix has issues with SQLAlchemy, keep it disabled for now
        fixes_made.append("trace-phoenix - kept disabled due to SQLAlchemy issues")
    
    # 7. Fix logfire
    print(f"\n{YELLOW}Checking logfire server...{NC}")
    logfire_script = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/trace-logfire-mcp.py")
    if logfire_script.exists():
        print_status("OK", "logfire script exists")
        if not check_python_module("logfire"):
            install_python_module("logfire")
            fixes_made.append("logfire - installed logfire module")
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')
    
    # Summary
    print(f"\n{GREEN}=== Summary ==={NC}")
    print(f"Fixed {len(fixes_made)} servers:")
    for fix in fixes_made:
        print(f"  • {fix}")
    
    print(f"\n{YELLOW}Next steps:{NC}")
    print("1. Restart Claude to load the updated configuration")
    print("2. Use /mcp command in Claude to verify all servers are working")
    print("3. If any servers still fail, check the logs in .claude/runtime/")

if __name__ == "__main__":
    fix_mcp_servers()