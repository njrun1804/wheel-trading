#!/usr/bin/env python3
"""
Memory monitoring for Claude Code to prevent string overflow errors.
"""

import psutil
import time
import json
from pathlib import Path

class MemoryMonitor:
    def __init__(self, warning_threshold=0.8, critical_threshold=0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
    def check_memory_status(self):
        """Check current memory status and return recommendations."""
        memory = psutil.virtual_memory()
        usage_ratio = memory.percent / 100
        
        status = {
            'total_gb': round(self.total_memory, 1),
            'used_gb': round(memory.used / (1024**3), 1),
            'available_gb': round(memory.available / (1024**3), 1),
            'usage_percent': memory.percent,
            'status': 'OK'
        }
        
        if usage_ratio > self.critical_threshold:
            status['status'] = 'CRITICAL'
            status['recommendation'] = 'Stop Claude sessions and restart'
        elif usage_ratio > self.warning_threshold:
            status['status'] = 'WARNING'
            status['recommendation'] = 'Use streaming mode or reduce output size'
        else:
            status['recommendation'] = 'Memory usage normal'
            
        return status
    
    def get_claude_processes(self):
        """Get information about running Claude processes."""
        claude_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'claude' in proc.info['name'].lower():
                    claude_procs.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': round(proc.info['memory_info'].rss / (1024**2), 1),
                        'cpu_percent': proc.info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return claude_procs
    
    def suggest_node_options(self):
        """Suggest optimal Node.js options based on available memory."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        # Use 75% of available memory for Node.js heap
        suggested_heap = int(available_gb * 0.75 * 1024)  # MB
        suggested_semi = max(512, suggested_heap // 40)  # Semi-space
        
        return {
            'NODE_OPTIONS': f'--max-old-space-size={suggested_heap} --max-semi-space-size={suggested_semi}',
            'heap_size_gb': round(suggested_heap / 1024, 1),
            'explanation': f'Optimized for {round(available_gb, 1)}GB available memory'
        }

def main():
    monitor = MemoryMonitor()
    
    print("=== Claude Code Memory Monitor ===")
    
    # Memory status
    status = monitor.check_memory_status()
    print(f"\nMemory Status: {status['status']}")
    print(f"Total: {status['total_gb']}GB | Used: {status['used_gb']}GB | Available: {status['available_gb']}GB")
    print(f"Usage: {status['usage_percent']:.1f}%")
    print(f"Recommendation: {status['recommendation']}")
    
    # Claude processes
    claude_procs = monitor.get_claude_processes()
    if claude_procs:
        print(f"\nClaude Processes ({len(claude_procs)}):")
        for proc in claude_procs:
            print(f"  PID {proc['pid']}: {proc['memory_mb']:.1f}MB")
    else:
        print("\nNo Claude processes found")
    
    # Node.js optimization
    node_opts = monitor.suggest_node_options()
    print(f"\nRecommended Node.js settings:")
    print(f"export {node_opts['NODE_OPTIONS']}")
    print(f"Heap size: {node_opts['heap_size_gb']}GB ({node_opts['explanation']})")
    
    return status['status'] == 'OK'

if __name__ == '__main__':
    exit(0 if main() else 1)