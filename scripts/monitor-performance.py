#!/usr/bin/env python3
"""
Real-time performance monitor for Claude Code development
"""
import os
import time
import psutil
import subprocess
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()

def get_git_stats():
    """Get git repository statistics"""
    try:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        changes = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        changed_files = len(changes.strip().split('\n')) if changes.strip() else 0
        return branch, changed_files
    except:
        return "N/A", 0

def get_python_processes():
    """Get Python process information"""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        if 'python' in proc.info['name'].lower():
            python_procs.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'cpu': proc.info['cpu_percent'],
                'memory': proc.info['memory_info'].rss / 1024 / 1024  # MB
            })
    return python_procs

def create_dashboard():
    """Create performance dashboard"""
    # System stats
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Git stats
    branch, changed_files = get_git_stats()
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=4)
    )
    
    # Header
    header = f"[bold cyan]Claude Code Performance Monitor[/bold cyan] - {datetime.now().strftime('%H:%M:%S')}"
    layout["header"].update(Panel(header))
    
    # Body - split into columns
    layout["body"].split_row(
        Layout(name="system"),
        Layout(name="processes")
    )
    
    # System stats table
    system_table = Table(title="System Resources", expand=True)
    system_table.add_column("Resource", style="cyan")
    system_table.add_column("Usage", style="green")
    system_table.add_column("Details", style="yellow")
    
    # CPU (M4 Pro specific)
    cpu_str = f"{sum(cpu_percent)/len(cpu_percent):.1f}%"
    cpu_detail = f"Cores: {' '.join(f'{c:.0f}%' for c in cpu_percent[:4])}..."
    system_table.add_row("CPU (12 cores)", cpu_str, cpu_detail)
    
    # Memory
    mem_str = f"{memory.percent:.1f}%"
    mem_detail = f"{memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB"
    system_table.add_row("Memory", mem_str, mem_detail)
    
    # Disk
    disk_str = f"{disk.percent:.1f}%"
    disk_detail = f"{disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB"
    system_table.add_row("Disk", disk_str, disk_detail)
    
    # Git
    system_table.add_row("Git Branch", branch, f"{changed_files} changed files")
    
    layout["system"].update(Panel(system_table))
    
    # Process table
    proc_table = Table(title="Python Processes", expand=True)
    proc_table.add_column("PID", style="cyan")
    proc_table.add_column("Name", style="green")
    proc_table.add_column("CPU %", style="yellow")
    proc_table.add_column("Memory (MB)", style="magenta")
    
    for proc in get_python_processes()[:5]:  # Top 5
        proc_table.add_row(
            str(proc['pid']),
            proc['name'],
            f"{proc['cpu']:.1f}%",
            f"{proc['memory']:.1f}"
        )
    
    layout["processes"].update(Panel(proc_table))
    
    # Footer - Quick commands
    footer_text = """[bold]Quick Commands:[/bold]
[cyan]pytest -n 12[/cyan] - Run tests with 12 cores | [cyan]git claude-commit "msg"[/cyan] - Quick commit
[cyan]./orchestrate "task"[/cyan] - Hardware accelerated analysis | [cyan]Ctrl+C[/cyan] - Exit"""
    layout["footer"].update(Panel(footer_text))
    
    return layout

def main():
    """Run performance monitor"""
    console.print("[bold green]Starting Claude Code Performance Monitor...[/bold green]")
    console.print("Press Ctrl+C to exit\n")
    
    try:
        with Live(create_dashboard(), refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(create_dashboard())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")

if __name__ == "__main__":
    main()