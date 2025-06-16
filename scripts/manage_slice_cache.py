#!/usr/bin/env python3
"""
Slice cache management utility.
Monitor performance, evict old entries, and analyze cache usage.
"""

import argparse
import sys
from pathlib import Path

import duckdb
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.unity_wheel.mcp.embedding_pipeline import EmbeddingPipeline
from src.unity_wheel.storage.slice_cache import SliceCache

console = Console()


def show_stats(cache_path: Path):
    """Display cache statistics."""
    cache = SliceCache(db_path=cache_path)
    stats = cache.get_cache_stats()

    # Overall stats table
    console.print("\n[bold blue]Cache Overview[/bold blue]")
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    overall = stats["overall"]
    table.add_row("Total Slices", f"{overall['total_slices']:,}")
    table.add_row("Total Uses", f"{overall['total_uses']:,}")
    table.add_row("Avg Uses/Slice", f"{overall['avg_uses_per_slice']:.1f}")
    table.add_row("Tokens Cached", f"{overall['total_tokens_cached']:,}")
    table.add_row("Cache Size", overall["cache_size"])

    console.print(table)

    # Today's performance
    console.print("\n[bold blue]Today's Performance[/bold blue]")
    today = stats["today"]

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Lookups", f"{today['lookups']:,}")
    table.add_row("Cache Hits", f"{today['hits']:,}")
    table.add_row("Cache Misses", f"{today['misses']:,}")
    table.add_row("Hit Rate", f"{today['hit_rate']:.1f}%")
    table.add_row("Bandwidth Saved", today["bytes_saved"])

    console.print(table)

    # Recent performance trend
    if stats["recent_performance"]:
        console.print("\n[bold blue]7-Day Trend[/bold blue]")
        table = Table(box=box.ROUNDED)
        table.add_column("Date", style="cyan")
        table.add_column("Lookups", style="yellow")
        table.add_column("Hit Rate", style="green")

        for day in stats["recent_performance"]:
            table.add_row(day["date"], f"{day['lookups']:,}", f"{day['hit_rate']:.1f}%")

        console.print(table)

    # Top slices
    if stats["top_slices"]:
        console.print("\n[bold blue]Most Used Slices[/bold blue]")
        table = Table(box=box.ROUNDED)
        table.add_column("File", style="cyan", max_width=40)
        table.add_column("Lines", style="yellow")
        table.add_column("Uses", style="green")
        table.add_column("Preview", style="white", max_width=40)

        for slice_info in stats["top_slices"][:5]:
            table.add_row(
                Path(slice_info["file"]).name,
                slice_info["lines"],
                str(slice_info["uses"]),
                slice_info["preview"][:40] + "...",
            )

        console.print(table)


def evict_old_slices(cache_path: Path, days: int):
    """Evict slices older than specified days."""
    cache = SliceCache(db_path=cache_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evicting slices older than {days} days...", total=None
        )

        evicted = cache.evict_old_slices(days)

        progress.update(task, completed=True)

    console.print(f"\n[green]✓[/green] Evicted {evicted:,} old slices")


def analyze_usage(cache_path: Path):
    """Analyze cache usage patterns."""
    with duckdb.connect(str(cache_path)) as conn:
        # File coverage analysis
        file_stats = conn.execute(
            """
            SELECT 
                file_path,
                COUNT(*) as slice_count,
                SUM(use_count) as total_uses,
                AVG(use_count) as avg_uses,
                MAX(last_used) as last_accessed
            FROM slice_cache
            GROUP BY file_path
            ORDER BY total_uses DESC
            LIMIT 20
        """
        ).fetchdf()

        console.print("\n[bold blue]File Coverage Analysis[/bold blue]")
        table = Table(box=box.ROUNDED)
        table.add_column("File", style="cyan", max_width=50)
        table.add_column("Slices", style="yellow")
        table.add_column("Total Uses", style="green")
        table.add_column("Avg Uses", style="magenta")
        table.add_column("Last Access", style="white")

        for _, row in file_stats.iterrows():
            table.add_row(
                Path(row["file_path"]).name,
                str(row["slice_count"]),
                f"{row['total_uses']:,}",
                f"{row['avg_uses']:.1f}",
                row["last_accessed"].strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

        # Token distribution
        token_stats = conn.execute(
            """
            SELECT 
                CASE 
                    WHEN token_count < 100 THEN '< 100'
                    WHEN token_count < 500 THEN '100-500'
                    WHEN token_count < 1000 THEN '500-1000'
                    WHEN token_count < 2000 THEN '1000-2000'
                    ELSE '> 2000'
                END as token_range,
                COUNT(*) as slice_count,
                SUM(use_count) as total_uses
            FROM slice_cache
            GROUP BY token_range
            ORDER BY MIN(token_count)
        """
        ).fetchdf()

        console.print("\n[bold blue]Token Distribution[/bold blue]")
        table = Table(box=box.ROUNDED)
        table.add_column("Token Range", style="cyan")
        table.add_column("Slices", style="yellow")
        table.add_column("Total Uses", style="green")

        for _, row in token_stats.iterrows():
            table.add_row(
                row["token_range"], f"{row['slice_count']:,}", f"{row['total_uses']:,}"
            )

        console.print(table)


def warmup_directory(cache_path: Path, directory: Path, pattern: str = "*.py"):
    """Warm up cache for a directory."""
    pipeline = EmbeddingPipeline(cache_path=cache_path)

    console.print(f"\n[bold blue]Warming up cache for: {directory}[/bold blue]")
    console.print(f"Pattern: {pattern}")

    async def run_warmup():
        await pipeline.warmup_cache(str(directory), pattern)
        return pipeline.get_pipeline_stats()

    import asyncio

    stats = asyncio.run(run_warmup())

    pipeline.cleanup()

    # Show results
    console.print("\n[green]✓[/green] Cache warmup complete!")
    console.print(f"\nFiles processed: {stats['pipeline']['files_processed']}")
    console.print(f"Slices cached: {stats['pipeline']['slices_processed']}")
    console.print(f"Tokens processed: {stats['pipeline']['tokens_processed']:,}")
    console.print(
        f"Estimated cost: ${stats['pipeline']['tokens_processed'] * 0.0001:.2f}"
    )


def export_stats(cache_path: Path, output_path: Path):
    """Export cache statistics to JSON."""
    cache = SliceCache(db_path=cache_path)
    cache.export_stats(output_path)

    console.print(f"\n[green]✓[/green] Exported stats to: {output_path}")


def clear_cache(cache_path: Path):
    """Clear all cache entries."""
    console.print(
        "\n[bold red]WARNING: This will delete all cached embeddings![/bold red]"
    )
    confirm = console.input("Type 'yes' to confirm: ")

    if confirm.lower() == "yes":
        cache = SliceCache(db_path=cache_path)
        cache.clear_cache()
        console.print("\n[green]✓[/green] Cache cleared")
    else:
        console.print("\n[yellow]Cancelled[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Manage embedding slice cache")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path.home() / ".wheel_trading" / "cache" / "slice_cache.duckdb",
        help="Path to cache database",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")

    # Evict command
    evict_parser = subparsers.add_parser("evict", help="Evict old cache entries")
    evict_parser.add_argument(
        "--days", type=int, default=30, help="Evict entries older than N days"
    )

    # Analyze command
    subparsers.add_parser("analyze", help="Analyze cache usage patterns")

    # Warmup command
    warmup_parser = subparsers.add_parser("warmup", help="Warm up cache for directory")
    warmup_parser.add_argument("directory", type=Path, help="Directory to scan")
    warmup_parser.add_argument(
        "--pattern", default="*.py", help="File pattern to match"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export stats to JSON")
    export_parser.add_argument("output", type=Path, help="Output JSON file")

    # Clear command
    subparsers.add_parser("clear", help="Clear all cache entries")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Ensure cache exists
    if not args.cache_path.exists() and args.command != "warmup":
        console.print(f"[red]Cache not found at: {args.cache_path}[/red]")
        console.print("Run with 'warmup' command to create cache")
        return

    # Execute command
    if args.command == "stats":
        show_stats(args.cache_path)
    elif args.command == "evict":
        evict_old_slices(args.cache_path, args.days)
    elif args.command == "analyze":
        analyze_usage(args.cache_path)
    elif args.command == "warmup":
        warmup_directory(args.cache_path, args.directory, args.pattern)
    elif args.command == "export":
        export_stats(args.cache_path, args.output)
    elif args.command == "clear":
        clear_cache(args.cache_path)


if __name__ == "__main__":
    main()
