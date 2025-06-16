#!/usr/bin/env python3
"""Benchmark harness for MLX vs PyTorch on M4 Pro.

Usage:
    python bench.py --backend mlx --batch 4096
    python bench.py --backend torch --batch 4096
    python bench.py --compare --batch 4096
"""
import argparse
import json
import platform
import time
from pathlib import Path

import psutil

# Ensure we're on M4 Pro
IS_M4PRO = platform.machine() == "arm64" and "Apple M4" in platform.platform()


class BenchmarkSuite:
    """Comprehensive benchmarks for Jarvis2 operations."""

    def __init__(self, backend: str, batch_size: int = 4096):
        self.backend = backend
        self.batch_size = batch_size
        self.results = {}

        # Initialize backend
        if backend == "mlx":
            import mlx.core as mx

            self.mx = mx
            self.device = mx.gpu
            print("MLX initialized on Metal GPU")
        elif backend == "torch":
            import torch

            self.torch = torch
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("PyTorch initialized on MPS")
            else:
                self.device = torch.device("cpu")
                print("PyTorch initialized on CPU")

    def benchmark_embedding(self, vocab_size: int = 50000, embed_dim: int = 768):
        """Benchmark embedding lookup."""
        print(
            f"\nBenchmarking embedding lookup ({vocab_size} vocab, {embed_dim} dim)..."
        )

        if self.backend == "mlx":
            import mlx.nn as nn

            embed = nn.Embedding(vocab_size, embed_dim)
            indices = self.mx.random.randint(0, vocab_size, (self.batch_size, 128))

            # Warmup
            for _ in range(10):
                _ = embed(indices)
                self.mx.eval(embed.weight)

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                output = embed(indices)
                self.mx.eval(output)
            elapsed = time.perf_counter() - start

        else:  # torch
            import torch.nn as nn

            embed = nn.Embedding(vocab_size, embed_dim).to(self.device)
            indices = self.torch.randint(
                0, vocab_size, (self.batch_size, 128), device=self.device
            )

            # Warmup
            for _ in range(10):
                _ = embed(indices)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                output = embed(indices)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_op = (elapsed / 100) * 1000
        self.results["embedding"] = ms_per_op
        print(f"  {ms_per_op:.2f}ms per batch")

    def benchmark_matmul(self, size: int = 768):
        """Benchmark matrix multiplication."""
        print(f"\nBenchmarking matmul ({self.batch_size}x{size} @ {size}x{size})...")

        if self.backend == "mlx":
            a = self.mx.random.normal((self.batch_size, size))
            b = self.mx.random.normal((size, size))

            # Warmup
            for _ in range(10):
                _ = a @ b
                self.mx.eval(a)

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                c = a @ b
                self.mx.eval(c)
            elapsed = time.perf_counter() - start

        else:  # torch
            a = self.torch.randn(self.batch_size, size, device=self.device)
            b = self.torch.randn(size, size, device=self.device)

            # Warmup
            for _ in range(10):
                _ = a @ b
                if self.device.type == "mps":
                    self.torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                c = a @ b
                if self.device.type == "mps":
                    self.torch.mps.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_op = (elapsed / 100) * 1000
        self.results["matmul"] = ms_per_op
        print(f"  {ms_per_op:.2f}ms per batch")

    def benchmark_transformer(self, seq_len: int = 128, hidden: int = 768):
        """Benchmark transformer-like operations."""
        print(
            f"\nBenchmarking transformer ops (batch={self.batch_size}, seq={seq_len}, hidden={hidden})..."
        )

        if self.backend == "mlx":
            # Simplified transformer op
            q = self.mx.random.normal((self.batch_size, seq_len, hidden))
            k = self.mx.random.normal((self.batch_size, seq_len, hidden))
            v = self.mx.random.normal((self.batch_size, seq_len, hidden))

            def attention(q, k, v):
                scores = (q @ k.swapaxes(-2, -1)) / (hidden**0.5)
                weights = self.mx.softmax(scores, axis=-1)
                return weights @ v

            # Warmup
            for _ in range(10):
                _ = attention(q, k, v)
                self.mx.eval(q)

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                output = attention(q, k, v)
                self.mx.eval(output)
            elapsed = time.perf_counter() - start

        else:  # torch
            q = self.torch.randn(self.batch_size, seq_len, hidden, device=self.device)
            k = self.torch.randn(self.batch_size, seq_len, hidden, device=self.device)
            v = self.torch.randn(self.batch_size, seq_len, hidden, device=self.device)

            def attention(q, k, v):
                scores = (q @ k.transpose(-2, -1)) / (hidden**0.5)
                weights = scores.softmax(dim=-1)
                return weights @ v

            # Warmup
            for _ in range(10):
                _ = attention(q, k, v)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                output = attention(q, k, v)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_op = (elapsed / 50) * 1000
        self.results["transformer"] = ms_per_op
        print(f"  {ms_per_op:.2f}ms per batch")

    def benchmark_tree_ops(self, num_nodes: int = 10000):
        """Benchmark MCTS tree operations (UCB calculation)."""
        print(f"\nBenchmarking tree ops (UCB for {num_nodes} nodes)...")

        if self.backend == "mlx":
            values = self.mx.random.uniform(0, 1, (num_nodes,))
            visits = self.mx.random.randint(1, 100, (num_nodes,))
            parent_visits = self.mx.full((num_nodes,), 1000)
            c_puct = 1.414

            def ucb(values, visits, parent_visits, c_puct):
                return values + c_puct * self.mx.sqrt(
                    self.mx.log(parent_visits) / visits
                )

            # Warmup
            for _ in range(10):
                _ = ucb(values, visits, parent_visits, c_puct)
                self.mx.eval(values)

            # Benchmark
            start = time.perf_counter()
            for _ in range(1000):
                scores = ucb(values, visits, parent_visits, c_puct)
                self.mx.eval(scores)
            elapsed = time.perf_counter() - start

        else:  # torch
            values = self.torch.rand(num_nodes, device=self.device)
            visits = self.torch.randint(
                1, 100, (num_nodes,), device=self.device, dtype=self.torch.float32
            )
            parent_visits = self.torch.full(
                (num_nodes,), 1000, device=self.device, dtype=self.torch.float32
            )
            c_puct = 1.414

            def ucb(values, visits, parent_visits, c_puct):
                return values + c_puct * self.torch.sqrt(
                    self.torch.log(parent_visits) / visits
                )

            # Warmup
            for _ in range(10):
                _ = ucb(values, visits, parent_visits, c_puct)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(1000):
                scores = ucb(values, visits, parent_visits, c_puct)
                if self.device.type == "mps":
                    self.torch.mps.synchronize()
            elapsed = time.perf_counter() - start

        ms_per_op = (elapsed / 1000) * 1000
        self.results["tree_ops"] = ms_per_op
        print(f"  {ms_per_op:.2f}ms per calculation")

    def run_all(self):
        """Run all benchmarks."""
        print(f"\n{'='*60}")
        print(
            f"Running benchmarks on {self.backend.upper()} (batch_size={self.batch_size})"
        )
        print(f"{'='*60}")

        # System info
        print("\nSystem Info:")
        print(f"  Platform: {platform.platform()}")
        print(f"  CPU: {psutil.cpu_count()} cores")
        print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")

        # Run benchmarks
        self.benchmark_embedding()
        self.benchmark_matmul()
        self.benchmark_transformer()
        self.benchmark_tree_ops()

        # Summary
        print(f"\n{'='*60}")
        print(f"Summary for {self.backend.upper()}:")
        for op, time_ms in self.results.items():
            print(f"  {op:15s}: {time_ms:8.2f}ms")
        print(f"{'='*60}\n")

        return self.results


def compare_backends(batch_size: int = 4096):
    """Compare MLX and PyTorch performance."""
    results = {}

    # Run MLX benchmarks
    try:
        mlx_bench = BenchmarkSuite("mlx", batch_size)
        results["mlx"] = mlx_bench.run_all()
    except ImportError:
        print("MLX not available, skipping...")

    # Run PyTorch benchmarks
    try:
        torch_bench = BenchmarkSuite("torch", batch_size)
        results["torch"] = torch_bench.run_all()
    except ImportError:
        print("PyTorch not available, skipping...")

    # Compare results
    if len(results) == 2:
        print("\nComparison (speedup: MLX vs PyTorch):")
        print(
            f"{'Operation':15s} {'MLX (ms)':>10s} {'PyTorch (ms)':>15s} {'Speedup':>10s}"
        )
        print("-" * 55)

        for op in results["mlx"]:
            mlx_time = results["mlx"][op]
            torch_time = results["torch"][op]
            speedup = torch_time / mlx_time
            print(f"{op:15s} {mlx_time:10.2f} {torch_time:15.2f} {speedup:10.2f}x")

    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "batch_size": batch_size,
                "platform": platform.platform(),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX vs PyTorch on M4 Pro")
    parser.add_argument(
        "--backend", choices=["mlx", "torch"], help="Backend to benchmark"
    )
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--compare", action="store_true", help="Compare all backends")

    args = parser.parse_args()

    if not IS_M4PRO:
        print("Warning: Not running on M4 Pro, results may vary")

    if args.compare:
        compare_backends(args.batch)
    elif args.backend:
        bench = BenchmarkSuite(args.backend, args.batch)
        bench.run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
