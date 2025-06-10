# GitHub Actions Workflow Optimization

## Runner Configuration

This repository uses GitHub's larger hosted runners for improved CI/CD performance:

### Runner Sizes

- **ubuntu-latest** (2 vCPU, 7GB RAM) - Basic jobs, simple scripts
- **ubuntu-latest-4-cores** (4 vCPU, 16GB RAM) - Medium workloads: linting, type checking, validation
- **ubuntu-latest-8-cores** (8 vCPU, 32GB RAM) - Heavy workloads: integration tests, risk analysis
- **ubuntu-latest-16-cores** (16 vCPU, 64GB RAM) - Available for extreme workloads
- **ubuntu-latest-32-cores** (32 vCPU, 128GB RAM) - Available for extreme workloads
- **ubuntu-latest-64-cores** (64 vCPU, 256GB RAM) - Available for extreme workloads

### Performance Improvements

With larger runners, we see:
- **3-8x faster test execution** due to more CPU cores
- **Reduced queuing** with max-parallel: 40
- **Better caching** with more memory available
- **Faster dependency installation** with parallel downloads

### Cost Optimization

Larger runners are billed per-minute, so we:
1. Use appropriate sizes for each job (don't over-provision)
2. Enable fail-fast to stop on first failure
3. Use concurrency groups to cancel redundant runs
4. Cache aggressively to reduce setup time

### Workflow Structure

1. **ci-fast.yml** - Quick checks (4-core runners, <1 min)
   - Linting, formatting, security scanning
   - Run in parallel with no dependencies

2. **ci-optimized.yml** - Parallel test matrix (4-8 core runners)
   - Split tests by type (math, risk, integration)
   - Platform-specific optimizations

3. **ci.yml** - Main workflow (8-core for Ubuntu, standard for macOS)
   - Comprehensive testing
   - Final validation

## Concurrency Settings

- **max-parallel: 40** - Utilize organization's high concurrency limit
- **cancel-in-progress: true** - Cancel old runs when new commits are pushed
- **concurrency groups** - Prevent duplicate runs on same branch

## Best Practices

1. Monitor usage in Actions tab to optimize runner sizes
2. Use matrix strategy to parallelize independent tests
3. Place heaviest jobs on largest runners
4. Keep macOS on standard runners (no larger options)
5. Use caching to minimize setup time on every run
