# GitHub Actions Workflows - Optimized CI/CD

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

## Optimized Workflow Structure (NEW)

### Primary Workflows

1. **ci-unified.yml** (NEW - Recommended) ⚡
   - Replaces both `ci.yml` and `ci-optimized.yml`
   - Runs in ~3 minutes (vs 8 minutes previously)
   - Features:
     - Path filtering to skip non-code changes
     - Parallel test execution across categories
     - Dependency caching and artifact sharing
     - Performance benchmark tracking
     - SLA compliance checking

2. **ci-fast.yml** 🚀
   - Quick pre-commit checks only
   - Runs in <1 minute
   - Use for: Rapid feedback on formatting/linting

3. **security.yml** (NEW) 🔒
   - Optimized CodeQL scanning (10 min timeout)
   - Dependency vulnerability checks (Safety, Bandit)
   - Secret scanning (TruffleHog)
   - Runs on PRs and weekly schedule

### Performance Improvements

| Optimization | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Path Filtering | Run on all changes | Skip docs-only changes | 30% fewer runs |
| Parallel Tests | Sequential (8 min) | 3 parallel groups | 3x faster |
| Dependency Caching | Download every time | Cache + artifacts | 60s saved |
| CodeQL Config | Scan everything | Scan src/ only | 50% faster |
| Pre-compilation | JIT compilation | Pre-compiled .pyc | 10% faster imports |

### When to Use Each Workflow

| Workflow | Trigger | Use Case | Runtime |
|----------|---------|----------|---------|
| ci-unified | PR/Push | Full CI suite | ~3 min |
| ci-fast | PR/Push | Quick checks | <1 min |
| security | PR/Weekly | Security scans | ~5 min |
| ci (legacy) | Manual | Fallback option | ~8 min |

### Key Optimizations

1. **Path Filtering**
   ```yaml
   paths:
     - 'src/**'
     - 'tests/**'
     - 'pyproject.toml'
   ```

2. **Test Parallelization**
   ```yaml
   strategy:
     matrix:
       test-suite:
         - "unit-fast"      # 30s
         - "unit-slow"      # 1m
         - "integration"    # 1.5m
   ```

3. **Smart Caching**
   - Poetry installation cached
   - Virtual environment as artifact
   - Pre-compiled Python bytecode

4. **CodeQL Optimization**
   - Custom config in `.github/codeql/codeql-config.yml`
   - Limited scan paths
   - Focused query sets

## Migration Guide

To use the new optimized workflows:

1. **For new PRs**: Workflows run automatically
2. **To disable old workflows**:
   ```bash
   gh workflow disable "CI"
   gh workflow disable "CI Optimized"
   ```
3. **To monitor performance**:
   ```bash
   gh run list --workflow="CI Unified" --limit=10
   ```

## Troubleshooting

### Common Issues

1. **Import errors after optimization**
   - Clear caches: `gh cache delete --all`
   - Check artifact uploads in build job

2. **CodeQL timeouts**
   - Verify paths in `.github/codeql/codeql-config.yml`
   - Reduce query scope if needed

3. **Test failures in parallel**
   - Some tests may have hidden dependencies
   - Add to same test group if needed
