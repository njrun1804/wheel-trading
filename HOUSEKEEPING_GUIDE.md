# Project Housekeeping Guide

## Quick Instruction for Claude

To request housekeeping, simply say:
> "Please run project housekeeping according to HOUSEKEEPING_GUIDE.md"

Or for specific areas:
> "Please run housekeeping on the new files we just created"

## Housekeeping Principles

### 1. Project Structure Standards

```
wheel-trading/
├── Root (minimal - only essentials)
│   ├── Configuration files (.yaml, .toml, .txt)
│   ├── Core documentation (README, CLAUDE, guides)
│   ├── Primary entry points (run_aligned.py)
│   └── User-facing operational scripts
├── src/                    # All source code
├── tests/                  # ALL test files
├── examples/               # Organized by category
│   ├── core/              # Config, risk, validation
│   ├── data/              # Integration examples
│   └── auth/              # Authentication examples
├── tools/                  # Development utilities
│   ├── debug/             # Debugging tools
│   ├── analysis/          # Data analysis scripts
│   └── verification/      # System checks
├── deployment/            # Docker, cloud configs
├── scripts/               # Shell scripts
└── docs/archive/          # Old/outdated docs
```

### 2. File Placement Rules

#### Keep in Root
- Primary entry points (run_aligned.py)
- Core documentation (README.md, CLAUDE.md, *_GUIDE.md)
- Configuration files (config.yaml, my_positions.yaml)
- Build files (Makefile, pyproject.toml, requirements*.txt)
- User-facing operational scripts (daily_health_check.py, monitor_live.py)

#### Move to tests/
- ANY file starting with `test_`
- Must fix imports: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`

#### Move to tools/
- One-off analysis scripts
- Debug utilities
- Data fetching scripts
- Verification tools
- Scripts that duplicate main functionality

#### Move to examples/
- Files demonstrating usage (example_*.py)
- Template configurations
- Sample data files

#### Archive in docs/archive/
- Implementation summaries
- Status reports
- Migration guides
- Any documentation that's been consolidated

### 3. Import Fixing

When moving files, check and fix:

```python
# FROM (when in root):
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TO (when in subdirectory):
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Or for deeper nesting:
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add more .parent as needed
```

### 4. Cleanup Checklist

1. **Identify Violations**
   - [ ] Files with `test_` prefix not in tests/
   - [ ] Example files not in examples/
   - [ ] Utility scripts in root instead of tools/
   - [ ] Outdated documentation not archived

2. **Check for Duplicates**
   - [ ] Multiple scripts doing same thing
   - [ ] Overlapping documentation
   - [ ] Legacy vs new implementations

3. **Verify v2.0 Alignment**
   - [ ] Remove trading execution code
   - [ ] Keep recommendation-only focus
   - [ ] Ensure autonomous operation

4. **Fix Imports**
   - [ ] Test moved files compile
   - [ ] Update sys.path manipulations
   - [ ] Fix module imports for moved files

5. **Update References**
   - [ ] Update documentation if needed
   - [ ] Add deprecation warnings to legacy scripts
   - [ ] Ensure CI/CD configs still work

6. **Check for Hardcoded Values**
   - [ ] Search for numeric constants in source files
   - [ ] Identify execution-level parameters
   - [ ] Assess if values should be centralized
   - [ ] Consider if values should be "smart"/adaptive

7. **Make Values Smart**
   - [ ] Replace static thresholds with adaptive ones
   - [ ] Use ML/analytics for dynamic adjustment
   - [ ] Consider market conditions in limits
   - [ ] Implement confidence-based scaling

### 5. Decision Guidelines

#### When to Keep in Root
- Is it a primary user interface? (run_aligned.py)
- Is it run daily by users? (daily_health_check.py)
- Is it core configuration? (config.yaml)
- Is it essential documentation? (README.md)

#### When to Move to tools/
- Is it a debugging script?
- Is it a one-off analysis?
- Does it duplicate main functionality?
- Is it a development utility?

#### When to Archive
- Has it been superseded?
- Is it a status/summary document?
- Has it been consolidated into another doc?
- Is it no longer relevant to v2.0?

### 6. Centralization Guidelines

#### Values to Centralize in config.yaml
1. **Risk Limits**
   - Position size limits
   - Loss limits (daily, weekly, consecutive)
   - Margin/leverage limits
   - Greek exposure limits

2. **Strategy Parameters**
   - Delta targets and ranges
   - DTE targets and minimums
   - Roll triggers and thresholds
   - Strike selection intervals

3. **Performance Thresholds**
   - Minimum confidence scores
   - SLA targets (response times)
   - Cache TTL values
   - Retry counts and timeouts

4. **Market Condition Thresholds**
   - Volatility limits
   - Volume requirements
   - Liquidity minimums
   - Gap/shock thresholds

#### Values to Keep Local (with rationale)
1. **Math Constants**
   - Black-Scholes parameters
   - Greeks calculation constants
   - Numerical solver tolerances

2. **Technical Limits**
   - Database connection pools
   - API rate limits (provider-specific)
   - Thread pool sizes

3. **Debug/Development Values**
   - Log levels for specific modules
   - Test timeouts
   - Mock data parameters

### 7. Smart Value Implementation

#### Adaptive Value Patterns

1. **Market-Aware Scaling**
```python
# Instead of:
max_position_size = 0.20  # Static 20%

# Use:
max_position_size = config.risk.base_position_size * market_volatility_scalar
```

2. **Confidence-Based Adjustment**
```python
# Instead of:
min_delta = 0.10  # Always 10 delta

# Use:
min_delta = optimizer.get_dynamic_delta_bound(
    confidence_score, 
    market_state,
    historical_performance
)
```

3. **ML-Driven Parameters**
```python
# Instead of:
roll_threshold = 0.50  # Always roll at 50% profit

# Use:
roll_threshold = decision_engine.predict_optimal_roll_point(
    position_metrics,
    market_conditions,
    learned_patterns
)
```

4. **Regime-Adaptive Limits**
```python
# Instead of:
max_contracts = 10  # Fixed limit

# Use:
max_contracts = risk_regime.calculate_position_limit(
    account_size,
    market_regime,
    recent_performance
)
```

#### Implementation Checklist
- [ ] Identify all static thresholds
- [ ] Categorize by adaptability potential
- [ ] Design adaptation functions
- [ ] Add confidence scoring
- [ ] Implement fallback to static values
- [ ] Monitor adaptation effectiveness

### 8. Standard Housekeeping Report

After housekeeping, create a summary:

```markdown
## Housekeeping Summary

### Changes Made
- Moved X test files to tests/
- Moved Y scripts to tools/
- Archived Z documentation files
- Fixed imports in N files

### Final State
- Root directory: X files (was Y)
- Tests organized: X files
- Documentation consolidated: X guides

### Key Improvements
- [List major improvements]
```

## Usage Examples

### Full Housekeeping
> "Please run complete project housekeeping"

### After Adding Features
> "We just added new analytics modules and tests. Please run housekeeping on these new files."

### Documentation Cleanup
> "Please consolidate and archive outdated documentation"

### Test Organization
> "Please ensure all test files are properly organized in tests/"

## Automated Checking

### Run Housekeeping Check
```bash
make housekeeping-check
```

This command will:
- ✅ Check all test files are in tests/
- ✅ Check all example files are in examples/
- ✅ Check no status/summary docs in root
- ✅ Check data scripts are in tools/
- ✅ Check for empty directories
- 📊 Show project statistics

The check will exit with error code 1 if any violations are found, making it suitable for CI/CD pipelines.

### Search for Hardcoded Values

#### Common Patterns to Search
```bash
# Risk limits and thresholds
grep -r "max_.*=" src/ | grep -E "0\.[0-9]+|[0-9]+"
grep -r "min_.*=" src/ | grep -E "0\.[0-9]+|[0-9]+"
grep -r "threshold.*=" src/ | grep -E "0\.[0-9]+|[0-9]+"

# Default values in function signatures
grep -r "def.*=" src/ | grep -E "0\.[0-9]+|[0-9]+"

# Hardcoded timeouts and retries
grep -r "timeout.*=" src/ | grep -E "[0-9]+"
grep -r "retry.*=" src/ | grep -E "[0-9]+"

# Performance constants
grep -r "limit.*=" src/ | grep -E "[0-9]+"
grep -r "DEFAULT_" src/
```

#### Examples of Values to Centralize
```python
# BEFORE: Hardcoded in src/unity_wheel/risk/limits.py
max_position_pct: float = 0.20
max_consecutive_losses: int = 3
min_confidence: float = 0.30

# AFTER: In config.yaml
risk:
  limits:
    max_position_pct: 0.20
    max_consecutive_losses: 3
    min_confidence: 0.30
```

#### Examples of Smart Value Implementation
```python
# BEFORE: Static threshold
if volatility > 1.5:  # Hardcoded 150% vol limit
    return "Too volatile"

# AFTER: Adaptive threshold
volatility_limit = config.get_adaptive_limit(
    'max_volatility',
    base_value=1.5,
    market_conditions=market_state,
    confidence=confidence_score
)
if volatility > volatility_limit:
    return f"Volatility {volatility:.1%} exceeds adaptive limit {volatility_limit:.1%}"
```

---

Remember: The goal is a clean, intuitive structure that supports the v2.0 autonomous, recommendation-only architecture.
