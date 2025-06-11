# HOUSEKEEPING_GUIDE.md

Unity Wheel Trading Bot v2.2 - File organization rules for autonomous operation.

## Agent Quick Start

- **No execution code** – only recommendations.
- **Return confidence scores** from all calculations.
- **Never hardcode** ticker "U"; use `config.unity.ticker`.

**Start each session**
```bash
./scripts/housekeeping.sh --unity-check
```

**Before committing**
```bash
./scripts/housekeeping.sh --check-staged
```

**Detailed violations**
```bash
./scripts/housekeeping.sh --explain --json
```

## Session Start Checklist

```bash
# 1. Ultra-quick Unity check (2 seconds)
./scripts/housekeeping.sh --unity-check

# 2. Standard check if issues found (5 seconds)
./scripts/housekeeping.sh --quick

# 3. Auto-fix if needed
./scripts/housekeeping.sh --fix --quiet

# Note: Use FULL check before commits (not --quick)
```

## Pre-Commit Checklist

```bash
# Run before EVERY commit
./scripts/housekeeping.sh --check-staged || exit 1
```

## Complete Commit Workflow

Use the automated commit workflow script for a comprehensive commit process:

```bash
# Full workflow: checks + commit + push + CI monitoring
./scripts/commit-workflow.sh -m "Add feature X"

# Quick commit (skip CI wait)
./scripts/commit-workflow.sh -m "Update documentation" -w

# Local commit only (no push)
./scripts/commit-workflow.sh -m "WIP: Testing changes" -n

# Skip checks for docs
./scripts/commit-workflow.sh -m "Fix typo in README" -s -w
```

### Workflow Steps:
1. **Housekeeping checks** - Ensures code organization
2. **Stage changes** - Adds all modified files
3. **Pre-commit hooks** - Runs formatters and validators
4. **Create commit** - With your message
5. **Push to GitHub** - Updates remote repository
6. **Monitor CI/CD** - Waits for tests to pass

### Options:
- `-m MSG` - Commit message (required)
- `-s` - Skip housekeeping checks
- `-n` - No push (local commit only)
- `-w` - No wait (skip CI/CD monitoring)
- `-y` - Auto-yes to all prompts (see note below)

### Auto-Yes Coverage (-y flag):
**✅ Automatically handles:**
- Git operations (add, commit, push, merge)
- SSH host verification
- File overwrites
- Pre-commit hook changes
- All script confirmations

**❌ Cannot auto-approve:**
- Claude Code prompts ("Can I check/create/run...?")
- GitHub OAuth logins
- Security prompts requiring passwords

See `AUTO_YES_NOTE.md` for full details.

### Manual Alternative:
If you prefer manual control:
```bash
./scripts/housekeeping.sh        # Check issues
git add -A                       # Stage all
pre-commit run --all-files       # Run hooks
git commit -m "Your message"     # Commit
git push                         # Push
gh run watch                     # Watch CI
```

## Core Rules (Non-Negotiable)

### 1. File Placement - EXACT PATTERNS

| Pattern                    | MUST GO TO                  | Example                                                              |
| -------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `test_*.py` or `*_test.py` | `tests/`                    | `test_risk.py` → `tests/test_risk.py`                                |
| `adaptive_*.py`            | `src/unity_wheel/adaptive/` | `adaptive_sizing.py` → `src/unity_wheel/adaptive/adaptive_sizing.py` |
| `fetch_*.py`               | `tools/data/`               | `fetch_chains.py` → `tools/data/fetch_chains.py`                     |

### 2. Root Directory - ALLOWED FILES

```
run.py                   # Main entry point
config.yaml              # Configuration
my_positions.yaml        # User positions
README.md, CLAUDE.md     # Core docs
*_GUIDE.md               # Guides (HOUSEKEEPING_GUIDE, etc.)
pyproject.toml           # Poetry config
requirements*.txt        # Dependencies
Makefile                 # Build automation
SESSION_START.txt        # Pre-task checklist
SESSION_END.txt          # Post-task checklist
```

Python scripts starting with specific patterns MUST be in subdirectories!

### 3. Unity-Specific Structure

```
src/unity_wheel/
├── adaptive.py    # Volatility-based sizing (OR adaptive/ directory)
├── strategy/      # Wheel logic (NO EXECUTION)
├── risk/          # VaR, CVaR calculations
├── schwab/        # Data fetch only (NO TRADING)
└── databento/     # Options chains
```

## Confidence Score Compliance

Math and risk functions in `src/unity_wheel/math/` and `src/unity_wheel/risk/` that perform calculations MUST return confidence scores:

```python
# WRONG
def black_scholes_price(S, K, T, r, sigma):
    return price

# CORRECT (using CalculationResult)
def black_scholes_price_validated(S, K, T, r, sigma):
    price = calculate_price(...)
    confidence = 0.99 if inputs_valid else 0.0
    return CalculationResult(value=price, confidence=confidence)
```

The script specifically checks functions like:
- `black_scholes*`
- `calculate_greeks*`
- `calculate_var*`
- `calculate_iv*`
- `calculate_risk*`

Note: General getters/fetchers (like `get_config()`, `fetch_data()`) do NOT need confidence scores.

## Hardcoded Values Check

NO HARDCODED:
- Position sizes (use adaptive system)
- Unity ticker "U" (use config)
- Volatility thresholds (use config)
- Time constants > 1 (use config)

```bash
# Quick scan (should return NOTHING or only legitimate uses)
# Check for hardcoded position sizes in trading logic
rg -P "(position_size|num_contracts|contract_count)\s*=\s*[0-9]+(?!\s*#)(?!.*\*)" src/

# Check for hardcoded 'U' ticker in assignments (not docs/config)
rg -P "(symbol|ticker|underlying)\s*=\s*['\"]U['\"]" src/ | grep -v "Field" | grep -v "example"

# Check for hardcoded volatility thresholds
rg "if.*volatility.*[<>].*[0-9]" src/ --type py | grep -v config
```

## Import Fix Pattern (One Way Only)

```python
# At TOP of any moved file
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N = directory depth
```

Depth examples:
- `tests/test_foo.py` → `parents[1]`
- `tools/debug/bar.py` → `parents[2]`
- `tests/unit/baz.py` → `parents[2]`

## Automated Enforcement

### housekeeping.sh v2.2 Features

```bash
# Show all options
./scripts/housekeeping.sh --help

# Key modes
--fix             # Auto-fix file placement
--json            # Machine-parseable output
--quick           # Skip expensive checks (5 sec)
--check-staged    # Pre-commit mode
--dry-run         # Preview changes without applying
--explain         # Include violation details
--unity-check     # Quick Unity-specific validation (2 sec)

# Exit codes
0 = Success (no issues)
1 = Non-critical (can be auto-fixed)
2 = CRITICAL (execution code found)
```

### Performance Optimizations (v2.2)

- Combined pattern matching - Single pass per file
- Pre-flight validation - Fails fast if not in Unity repo
- Unity-check mode - Ultra-fast critical checks only
- Uses ripgrep if available (10x faster)
- OS-aware (Mac/Linux compatible)

### When to Use Each Mode

| Mode | Use Case | Time | What It Checks |
|------|----------|------|----------------|
| `--unity-check` | Session start | 2 sec | Execution code, adaptive system, Unity config |
| `--quick` | During development | 5 sec | File placement + basic Unity compliance |
| *(no flag)* | **Before commits** | 10-15 sec | Everything including confidence scores |
| `--check-staged` | Pre-commit hook | Varies | Only staged files (full checks) |

### Critical Unity Checks

1. Execution Code (exit 2)
   - `execute_trade`, `place_order`, `submit_order`
   - `broker.execute`, `broker.place`
   - Cannot be auto-fixed
   - Blocks all commits

2. File Placement (exit 1)
   - Auto-fixable with `--fix`
   - Updates imports automatically

3. Unity Compliance (warnings)
   - Hardcoded ticker "U"
   - Static position sizes
   - Missing confidence scores

### Quick Fix Commands

```bash
# See what would be fixed (v2.1+)
./scripts/housekeeping.sh --fix --dry-run

# Auto-fix all placement issues
./scripts/housekeeping.sh --fix

# Get detailed violation report
./scripts/housekeeping.sh --explain --json

# Extract a hardcoded value to config
# Add to config.yaml:
risk:
  limits:
    max_volatility: 1.0
# Then update code:
sed -i 's/MAX_VOLATILITY = 1.0/config.risk.limits.max_volatility/g' src/file.py
```

## CI/CD Integration

```yaml
# .github/workflows/housekeeping.yml
name: Housekeeping
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Housekeeping
        run: |
          ./scripts/housekeeping.sh --check-staged
          # Fail if execution code exists
          ! grep -r "execute_trade\|place_order" src/
```

## Common Violations & Fixes

### 1. Test in Wrong Place
```bash
# Fix
git mv test_adaptive.py tests/
echo 'from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))' | cat - tests/test_adaptive.py > temp && mv temp tests/test_adaptive.py
```

### 2. Hardcoded Unity Reference
```python
# Before
ticker = "U"

# After
ticker = config.unity.ticker
```

### 3. Missing Confidence Score
```python
# Before
def fetch_positions():
    return schwab_client.get_positions()

# After
def fetch_positions():
    try:
        positions = schwab_client.get_positions()
        confidence = 1.0 if positions.timestamp > time.time() - 300 else 0.5
        return positions, confidence
    except Exception:
        return [], 0.0
```

### 4. Execution Code (CRITICAL)
```python
# NEVER HAVE THIS
def execute_trade(order):  # DELETE ENTIRELY
    broker.place_order(order)

# ONLY THIS
def recommend_trade(params):
    recommendation = strategy.analyze(params)
    return recommendation, confidence
```

## Exceptions

The script automatically excludes:
- `.venv/*`, `venv/*`, `*env/*`
- `__pycache__/*`
- `build/*`, `dist/*`
- `.git/*`

No configuration file needed - these are hardcoded exclusions.

## Health Score

```bash
# Human-readable
./scripts/housekeeping.sh

Unity Wheel Bot Housekeeping v2.2.0
Score: 92/100

⚠️  2 issues found
   Run with --fix to auto-resolve file placement
   Run with --explain for details

# Machine-readable
./scripts/housekeeping.sh --json

{
  "timestamp": "2025-01-15T09:00:00Z",
  "version": "2.2.0",
  "score": 92,
  "critical": false,
  "violations": {
    "test_files": 1,
    "missing_confidence": 1,
    "execution_code": 0,
    "adaptive_files": 0,
    "fetch_files": 0,
    "hardcoded_ticker": 0,
    "static_positions": 0
  }
}

# With details
./scripts/housekeeping.sh --json --explain
```

## Remember

1. Start every session: Run unity-check (2 seconds)
2. Before every commit: Run staged check (blocks bad commits)
3. No execution code: Recommendations only
4. Confidence scores: Every data function returns (result, confidence)
5. Use adaptive sizing: Never hardcode positions
6. Unity in config: Never hardcode ticker "U"

This is not optional - it's how we maintain autonomous operation.

## Complete Session Workflow

### Session Start
Paste at the beginning of every Claude Code session:
```
### UNITY WHEEL v2.2 — PRE-TASK CHECKS

**RUN FIRST (2 sec):**
```bash
./scripts/housekeeping.sh --unity-check
```
✓ Checks: No execution code, adaptive system exists, Unity config

**CRITICAL RULES:**
1. NO TRADING EXECUTION (recommendations only)
2. Every data function returns `(result, confidence_score)`
3. Unity ticker from `config.unity.ticker` only (line 238)
4. Files: `test_*.py` → `tests/`, `adaptive_*.py` → `src/unity_wheel/adaptive/`

**COMMIT GATE:**
```bash
./scripts/housekeeping.sh --check-staged || exit 1
```
--- BEGIN TASK ---
```

### Session End
When ready to commit your work:
```
### UNITY WHEEL v2.2 — SESSION END

**FINAL CHECKS:**
```bash
# Full housekeeping validation
./scripts/housekeeping.sh

# View changes
git status
```

**COMMIT & PUSH:**
```bash
# Full workflow (auto-yes to all prompts)
./scripts/commit-workflow.sh -m "Your commit message" -y

# Quick commit without CI wait
./scripts/commit-workflow.sh -m "Your commit message" -y -w

# Local only (no push)
./scripts/commit-workflow.sh -m "WIP: Your message" -y -n
```

*Note: -y auto-approves git/system prompts only, not Claude's "Can I...?" prompts*

--- END SESSION ---
```

### Quick Reference Files
- **SESSION_START.txt** - Copy/paste for session start
- **SESSION_END.txt** - Copy/paste for session end
- **./scripts/housekeeping.sh** - File organization enforcement
- **./scripts/commit-workflow.sh** - Complete commit automation
