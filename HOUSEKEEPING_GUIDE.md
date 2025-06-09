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

### 6. Standard Housekeeping Report

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

---

Remember: The goal is a clean, intuitive structure that supports the v2.0 autonomous, recommendation-only architecture.
