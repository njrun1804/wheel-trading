# Harmonization Quick Start Guide

**🚀 Execute Immediate Cleanup in 15 Minutes**

This guide provides step-by-step instructions to begin codebase harmonization with zero-risk Phase 1 cleanup.

## Prerequisites

- Git repository with current changes committed
- Backup storage available (~3GB)
- Terminal access to project directory

## Quick Execution Steps

### 1. Safety Validation (2 minutes)
```bash
# Validate cleanup safety
cd /path/to/wheel-trading
python scripts/validate_cleanup_safety.py

# ✅ Proceed only if validation passes
```

### 2. Execute Phase 1 Cleanup (10 minutes)
```bash
# Run automated cleanup script
./scripts/phase1_emergency_cleanup.sh

# Monitor output for any issues
```

### 3. Verification (3 minutes)
```bash
# Test basic functionality
python run.py --diagnose

# Run quick test suite
pytest tests/ -v -x --tb=short

# Check git status
git status
```

## Expected Results

### Before Cleanup
- **Total files**: ~3,000+
- **Documentation**: 291 files
- **Test files**: 504 files
- **Storage**: ~15GB

### After Phase 1
- **Total files**: ~2,400 files (20% reduction)
- **Documentation**: ~50 files (83% reduction)
- **Test files**: ~300 files (40% reduction)
- **Storage**: ~10GB (33% reduction)

## Safety Features

### Automatic Backups
All removed files automatically backed up to:
```
consolidation_backups/YYYYMMDD_HHMMSS/
├── docs_backup/           # All documentation
├── tests_backup/          # All test files
├── config_backup/         # All configuration
├── databases_backup/      # All databases
└── cleanup_report.md      # Detailed report
```

### Recovery Commands
```bash
# Restore specific category
cp -r consolidation_backups/*/docs_backup/* ./

# Restore everything
cp -r consolidation_backups/*/. ./

# Git-based recovery
git checkout HEAD -- <file_path>
```

### Protected Files
These files are NEVER removed:
- `README.md`, `CLAUDE.md`, `ARCHITECTURE.md`
- `src/unity_wheel/**/*.py` (core source code)
- `config.yaml`, `pyproject.toml`
- `requirements*.txt`
- Main test files (without suffixes like `_simple`)

## Cleanup Categories

### 1. Archive Directories ✅ ZERO RISK
```
Removes:
- archive/old_orchestrator_tests/
- jarvis2/backup_*/
- backups/*/

Impact: ~2GB storage saved, 47 directories removed
```

### 2. Documentation Consolidation ✅ LOW RISK
```
Removes files matching:
- *_SUMMARY.md
- *_REPORT.md  
- *_COMPLETE.md
- *_VALIDATION.md
- *_FINAL.md

Preserves: README.md, CLAUDE.md, ARCHITECTURE.md, LICENSE
Impact: 200+ redundant docs removed
```

### 3. Test File Consolidation ✅ LOW RISK
```
Removes redundant patterns:
- test_*_simple.py (when test_*.py exists)
- test_*_validation.py
- test_*_old.py
- test_*_backup.py

Preserves: Main test files for each component
Impact: 150+ redundant tests removed
```

### 4. Configuration Cleanup ✅ LOW RISK
```
Removes:
- mcp-servers-*.json duplicates
- *.backup files
- Duplicate YAML configs

Preserves: config.yaml, pyproject.toml, requirements.txt
Impact: 30+ duplicate configs removed
```

### 5. Database Cleanup ✅ MEDIUM RISK
```
Archives development databases:
- *.db files outside /data/ directory
- Test and temporary databases

Preserves: Production databases in /data/
Impact: ~1GB temporary data archived
```

## Verification Checklist

After cleanup completion, verify:

- [ ] System starts: `python run.py --diagnose`
- [ ] Tests pass: `pytest tests/ -x` 
- [ ] Core functionality: Basic trading analysis works
- [ ] Configuration loads: No config errors
- [ ] Essential files present: All protected files exist

## Rollback Procedures

### Immediate Rollback (if issues found)
```bash
# 1. Stop all running processes
pkill -f "python.*wheel"

# 2. Restore from backup  
BACKUP_DIR="consolidation_backups/$(ls -1t consolidation_backups/ | head -1)"
cp -r "$BACKUP_DIR"/* ./

# 3. Verify restoration
python run.py --diagnose
```

### Selective Rollback
```bash
# Restore only documentation
cp -r consolidation_backups/*/docs_backup/* ./

# Restore only tests
cp -r consolidation_backups/*/tests_backup/* ./

# Restore only configuration
cp -r consolidation_backups/*/config_backup/* ./
```

### Git-based Recovery
```bash
# See what was removed
git status

# Restore specific files
git checkout HEAD -- path/to/file

# Restore entire directories
git checkout HEAD -- path/to/directory/
```

## Next Steps After Phase 1

### Immediate (Same Day)
1. **Monitor system stability** for 2-4 hours
2. **Run full test suite** to identify any regressions
3. **Check performance metrics** to ensure no degradation

### Next Week (Phase 2 Preparation)
1. **Review cleanup report** for insights
2. **Plan system integration** (Einstein, Bolt, Memory systems)
3. **Prepare Phase 2 testing strategy**

### Following Weeks (Phases 3-5)
1. **Configuration harmonization** (Week 4)
2. **Architecture refactoring** (Week 5) 
3. **Performance optimization** (Week 6)

## Troubleshooting

### Common Issues

**Issue**: "Permission denied" errors
```bash
# Solution: Fix permissions
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

**Issue**: "Module not found" errors after cleanup
```bash
# Solution: Check if essential imports were removed
python scripts/validate_cleanup_safety.py
# Restore from backup if needed
```

**Issue**: Tests failing after cleanup
```bash
# Solution: Restore test files selectively
cp -r consolidation_backups/*/tests_backup/* ./
# Re-run cleanup with different test patterns
```

**Issue**: Configuration errors
```bash
# Solution: Restore configuration files
cp -r consolidation_backups/*/config_backup/* ./
```

### Emergency Contacts

If major issues occur:
1. **Stop all processes**: `pkill -f python`
2. **Full restore**: `cp -r consolidation_backups/latest/* ./`
3. **Verify restoration**: `python run.py --diagnose`
4. **Report issue**: Document what happened for analysis

## Success Metrics

You've successfully completed Phase 1 if:

- ✅ System functionality unchanged
- ✅ Test suite passes (>95% pass rate)
- ✅ Performance maintained or improved
- ✅ Storage reduced by 30%+
- ✅ File count reduced by 20%+
- ✅ No critical functionality lost

## Advanced Options

### Custom Exclusions
Edit `scripts/phase1_emergency_cleanup.sh` to exclude specific files:
```bash
# Add to PROTECTED_PATTERNS array
PROTECTED_PATTERNS+=(
    "my_important_file.md"
    "special_test_*.py"
)
```

### Verbose Logging
```bash
# Run with detailed logging
./scripts/phase1_emergency_cleanup.sh 2>&1 | tee cleanup.log
```

### Dry Run Mode
```bash
# Preview what would be removed (modify script)
DRY_RUN=true ./scripts/phase1_emergency_cleanup.sh
```

---

**Ready to proceed?** Run the safety validation, then execute Phase 1 cleanup! 🚀