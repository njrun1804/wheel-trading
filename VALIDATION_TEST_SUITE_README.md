# Validation Test Suite

This comprehensive test suite validates that all critical fixes have been properly implemented after other agents complete their work.

## 🎯 Purpose

The validation test suite ensures that:

1. **Bolt initializes without errors** - Core system and orchestrator work correctly
2. **Einstein searches work without MergedResult errors** - No more search result issues
3. **No async subprocess warnings** - Clean async operations without warnings
4. **Accelerated tools are available to Bolt agents** - Hardware acceleration working
5. **Unified CLI works correctly** - Both Einstein and Bolt routing functional

## 📁 Files Created

### Core Test Files

- **`comprehensive_validation_test.py`** - Main validation test suite
- **`quick_validation_check.py`** - Quick assessment of current state  
- **`run_validation_tests.sh`** - Test runner script with multiple modes

### Test Categories

The comprehensive test suite includes 8 major test categories:

1. **Bolt System Initialization** - Core imports, config, orchestrator, analysis
2. **Einstein Search Functionality** - Imports, ResultMerger, UnifiedIndex, launcher
3. **Async Subprocess Warnings** - Clean async operations without warnings
4. **Accelerated Tools Availability** - All turbo tools accessible to Bolt
5. **Unified CLI Routing** - Query classification and routing accuracy
6. **Executable Commands** - bolt_executable and unified CLI commands work
7. **Database Connections** - DuckDB operations and file access
8. **Memory Management** - Memory cleanup and process monitoring

## 🚀 Usage

### Quick Check (Run Anytime)

Check the current state before other agents make fixes:

```bash
# Quick assessment of what needs fixing
./run_validation_tests.sh quick

# Or run directly
python3 quick_validation_check.py
```

### Comprehensive Validation (After Fixes)

Run after other agents complete their fixes:

```bash
# Full validation suite
./run_validation_tests.sh full

# With debug output for detailed errors
./run_validation_tests.sh debug

# Or run directly
python3 comprehensive_validation_test.py
python3 comprehensive_validation_test.py --debug
```

## 📊 Exit Codes

The comprehensive test suite returns meaningful exit codes:

- **0** - All tests passed! System ready for production
- **1** - Minor issues detected, system mostly functional  
- **2** - Several issues need attention before production
- **3** - Major issues detected, significant fixes needed

## 🧪 Test Details

### Bolt System Tests

- ✅ Core imports (config, solve, orchestrator)
- ✅ Configuration loading with agent count
- ✅ Orchestrator initialization  
- ✅ Simple analysis execution

### Einstein Search Tests

- ✅ Core imports (unified_index, result_merger)
- ✅ ResultMerger functionality (fixes MergedResult errors)
- ✅ UnifiedIndex search operations
- ✅ Einstein launcher integration

### Async Subprocess Tests

- ✅ Clean asyncio subprocess operations
- ✅ No async warnings generated
- ✅ Proper subprocess handling in async context

### Accelerated Tools Tests

- ✅ Ripgrep turbo availability
- ✅ Dependency graph turbo access
- ✅ Python analysis turbo functionality
- ✅ DuckDB turbo operations
- ✅ Trace turbo integration
- ✅ Python helpers turbo access
- ✅ Bolt integration with accelerated tools

### Unified CLI Tests

- ✅ Query router classification accuracy (>75% threshold)
- ✅ Einstein routing for simple queries
- ✅ Bolt routing for complex analysis queries
- ✅ CLI initialization and configuration

### Executable Tests

- ✅ bolt_executable status command works
- ✅ unified_cli.py help command works
- ✅ Proper exit codes and error handling

### Database Tests

- ✅ DuckDB connection and operations
- ✅ Database file access and table listing
- ✅ Memory database operations

### Memory Management Tests

- ✅ Memory cleanup after operations
- ✅ Garbage collection effectiveness
- ✅ Process monitoring (CPU, memory usage)

## 🔍 Validation Criteria

### Success Thresholds

- **Bolt Initialization**: All core components must load successfully
- **Einstein Search**: No MergedResult errors, proper result structures
- **Async Operations**: Zero async subprocess warnings detected
- **Accelerated Tools**: ≥80% of tools must be available and accessible
- **CLI Routing**: ≥75% routing accuracy on test queries
- **Executables**: Commands must return proper exit codes
- **Database**: Basic operations must work without errors
- **Memory**: <50MB memory increase during test operations

### Error Detection

The test suite detects and reports:

- Import failures and missing dependencies
- Configuration loading errors  
- Async warning generation
- Tool availability and integration issues
- Routing accuracy problems
- Command execution failures
- Database connection problems
- Memory leaks or excessive usage

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```bash
# Check Python path and dependencies
python3 -c "import sys; print('\\n'.join(sys.path))"
pip install -r requirements.txt
```

**Async Warnings**
```bash
# Run with warnings visible
python3 -W all comprehensive_validation_test.py --debug
```

**Database Issues**
```bash
# Check database files exist and are accessible
ls -la data/*.duckdb
python3 -c "import duckdb; print(duckdb.connect(':memory:').execute('SELECT 1').fetchone())"
```

**Memory Issues**
```bash
# Monitor memory during testing
python3 -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Debug Mode

For detailed error information:

```bash
./run_validation_tests.sh debug
```

This enables:
- Full stack traces for all errors
- Detailed async operation logging
- Memory usage tracking
- Extended error messages

## 📈 Continuous Validation

### Integration with Development Workflow

1. **Before fixes**: Run `quick_validation_check.py` to identify issues
2. **During development**: Monitor specific test categories
3. **After fixes**: Run full `comprehensive_validation_test.py`
4. **Before production**: Ensure 100% pass rate on critical tests

### Automated Testing

The test suite can be integrated into CI/CD pipelines:

```bash
# In CI/CD script
./run_validation_tests.sh full
if [ $? -eq 0 ]; then
    echo "✅ All validation tests passed - deploying"
else
    echo "❌ Validation tests failed - blocking deployment"
    exit 1
fi
```

## 🎉 Success Indicators

When all fixes are complete, you should see:

```
🧪 COMPREHENSIVE VALIDATION TEST RESULTS
=======================================
⏱️  Duration: X.XX seconds
✅ Passed: 25+
❌ Failed: 0
⏭️  Skipped: 0
📊 Success Rate: 100.0%

🎯 OVERALL ASSESSMENT:
🎉 ALL TESTS PASSED! System is ready for production.
```

This indicates that:
- Bolt initializes cleanly without errors
- Einstein searches work without MergedResult issues
- No async subprocess warnings are generated
- All accelerated tools are available to Bolt agents
- Unified CLI correctly routes queries to Einstein and Bolt

The system is then ready for production use with all critical fixes validated and working correctly.