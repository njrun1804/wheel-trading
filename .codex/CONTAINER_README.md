# 🐳 Codex Container Setup Guide

## ✅ Setup Status: SUCCESSFUL!

All Python packages installed successfully:
- numpy 2.3.0 ✓
- pandas 2.3.0 ✓
- scipy 1.15.3 ✓
- pydantic 2.11.5 ✓

## Quick Commands

```bash
# Check current status
./.codex/status.sh

# Quick setup (if needed)
source .codex/instant_setup.sh

# Verify setup
python3 .codex/verify_setup.py
```

## Known Issues & Solutions

### Math Module Conflict

This project has a `src.unity_wheel.math` module that conflicts with Python's standard library `math` module. This can prevent pip from working properly.

**The setup scripts handle this automatically by:**
1. Installing packages BEFORE adding the project to PYTHONPATH
2. Using Python's `-S` flag to skip site packages during installation
3. Setting PYTHONPATH only after packages are installed

**If you still have issues:**
```bash
# Run the import fix script
./.codex/fix_imports.sh

# Or manually install in a clean shell:
cd /tmp
python3 -m pip install numpy pandas scipy pydantic
cd -
source .codex/.env
```

## Available Scripts

### 🚀 container_setup.sh
Full setup with dependency installation, environment configuration, and validation tests.

### 🏃 quick_setup.sh
Minimal setup that just installs packages and sets environment variables.

### 🔧 fix_imports.sh
Fixes import conflicts using alternative installation methods.

### 🔍 diagnose.sh
Checks your environment and helps debug issues:
```bash
./.codex/diagnose.sh
```

### 🧪 container_test.sh
Tests that everything is working:
```bash
./.codex/container_test.sh
```

### 💾 container_commit.sh
Quick git commit for container changes:
```bash
./.codex/container_commit.sh "your commit message"
```

## Environment Variables

The setup creates these environment variables:
- `USE_MOCK_DATA=true` - Use mock data instead of live APIs
- `OFFLINE_MODE=true` - Work without internet connection
- `DATABENTO_SKIP_VALIDATION=true` - Skip data provider validation
- `USE_PURE_PYTHON=true/false` - Set based on package availability
- `CONTAINER_MODE=true` - Indicates running in container

## Troubleshooting

### "No module named 'numpy'" during pip install
This is the math module conflict. Run:
```bash
./.codex/fix_imports.sh
```

### Import errors when running code
Make sure you've sourced the environment:
```bash
source .codex/.env
# or
source .codex/activate_container.sh
```

### Tests fail with import errors
This is OK! The code has pure Python fallbacks. You can still optimize the code.

### Need to install additional packages
Install them BEFORE sourcing the project environment:
```bash
# In a new shell (no .env sourced)
pip install package_name

# Then return and source
source .codex/.env
```

## Pure Python Mode

If packages can't be installed, the code automatically uses pure Python fallbacks:
- Math calculations use stdlib math instead of numpy
- Data structures use lists/dicts instead of pandas
- This is slower but fully functional for development

## For New Shell Sessions

Always activate the environment:
```bash
source .codex/.env
# or
source .codex/activate_container.sh
```

## Tips

1. **Check your environment**: Run `./.codex/diagnose.sh` if something seems wrong
2. **Use mock data**: The container is configured for offline development
3. **Commit often**: Use `./.codex/container_commit.sh "message"` for quick commits
4. **Pure Python is OK**: Don't worry if numpy/pandas won't install - fallbacks work
