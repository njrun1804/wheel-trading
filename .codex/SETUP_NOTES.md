# Setup Notes & Lessons Learned

## The Math Module Conflict

This project has `unity_trading.math` which conflicts with Python's stdlib `math` module.
This breaks pip and many imports when the project is in PYTHONPATH.

### Solutions:

1. **Install packages BEFORE setting PYTHONPATH**
2. **Install from /tmp directory** to avoid the conflict entirely
3. **Use `-S` flag carefully** - it prevents finding pip module
4. **Pure Python mode works fine** - don't stress if NumPy won't install

## Container Test Results

From the test runs, we learned:
- Python 3.12.10 is pre-installed ✓
- pip can be installed via curl if needed ✓
- The math module conflict is real and breaks pip
- Installing from /tmp directory works best
- Pure Python fallbacks are sufficient for development

## Recommended Approach

For containers, use this order:
1. DON'T set PYTHONPATH initially
2. cd to /tmp
3. Install packages with pip
4. cd back to project
5. NOW set PYTHONPATH
6. Set USE_PURE_PYTHON=true if packages fail

## Quick Commands

```bash
# Fastest setup
source .codex/instant_setup.sh

# Full setup with all features
./.codex/container_setup.sh

# Verify what's working
python3 .codex/verify_setup.py

# If all else fails - Pure Python works!
export USE_PURE_PYTHON=true
export USE_MOCK_DATA=true
export PYTHONPATH="$(pwd):$PYTHONPATH"
```
