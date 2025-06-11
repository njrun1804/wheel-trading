# CLAUDE.md - Unity Wheel Trading Bot v2.2

Minimal context file. For details, read referenced docs below.

## Quick Commands
```bash
python run.py -p 100000          # Get recommendation
python run.py --diagnose         # System health
pytest -v -m "not slow"          # Fast tests
```

## Key Files
- `advisor.py:106` - Main logic
- `wheel.py:153` - Strike selection
- `options.py:746` - Math
- `config.yaml` - Settings

## Reference Docs
When you need details, read these:
- `docs/QUICK_REFERENCE.md` - Full commands & workflows
- `docs/ARCHITECTURE.md` - System design
- `docs/DATABENTO_UNITY_GUIDE.md` - Data integration
- `CLAUDE-BACKUP-FULL.md` - Original 949-line reference

## Critical Values
Unity (U) | Delta: 0.30 | Max position: 100% | Min confidence: 0.30
