# CLAUDE.md - Unity Wheel Trading Bot v2.2

Options wheel strategy for Unity Software (U). This file is kept minimal to save context tokens.

## 🚀 Quick Start
```bash
python run.py -p 100000               # Get recommendation
python run.py --diagnose              # System health
pytest -v -m "not slow"               # Fast tests
```

## 📍 Core Navigation
- **Entry**: `run.py` → `src/unity_wheel/cli/run.py:112`
- **Logic**: `src/unity_wheel/api/advisor.py:106`
- **Math**: `src/unity_wheel/math/options.py:746`
- **Config**: `config.yaml`

## 🔧 Common Fixes
- "Invalid credentials" → `python scripts/setup-secrets.py`
- "Rate limit" → Wait 60s
- "No liquid strikes" → `export DATABENTO_SKIP_VALIDATION=true`

## 📊 Key Values
- Ticker: "U" | Delta: 0.30 | Position: 100% | Confidence: 0.30

## 📚 Detailed Documentation
For comprehensive information, please read:
- `docs/QUICK_REFERENCE.md` - Full command reference
- `docs/ARCHITECTURE.md` - System design
- `docs/INTEGRATION_GUIDE.md` - API integration
- `docs/ADVANCED_FINANCIAL_MODELING_GUIDE.md` - Math details

## 🎯 Design Principles
1. Every calculation returns (value, confidence)
2. NO synthetic data - real market data only
3. Use @with_recovery for external calls
4. Track performance with @timed_operation
