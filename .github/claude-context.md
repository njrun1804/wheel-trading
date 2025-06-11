# Claude Context for Wheel Trading Bot

## What This Is
A personal wheel trading recommendation system with manual position entry.
- **No broker integration** - all positions entered manually
- **Local-only** - runs on macOS, no cloud deployment
- **Single user** - personal trading tool

## Architecture Overview
```
run.py → advisor.py → wheel.py → options.py
                ↓
        decision_engine.py
                ↓
        recommendations
```

## Key Design Decisions
1. **Manual Entry**: No automated trading execution
2. **DuckDB**: Single local database for all data
3. **Minimal Dependencies**: Only essential packages
4. **Unity (U)**: Primary trading symbol

## Common Tasks
- Add new risk check: See `src/unity_wheel/risk/`
- Modify strategy: Edit `src/unity_wheel/strategy/wheel.py`
- Change parameters: Update `config.yaml`

## Performance Notes
- Data cached in `data/wheel_trading_master.duckdb`
- Options calculations optimized for Unity symbol
- Test with `pytest -v -m "not slow"` for quick feedback

## Development Workflow
1. Make changes locally
2. Test with pytest
3. Run `python run.py -p 100000` to verify
4. Commit with descriptive message
