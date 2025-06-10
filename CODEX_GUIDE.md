# CODEX_GUIDE.md

This guide outlines best practices for using OpenAI Codex with the Unity Wheel Trading Bot.

1. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run Pre-flight Checks**
   ```bash
   ruff check --select F,E,I .
   mypy --strict unity_trading data_pipeline ml_engine strategy_engine risk_engine app --ignore-missing-imports
   pytest -q tests/smoke
   ```

3. **Iterative Development**
   - Make changes inside `unity_trading/`, `data_pipeline/`, `ml_engine/`, `strategy_engine/`, `risk_engine/`, or `app/`.
   - Use `ruff format . && black .` and `ruff check --fix .` before committing.
   - Full tests: `pytest -q`.

4. **Submitting Changes**
   - Open a pull request with a clear summary of changes and test results.
   - Follow the commit hygiene in `CONTRIBUTING.md`.

For the latest repository status, see [CURRENT_STATE.md](CURRENT_STATE.md).
