# Contributing Guidelines

This project uses an automated housekeeping script to enforce file placement and other hygiene rules.

## Pre-commit workflow

1. Stage your changes.
2. Run `./scripts/housekeeping.sh --check-staged` to validate staged files.
   - If any files are flagged as misplaced, follow the script's suggestions:
     - Move the file to the indicated directory using `git mv`.
     - At the top of the moved file, insert:
       ```python
       from pathlib import Path
       import sys
       sys.path.insert(0, str(Path(__file__).resolve().parents[N]))
       ```
       Replace `N` with the appropriate directory depth as shown in `HOUSEKEEPING_GUIDE.md`.
   - Re-run the script until it reports "All checks passed!".
3. Run the standard formatting and testing commands:
   ```bash
   ruff format . && black .
   ruff check --fix .
   mypy --strict src/unity_wheel data_pipeline app --ignore-missing-imports
   pytest -q
   ```
4. Commit your changes once all checks pass.

Refer to `HOUSEKEEPING_GUIDE.md` for full details on the import fix pattern and other rules.

### Package Structure
Legacy directories `ml_engine`, `strategy_engine`, and `risk_engine` were removed.
Use the `src/unity_wheel` package for all imports.
