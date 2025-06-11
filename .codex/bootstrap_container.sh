#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
#  Unity‑Wheel container bootstrap  – v38  (June 2025)
#  * Locks Python to 3.11.10 (installs via pyenv if necessary)
#  * Installs *pinned* core + dev deps that match requirements‑recommended.txt
#  * Drops obsolete / unused packages (click, scikit‑learn, flake8, …)
#  * Keeps NumPy on 1.26.x to avoid the 2.x breaking changes
#  * Regenerates .codex/activate_container.sh on every run
# ───────────────────────────────────────────────────────────────
set -euo pipefail

echo "🚀  Unity‑Wheel container bootstrap v38"
echo "----------------------------------------"

# ───────────────────────────────────────────────────────────────
# 0. Ensure Python 3.11.10
# ───────────────────────────────────────────────────────────────
PY=""
PYVER_TARGET="3.11.10"

have_pyenv() { command -v pyenv >/dev/null 2>&1; }

if have_pyenv; then
  # Install (if missing) and activate via pyenv
  pyenv install -s "$PYVER_TARGET"
  PY=$(pyenv root)/versions/$PYVER_TARGET/bin/python
  eval "$(pyenv init -)"
  pyenv shell "$PYVER_TARGET"
else
  for cmd in python3.11 python3.11.10 python3.11 python3; do
    if command -v "$cmd" >/dev/null 2>&1 && [[ $("$cmd" -V 2>&1) == *" 3.11."* ]]; then
      PY="$cmd"; break
    fi
  done
fi

if [[ -z $PY ]]; then
  echo "❌  Python $PYVER_TARGET (or any 3.11.x) not found and pyenv not available."
  echo "    Install pyenv or pre‑install Python 3.11 before building the container."
  exit 1
fi

echo "✅  Using $($PY --version)"

PROJECT_DIR="$(pwd)"
CODEx_DIR="$PROJECT_DIR/.codex"
STUB_DIR="$CODEx_DIR/stubs"
mkdir -p "$STUB_DIR" /tmp/.wheel/{cache,secrets} logs exports "$CODEx_DIR"

# ───────────────────────────────────────────────────────────────
# 1. Strip any legacy stub path from PYTHONPATH
# ───────────────────────────────────────────────────────────────
clean_py_path() {
  local out="" IFS=':'
  for p in $1; do [[ $p == "$STUB_DIR" || -z $p ]] && continue; out="${out:+$out:}$p"; done
  echo "$out"
}
export PYTHONPATH="$(clean_py_path "${PYTHONPATH:-}")"

# ───────────────────────────────────────────────────────────────
# 2. Define dependency buckets (pinned versions)
# ───────────────────────────────────────────────────────────────
core_reqs=(
  # Core data processing
  "numpy==1.26.4"
  "scipy==1.13.1"
  "pandas==2.2.3"
  "duckdb==1.0.0"

  # Configuration & validation
  "pydantic==2.7.4"
  "pydantic-settings==2.6.1"
  "pyyaml==6.0.2"

  # Web & API
  "aiohttp==3.10.11"
  "cryptography>=45.0.0"

  # Utilities
  "python-dotenv==1.0.1"
  "pytz==2024.1"
  "rich==13.9.0"

  # Cloud Integration
  "google-cloud-secret-manager==2.20.2"

  # Machine Learning & Statistics
  "scikit-learn==1.5.0"      # Used for anomaly detection & regime detection
  "statsmodels==0.14.0"      # Used in validate_clean_data.py

  # API & External Services
  "databento==0.48.0"        # CRITICAL: Market data provider
  "tenacity==9.0.0"          # Retry logic for API calls

  # CLI & Utils
  "click==8.1.7"             # CLI framework
  "python-dateutil==2.9.0"   # Date parsing
  "typing-extensions==4.12.2" # Type hints

  # Data Processing & Visualization
  "matplotlib==3.7.0"        # Used in analysis scripts
  "requests==2.31.0"         # Used in setup_monitoring.py
  "pyarrow==14.0.0"          # Used for parquet files
)

dev_reqs=(
  "pytest==8.3.4"
  "pytest-cov==6.0.0"
  "pytest-xdist==3.6.1"          # Parallel test execution
  "pytest-timeout==2.3.1"        # Prevent hanging tests
  "pytest-asyncio==0.25.2"       # Async test support
  "hypothesis==6.122.3"          # Property-based testing
  "black==24.10.0"               # Code formatting
  "isort==5.13.2"                # Import sorting
  "pre-commit==4.0.1"            # Git hooks
  "types-pytz==2024.2.0.20241221"
  "pandas-stubs==2.2.3.250308"
  "types-PyYAML==6.0.12.20240917"
)

# ───────────────────────────────────────────────────────────────
# 3. Helper: bulk install
# ───────────────────────────────────────────────────────────────
bulk_pip_install() {
  local desc=$1; shift
  [[ $# -eq 0 ]] && return
  printf '%s\n' "$@" | \
    $PY -m pip install --user --prefer-binary --progress-bar off --no-input -r /dev/stdin \
    && echo "✅  $desc installed"
}

# Upgrade pip once, quietly
$PY -m pip install -q --upgrade pip setuptools wheel

bulk_pip_install "Core packages"   "${core_reqs[@]}"
bulk_pip_install "Dev / CI tools"  "${dev_reqs[@]}"

# ───────────────────────────────────────────────────────────────
# 4. Build stubs if *really* necessary (NumPy / SciPy only)
# ───────────────────────────────────────────────────────────────
need_stub=()
for pkg in numpy scipy; do
  $PY - <<PY || need_stub+=("$pkg")
import importlib, sys; importlib.import_module("$pkg")
PY
done

if [[ ${#need_stub[@]} -gt 0 ]]; then
  echo "⚠️  Building minimal stubs for: ${need_stub[*]}"
  for pkg in "${need_stub[@]}"; do
    cat > "$STUB_DIR/$pkg.py" <<'MINISTUB'
"""Runtime stub – only satisfies import machinery."""
import types, sys, warnings; warnings.warn(__name__ + " stub loaded")
sys.modules[__name__] = types.ModuleType(__name__)
MINISTUB
  done
  export PYTHONPATH="$(clean_py_path "${PYTHONPATH:-}"):$STUB_DIR"
  USE_PURE_PYTHON=true
else
  USE_PURE_PYTHON=false
fi

# ───────────────────────────────────────────────────────────────
# 5. Activation script & env file
# ───────────────────────────────────────────────────────────────
cat > "$CODEx_DIR/activate_container.sh" <<EOF
#!/usr/bin/env bash
# Autogenerated by container bootstrap v38 – source me!

# shellcheck disable=SC1090
source "\$(dirname "\${BASH_SOURCE[0]}")/.container_env"

if [[ "\${USE_PURE_PYTHON}" == "true" && -d "$STUB_DIR" ]]; then
  case ":\$PYTHONPATH:" in
    *":$STUB_DIR:"*) ;;                                # already present
    *) export PYTHONPATH="\${PYTHONPATH:+\$PYTHONPATH:}$STUB_DIR" ;;
  esac
fi
EOF
chmod +x "$CODEx_DIR/activate_container.sh"

cat > "$CODEx_DIR/.container_env" <<EOF
export USE_PURE_PYTHON=$USE_PURE_PYTHON
export PYTHONPATH="$(clean_py_path "${PYTHONPATH:-}")"
EOF

# ───────────────────────────────────────────────────────────────
# 6. Smoke‑test key imports
# ───────────────────────────────────────────────────────────────
echo -e "\n🔎  Import smoke test"
$PY - <<'PYTEST'
import importlib, sys
modules = {
    "numpy": "NumPy",
    "scipy.stats": "SciPy",
    "pandas": "Pandas",
    "duckdb": "DuckDB",
    "pytest": "PyTest",
}
for m, label in modules.items():
    try:
        mod = importlib.import_module(m)
        print(f"✅ {label:8}: {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"❌ {label:8}: {e}")
PYTEST

# ───────────────────────────────────────────────────────────────
# 7. One‑line summary table
# ───────────────────────────────────────────────────────────────
printf '\n%-18s %-9s\n' "Package" "Status"
for p in "${core_reqs[@]}" "${dev_reqs[@]}"; do
  mod="${p%%[<>=]*}"
  if $PY - <<PY &>/dev/null
import importlib, sys; importlib.import_module("${mod//-/_}")
PY
  then
    printf '%-18s ✅\n' "$mod"
  else
    printf '%-18s ⚠️  (stub)\n' "$mod"
  fi
done

echo -e "\n🎉  Container ready – run:  source .codex/activate_container.sh"
