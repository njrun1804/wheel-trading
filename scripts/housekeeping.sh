#!/bin/bash
# Unity Wheel Bot v2.2 - Autonomous housekeeping enforcement
# Exit codes: 0=success, 1=non-critical issues, 2=critical failures

set -euo pipefail

# Version
readonly VERSION="2.2.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required Unity Wheel Bot structure
readonly REQUIRED_DIRS="src/unity_wheel src/unity_wheel/strategy src/unity_wheel/risk tests"
readonly REQUIRED_FILES="config.yaml run.py HOUSEKEEPING_GUIDE.md"

# Pattern definitions
readonly PAT_EXEC="execute_trade|place_order|submit_order|broker\.execute|broker\.place"
# Only flag hardcoded 'U' in actual trading logic, not docs/config/examples
readonly PAT_TICKER="(symbol|ticker|underlying)\s*=\s*['\"]U['\"](?!.*#.*example)(?!.*test)(?!.*Field)"
# Only flag hardcoded position sizes in actual trading calculations
readonly PAT_STATIC_POS="(position_size|num_contracts|contract_count)\s*=\s*[0-9]+(?!\s*#)(?!.*\*)"
# Only check specific math/risk functions that should return confidence
readonly PAT_CONFIDENCE="def (black_scholes|calculate_greeks|calculate_var|calculate_iv|calculate_risk)"

# OS compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INLINE="-i ''"
else
    SED_INLINE="-i"
fi

# Use ripgrep if available (10x faster)
if command -v rg >/dev/null 2>&1; then
    GREP_CMD="rg --no-heading --color=never -P"  # Enable PCRE2 for complex patterns
else
    GREP_CMD="grep -r --color=never --binary-files=without-match -E"  # Extended regex
fi

# Scoring weights
readonly WEIGHT_execution_code=100
readonly WEIGHT_test_files=5
readonly WEIGHT_adaptive_files=5
readonly WEIGHT_fetch_files=3
readonly WEIGHT_hardcoded_ticker=3
readonly WEIGHT_static_positions=2
readonly WEIGHT_missing_confidence=2

# Usage
usage() {
    cat <<EOF
Unity Wheel Bot Housekeeping v${VERSION}

Usage: $0 [OPTIONS]

OPTIONS:
    --fix             Auto-fix file placement violations
    --json            Output JSON summary only
    --quick           Skip expensive checks (~5 seconds)
    --check-staged    Check only staged files (pre-commit)
    --quiet           Suppress output (exit codes only)
    --dry-run         Show what --fix would do without changes
    --explain         Include detailed violation info
    --unity-check     Quick Unity-specific validation only
    --version         Show version and exit
    -h, --help        Show this help

EXIT CODES:
    0  Success - no issues found
    1  Non-critical issues (can be auto-fixed)
    2  Critical failures (manual intervention required)

EXAMPLES:
    # Morning check
    $0 --quick

    # Pre-commit validation
    $0 --check-staged

    # Unity-specific check only
    $0 --unity-check

    # Auto-fix with preview
    $0 --fix --dry-run

    # CI/CD integration
    $0 --json --check-staged
EOF
    exit 0
}

# Argument parsing
MODE="check"
QUICK=false
STAGED_ONLY=false
QUIET=false
DRY_RUN=false
EXPLAIN=false
UNITY_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            [[ "$MODE" == "json" ]] && { echo "Error: --fix and --json are mutually exclusive"; exit 1; }
            MODE="fix"
            ;;
        --json)
            [[ "$MODE" == "fix" ]] && { echo "Error: --fix and --json are mutually exclusive"; exit 1; }
            MODE="json"
            QUIET=true
            ;;
        --quick) QUICK=true ;;
        --check-staged) STAGED_ONLY=true ;;
        --quiet) QUIET=true ;;
        --dry-run) DRY_RUN=true ;;
        --explain) EXPLAIN=true ;;
        --unity-check) UNITY_CHECK=true ;;
        --version) echo "Unity Wheel Bot Housekeeping v${VERSION}"; exit 0 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Error handling
trap 'echo -e "\n❌ Error on line $LINENO (exit code: $?)" >&2' ERR

# Colors
if [[ "$MODE" != "json" && "$QUIET" != "true" && -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# State tracking
violation_execution_code=0
violation_test_files=0
violation_adaptive_files=0
violation_fetch_files=0
violation_hardcoded_ticker=0
violation_static_positions=0
violation_missing_confidence=0
violation_details=()
total_violations=0
critical_failures=0

# Pre-flight checks
preflight_check() {
    local errors=0

    # Verify we're in Unity Wheel Bot repo
    if [[ ! -f "config.yaml" || ! -f "run.py" ]]; then
        echo "Error: Not in Unity Wheel Bot root directory" >&2
        echo "Expected files: config.yaml, run.py" >&2
        ((errors++))
    fi

    # Verify git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        echo "Error: Not in a git repository" >&2
        ((errors++))
    fi

    # Verify required directories exist
    for dir in $REQUIRED_DIRS; do
        if [[ ! -d "$dir" ]]; then
            echo "Warning: Missing required directory: $dir" >&2
        fi
    done

    # Verify no active trading execution
    if [[ -f "src/unity_wheel/broker.py" || -f "src/unity_wheel/execution.py" ]]; then
        echo "Error: Found broker/execution files - this is recommendations only!" >&2
        ((errors++))
    fi

    return $errors
}

# File collection
collect_files() {
    if [[ "$STAGED_ONLY" == "true" ]]; then
        # Only staged files
        git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true
    else
        # All Python files (exclude virtual envs)
        find . -name "*.py" -type f \
            -not -path "./.venv/*" \
            -not -path "./venv/*" \
            -not -path "./*env/*" \
            -not -path "./__pycache__/*" \
            -not -path "./build/*" \
            -not -path "./dist/*" \
            -not -path "./.git/*" \
            2>/dev/null || true
    fi
}

# Unity quick check
unity_quick_check() {
    local errors=0

    # Check for execution code in key files
    for file in run.py src/unity_wheel/strategy/*.py; do
        if [[ -f "$file" ]] && grep -E "$PAT_EXEC" "$file" >/dev/null 2>&1; then
            echo "❌ CRITICAL: Execution code in $file"
            ((errors++))
        fi
    done

    # Verify adaptive system exists
    if [[ ! -f "src/unity_wheel/strategy/adaptive_base.py" && ! -d "src/unity_wheel/adaptive" ]]; then
        echo "❌ Missing adaptive system (neither adaptive.py nor adaptive/ directory found)"
        ((errors++))
    fi

    # Check config.yaml for Unity ticker
    if ! grep -q "ticker:.*U" config.yaml 2>/dev/null; then
        echo "⚠️  Unity ticker not found in config.yaml"
    fi

    # Quick confidence check on main entry
    if grep -q "def.*recommend" run.py 2>/dev/null; then
        if ! grep -A5 "def.*recommend" run.py | grep -q "return.*," 2>/dev/null; then
            echo "⚠️  Main recommendation function may lack confidence score"
        fi
    fi

    if [[ $errors -gt 0 ]]; then
        echo "Unity check failed with $errors critical errors"
        return 2
    else
        echo "✅ Unity-specific checks passed"
        return 0
    fi
}

# Critical check: execution code
check_execution_code() {
    local files
    local count=0

    while IFS= read -r file; do
        if [[ -f "$file" ]] && $GREP_CMD "$PAT_EXEC" "$file" >/dev/null 2>&1; then
            ((count++))
            if [[ "$EXPLAIN" == "true" ]]; then
                violation_details+=("execution_code:$file")
            fi
        fi
    done < <(collect_files)

    if [[ $count -gt 0 ]]; then
        violation_execution_code=$count
        critical_failures=1
        [[ "$MODE" == "check" && "$QUIET" != "true" ]] && \
            echo -e "${RED}❌ CRITICAL: Trading execution code found in $count files!${NC}"
    fi

    return $count
}

# File placement checks
check_file_placement() {
    local issues=0
    local file dest

    # Test files
    local test_count=0
    while IFS= read -r file; do
        ((test_count++))
    done < <(find . \( -name "test_*.py" -o -name "*_test.py" \) \
        -not -path "./tests/*" \
        -not -path "./.venv/*" \
        -not -path "./venv/*" \
        -not -path "./__pycache__/*" \
        2>/dev/null || true)

    if [[ $test_count -gt 0 ]]; then
        violation_test_files=$test_count
        issues=$((issues + test_count))

        if [[ "$MODE" == "fix" ]]; then
            mkdir -p tests
            find . \( -name "test_*.py" -o -name "*_test.py" \) \
                -not -path "./tests/*" \
                -not -path "./.venv/*" \
                -not -path "./venv/*" \
                -not -path "./__pycache__/*" \
                2>/dev/null | while IFS= read -r file; do
                dest="tests/$(basename "$file")"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "[DRY] git mv '$file' '$dest'"
                    echo "[DRY] Fix imports in '$dest'"
                else
                    git mv "$file" "$dest" 2>/dev/null || mv "$file" "$dest"
                    # Fix imports
                    {
                        echo "from pathlib import Path"
                        echo "import sys"
                        echo "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))"
                        echo ""
                        cat "$dest"
                    } > "$dest.tmp" && mv "$dest.tmp" "$dest"
                    [[ "$QUIET" != "true" ]] && echo "Moved: $file → $dest"
                fi
            done
        elif [[ "$MODE" == "check" && "$QUIET" != "true" ]]; then
            echo -e "${RED}Test files in wrong location: $test_count files${NC}"
        fi
    fi

    # Adaptive files
    local adaptive_count=0
    while IFS= read -r file; do
        ((adaptive_count++))
    done < <(find . -name "adaptive_*.py" \
        -not -path "./src/unity_wheel/adaptive/*" \
        -not -path "./.venv/*" \
        -not -path "./venv/*" \
        2>/dev/null || true)

    if [[ $adaptive_count -gt 0 ]]; then
        violation_adaptive_files=$adaptive_count
        issues=$((issues + adaptive_count))

        if [[ "$MODE" == "fix" ]]; then
            mkdir -p src/unity_wheel/adaptive
            find . -name "adaptive_*.py" \
                -not -path "./src/unity_wheel/adaptive/*" \
                -not -path "./.venv/*" \
                -not -path "./venv/*" \
                2>/dev/null | while IFS= read -r file; do
                dest="src/unity_wheel/adaptive/$(basename "$file")"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "[DRY] git mv '$file' '$dest'"
                else
                    git mv "$file" "$dest" 2>/dev/null || mv "$file" "$dest"
                    [[ "$QUIET" != "true" ]] && echo "Moved: $file → $dest"
                fi
            done
        fi
    fi

    # Fetch scripts
    local fetch_count=0
    while IFS= read -r file; do
        ((fetch_count++))
    done < <(find . -name "fetch_*.py" \
        -not -path "./tools/data/*" \
        -not -path "./.venv/*" \
        -not -path "./venv/*" \
        2>/dev/null || true)

    if [[ $fetch_count -gt 0 ]]; then
        violation_fetch_files=$fetch_count
        issues=$((issues + fetch_count))

        if [[ "$MODE" == "fix" ]]; then
            mkdir -p tools/data
            find . -name "fetch_*.py" \
                -not -path "./tools/data/*" \
                -not -path "./.venv/*" \
                -not -path "./venv/*" \
                2>/dev/null | while IFS= read -r file; do
                dest="tools/data/$(basename "$file")"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "[DRY] git mv '$file' '$dest'"
                else
                    git mv "$file" "$dest" 2>/dev/null || mv "$file" "$dest"
                    [[ "$QUIET" != "true" ]] && echo "Moved: $file → $dest"
                fi
            done
        fi
    fi

    total_violations=$((total_violations + issues))
    return $issues
}

# Unity-specific checks
check_unity_compliance() {
    [[ "$QUICK" == "true" ]] && return 0

    local issues=0
    local ticker_count=0
    local position_count=0
    local exec_count=0

    while IFS= read -r file; do
        if [[ "$file" =~ ^./src/ ]]; then
            # Check execution code
            if $GREP_CMD "$PAT_EXEC" "$file" >/dev/null 2>&1; then
                ((exec_count++))
            fi

            # Check ticker - only actual assignments, not docs/config
            if $GREP_CMD "$PAT_TICKER" "$file" 2>/dev/null >/dev/null; then
                ((ticker_count++))
                if [[ "$EXPLAIN" == "true" ]]; then
                    violation_details+=("hardcoded_ticker:$file")
                fi
            fi

            # Check static positions - only actual hardcoded values in logic
            if $GREP_CMD "$PAT_STATIC_POS" "$file" 2>/dev/null >/dev/null; then
                ((position_count++))
                if [[ "$EXPLAIN" == "true" ]]; then
                    violation_details+=("static_position:$file")
                fi
            fi
        fi
    done < <(collect_files)

    if [[ $exec_count -gt 0 ]]; then
        violation_execution_code=$exec_count
        critical_failures=1
    fi

    if [[ $ticker_count -gt 0 ]]; then
        violation_hardcoded_ticker=$ticker_count
        issues=$((issues + ticker_count))
        [[ "$MODE" == "check" && "$QUIET" != "true" ]] && \
            echo -e "${YELLOW}Found $ticker_count files with hardcoded 'U' ticker${NC}"
    fi

    if [[ $position_count -gt 0 ]]; then
        violation_static_positions=$position_count
        issues=$((issues + position_count))
        [[ "$MODE" == "check" && "$QUIET" != "true" ]] && \
            echo -e "${YELLOW}Found $position_count files with static position sizes${NC}"
    fi

    total_violations=$((total_violations + issues))
    return $issues
}

# Confidence score compliance
check_confidence_scores() {
    [[ "$QUICK" == "true" ]] && return 0

    local missing=0

    while IFS= read -r file; do
        if [[ "$file" =~ ^./src/unity_wheel/math/ ]] || [[ "$file" =~ ^./src/unity_wheel/risk/ ]]; then
            # Find functions that match our pattern
            if $GREP_CMD "$PAT_CONFIDENCE" "$file" 2>/dev/null >/dev/null; then
                # Check if they return CalculationResult or tuple with confidence
                while IFS= read -r func_line; do
                    # Extract function name and check its return pattern
                    local func_name=$(echo "$func_line" | sed -E 's/.*def ([a-z_]+).*/\1/')
                    # Look for the function and check if it returns a single value vs tuple/CalculationResult
                    if $GREP_CMD -A 10 "def $func_name" "$file" 2>/dev/null | grep -E "return\s+[^(,]*\s*$" | grep -v "CalculationResult" >/dev/null; then
                        ((missing++))
                        if [[ "$EXPLAIN" == "true" ]]; then
                            violation_details+=("missing_confidence:$file:$func_name")
                        fi
                    fi
                done < <($GREP_CMD "$PAT_CONFIDENCE" "$file" 2>/dev/null || true)
            fi
        fi
    done < <(collect_files)

    if [[ $missing -gt 0 ]]; then
        violation_missing_confidence=$missing
        total_violations=$((total_violations + missing))
        [[ "$MODE" == "check" && "$QUIET" != "true" ]] && \
            echo -e "${YELLOW}$missing math/risk functions missing confidence scores${NC}"
    fi

    return $missing
}

# Score calculation
calculate_score() {
    local max_score=100
    local deductions=0

    # Critical failures = immediate zero
    if [[ $critical_failures -gt 0 ]]; then
        echo 0
        return
    fi

    # Calculate weighted deductions
    deductions=$((deductions + violation_execution_code * WEIGHT_execution_code))
    deductions=$((deductions + violation_test_files * WEIGHT_test_files))
    deductions=$((deductions + violation_adaptive_files * WEIGHT_adaptive_files))
    deductions=$((deductions + violation_fetch_files * WEIGHT_fetch_files))
    deductions=$((deductions + violation_hardcoded_ticker * WEIGHT_hardcoded_ticker))
    deductions=$((deductions + violation_static_positions * WEIGHT_static_positions))
    deductions=$((deductions + violation_missing_confidence * WEIGHT_missing_confidence))

    local score=$((max_score - deductions))
    [[ $score -lt 0 ]] && score=0

    echo $score
}

# Output JSON
output_json() {
    local score=$1
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    cat <<EOF
{
  "timestamp": "$timestamp",
  "version": "$VERSION",
  "score": $score,
  "critical": $([[ $critical_failures -gt 0 ]] && echo "true" || echo "false"),
  "violations": {
    "execution_code": $violation_execution_code,
    "test_files": $violation_test_files,
    "adaptive_files": $violation_adaptive_files,
    "fetch_files": $violation_fetch_files,
    "hardcoded_ticker": $violation_hardcoded_ticker,
    "static_positions": $violation_static_positions,
    "missing_confidence": $violation_missing_confidence
  }
}
EOF
}

# Main execution
main() {
    # Handle Unity-check mode
    if [[ "$UNITY_CHECK" == "true" ]]; then
        unity_quick_check
        exit $?
    fi

    # Pre-flight checks
    if ! preflight_check; then
        echo "Pre-flight checks failed. Are you in the Unity Wheel Bot root?" >&2
        exit 2
    fi

    # Early exit for staged-only with no files
    if [[ "$STAGED_ONLY" == "true" ]]; then
        local staged_count=$(git diff --cached --name-only --diff-filter=ACM | grep -c '\.py$' || echo 0)
        if [[ $staged_count -eq 0 ]]; then
            [[ "$QUIET" != "true" ]] && echo "No Python files staged"
            exit 0
        fi
    fi

    # Run checks in order of importance
    check_file_placement || true
    check_unity_compliance || true
    check_execution_code || true
    check_confidence_scores || true

    # Calculate score
    local score=$(calculate_score)

    # Output based on mode
    case "$MODE" in
        json)
            output_json "$score"
            ;;

        check)
            if [[ "$QUIET" != "true" ]]; then
                echo ""
                echo "Unity Wheel Bot Housekeeping v${VERSION}"
                echo "Score: $score/100"
                echo ""

                if [[ $critical_failures -gt 0 ]]; then
                    echo -e "${RED}❌ CRITICAL FAILURES - Fix immediately!${NC}"
                    echo "   - Remove all trading execution code"
                    echo "   - This is a recommendations-only system"
                elif [[ $total_violations -eq 0 ]]; then
                    echo -e "${GREEN}✅ All checks passed!${NC}"
                else
                    echo -e "${YELLOW}⚠️  $total_violations issues found${NC}"
                    echo "   Run with --fix to auto-resolve file placement"
                    echo "   Run with --explain for details"
                fi
            fi
            ;;

        fix)
            if [[ "$QUIET" != "true" ]]; then
                echo ""
                if [[ $critical_failures -gt 0 ]]; then
                    echo -e "${RED}Cannot auto-fix critical failures${NC}"
                    echo "Manual intervention required:"
                    echo "  - Remove all execute_trade/place_order code"
                    echo "  - Ensure recommendations-only architecture"
                elif [[ "$DRY_RUN" == "true" ]]; then
                    echo -e "${YELLOW}Dry run complete - no changes made${NC}"
                else
                    echo -e "${GREEN}Fixed $total_violations issues${NC}"
                fi
            fi
            ;;
    esac

    # Exit with appropriate code
    if [[ $critical_failures -gt 0 ]]; then
        exit 2  # Critical failure
    elif [[ $total_violations -gt 0 ]]; then
        exit 1  # Non-critical issues
    else
        exit 0  # Success
    fi
}

# Run main
main "$@"
