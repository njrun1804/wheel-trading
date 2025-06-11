#!/bin/bash
# Unity Wheel CI/CD Manager - Uses GitHub CLI for workflow optimization
# Requires: gh (GitHub CLI) to be installed and authenticated

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="wheel-trading"
WORKFLOW_DIR=".github/workflows"
CACHE_DAYS=7
LOG_DIR="logs/ci-cd"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if gh is installed and authenticated
check_gh_cli() {
    if ! command -v gh &> /dev/null; then
        print_status "$RED" "❌ GitHub CLI (gh) is not installed"
        echo "Install it from: https://cli.github.com/"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        print_status "$RED" "❌ GitHub CLI is not authenticated"
        echo "Run: gh auth login"
        exit 1
    fi

    print_status "$GREEN" "✅ GitHub CLI is ready"
}

# Function to list all workflows
list_workflows() {
    print_status "$BLUE" "\n📋 Available Workflows:"
    gh workflow list --all
}

# Function to show workflow runs status
show_workflow_status() {
    print_status "$BLUE" "\n🔄 Recent Workflow Runs:"
    gh run list --limit 10
}

# Function to analyze workflow performance
analyze_workflow_performance() {
    local workflow_name=${1:-""}
    local days=${2:-7}

    print_status "$BLUE" "\n📊 Workflow Performance Analysis (last $days days):"

    if [ -z "$workflow_name" ]; then
        # Analyze all workflows
        workflows=$(gh workflow list --json name,id -q '.[] | .name')
        while IFS= read -r workflow; do
            analyze_single_workflow "$workflow" "$days"
        done <<< "$workflows"
    else
        analyze_single_workflow "$workflow_name" "$days"
    fi
}

# Function to analyze a single workflow
analyze_single_workflow() {
    local workflow_name=$1
    local days=$2
    local since_date=$(date -u -d "$days days ago" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v-${days}d '+%Y-%m-%dT%H:%M:%SZ')

    echo ""
    print_status "$YELLOW" "Workflow: $workflow_name"

    # Get workflow runs data
    local runs_data=$(gh run list \
        --workflow "$workflow_name" \
        --created ">=$since_date" \
        --json conclusion,createdAt,updatedAt,event \
        --limit 100)

    # Calculate statistics using jq
    if [ -n "$runs_data" ] && [ "$runs_data" != "[]" ]; then
        echo "$runs_data" | jq -r '
            . as $runs |
            ($runs | length) as $total |
            ($runs | map(select(.conclusion == "success")) | length) as $success |
            ($runs | map(select(.conclusion == "failure")) | length) as $failed |
            ($runs | map(
                ((.updatedAt | fromdate) - (.createdAt | fromdate)) / 60
            ) | add / length) as $avg_duration |
            "  Total runs: \($total)",
            "  Success rate: \(($success / $total * 100) | round)%",
            "  Average duration: \($avg_duration | round) minutes",
            "  Failed runs: \($failed)"
        '
    else
        echo "  No runs in the specified period"
    fi
}

# Function to enable/disable workflows
manage_workflow() {
    local action=$1
    local workflow=$2

    case $action in
        enable)
            gh workflow enable "$workflow"
            print_status "$GREEN" "✅ Enabled workflow: $workflow"
            ;;
        disable)
            gh workflow disable "$workflow"
            print_status "$YELLOW" "⚠️  Disabled workflow: $workflow"
            ;;
        *)
            print_status "$RED" "❌ Unknown action: $action"
            ;;
    esac
}

# Function to trigger a workflow manually
trigger_workflow() {
    local workflow=$1
    local branch=${2:-main}
    local inputs=${3:-"{}"}

    print_status "$BLUE" "🚀 Triggering workflow: $workflow on branch: $branch"

    if [ "$inputs" != "{}" ]; then
        gh workflow run "$workflow" --ref "$branch" --raw-field inputs="$inputs"
    else
        gh workflow run "$workflow" --ref "$branch"
    fi

    print_status "$GREEN" "✅ Workflow triggered successfully"

    # Wait a moment and show the run
    sleep 2
    gh run list --workflow "$workflow" --limit 1
}

# Function to cancel running workflows
cancel_workflows() {
    local workflow=${1:-""}

    print_status "$YELLOW" "⚠️  Cancelling workflows..."

    if [ -z "$workflow" ]; then
        # Cancel all running workflows
        gh run list --status in_progress --json databaseId -q '.[].databaseId' | \
        while read -r run_id; do
            gh run cancel "$run_id"
            print_status "$GREEN" "✅ Cancelled run: $run_id"
        done
    else
        # Cancel specific workflow runs
        gh run list --workflow "$workflow" --status in_progress --json databaseId -q '.[].databaseId' | \
        while read -r run_id; do
            gh run cancel "$run_id"
            print_status "$GREEN" "✅ Cancelled run: $run_id"
        done
    fi
}

# Function to download workflow artifacts
download_artifacts() {
    local run_id=$1
    local output_dir=${2:-"artifacts"}

    mkdir -p "$output_dir"

    print_status "$BLUE" "📥 Downloading artifacts from run: $run_id"
    gh run download "$run_id" --dir "$output_dir"
    print_status "$GREEN" "✅ Artifacts downloaded to: $output_dir"
}

# Function to view workflow logs
view_logs() {
    local run_id=$1
    local job_name=${2:-""}

    if [ -z "$job_name" ]; then
        gh run view "$run_id" --log
    else
        gh run view "$run_id" --log | grep -A 20 -B 5 "$job_name"
    fi
}

# Function to optimize workflow caching
optimize_cache() {
    print_status "$BLUE" "\n🗄️  Cache Optimization Analysis:"

    # List cache usage
    gh api \
        -H "Accept: application/vnd.github+json" \
        "/repos/{owner}/{repo}/actions/cache/usage" | \
    jq -r '
        "Repository cache usage:",
        "  Active caches: \(.active_caches_count)",
        "  Size: \(.active_caches_size_in_bytes / 1024 / 1024 | round) MB",
        "  Percentage used: \(.full_repo_percentage_of_size | round)%"
    '

    # List caches
    print_status "$BLUE" "\n📦 Active Caches:"
    gh api \
        -H "Accept: application/vnd.github+json" \
        "/repos/{owner}/{repo}/actions/caches" | \
    jq -r '.actions_caches[] |
        "  \(.key) - \(.size_in_bytes / 1024 / 1024 | round) MB - Last used: \(.last_accessed_at)"'
}

# Function to clean old workflow runs
clean_old_runs() {
    local days=${1:-30}
    local cutoff_date=$(date -u -d "$days days ago" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -v-${days}d '+%Y-%m-%dT%H:%M:%SZ')

    print_status "$YELLOW" "🧹 Cleaning workflow runs older than $days days..."

    # Get old runs
    local old_runs=$(gh run list \
        --status completed \
        --created "<$cutoff_date" \
        --json databaseId \
        --limit 100 \
        -q '.[].databaseId')

    if [ -z "$old_runs" ]; then
        print_status "$GREEN" "✅ No old runs to clean"
        return
    fi

    local count=0
    while IFS= read -r run_id; do
        gh run delete "$run_id" --yes
        ((count++))
    done <<< "$old_runs"

    print_status "$GREEN" "✅ Deleted $count old workflow runs"
}

# Function to create workflow dispatch event
create_dispatch() {
    local event_type=$1
    local client_payload=${2:-"{}"}

    gh api \
        --method POST \
        -H "Accept: application/vnd.github+json" \
        "/repos/{owner}/{repo}/dispatches" \
        -f "event_type=$event_type" \
        -f "client_payload=$client_payload"

    print_status "$GREEN" "✅ Dispatch event created: $event_type"
}

# Function to show workflow configuration
show_workflow_config() {
    local workflow=$1

    if [ -f "$WORKFLOW_DIR/$workflow" ]; then
        print_status "$BLUE" "\n📄 Workflow Configuration: $workflow"
        cat "$WORKFLOW_DIR/$workflow" | head -30
        echo "..."
    else
        print_status "$RED" "❌ Workflow file not found: $workflow"
    fi
}

# Function to validate all workflows
validate_workflows() {
    print_status "$BLUE" "\n🔍 Validating all workflows..."

    for workflow in "$WORKFLOW_DIR"/*.yml "$WORKFLOW_DIR"/*.yaml; do
        if [ -f "$workflow" ]; then
            local filename=$(basename "$workflow")
            if gh workflow view "$filename" &> /dev/null; then
                print_status "$GREEN" "✅ Valid: $filename"
            else
                print_status "$RED" "❌ Invalid: $filename"
            fi
        fi
    done
}

# Function to generate performance report
generate_report() {
    local output_file="$LOG_DIR/ci-cd-report-$(date +%Y%m%d-%H%M%S).md"

    print_status "$BLUE" "\n📊 Generating CI/CD Performance Report..."

    {
        echo "# Unity Wheel CI/CD Performance Report"
        echo "Generated: $(date)"
        echo ""
        echo "## Workflow Status"
        gh workflow list --all
        echo ""
        echo "## Recent Runs (Last 24 hours)"
        gh run list --limit 20
        echo ""
        echo "## Performance Metrics"
        analyze_workflow_performance "" 7
        echo ""
        echo "## Cache Usage"
        optimize_cache
    } > "$output_file"

    print_status "$GREEN" "✅ Report saved to: $output_file"
}

# Main menu
show_menu() {
    echo ""
    print_status "$BLUE" "🔧 Unity Wheel CI/CD Manager"
    echo "============================"
    echo "1. List workflows"
    echo "2. Show workflow status"
    echo "3. Analyze performance"
    echo "4. Trigger workflow"
    echo "5. Cancel workflows"
    echo "6. Download artifacts"
    echo "7. View logs"
    echo "8. Optimize cache"
    echo "9. Clean old runs"
    echo "10. Validate workflows"
    echo "11. Generate report"
    echo "12. Exit"
    echo ""
}

# Main function
main() {
    check_gh_cli

    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice

            case $choice in
                1) list_workflows ;;
                2) show_workflow_status ;;
                3)
                    read -p "Workflow name (leave empty for all): " workflow
                    read -p "Days to analyze (default 7): " days
                    days=${days:-7}
                    analyze_workflow_performance "$workflow" "$days"
                    ;;
                4)
                    read -p "Workflow file name: " workflow
                    read -p "Branch (default main): " branch
                    branch=${branch:-main}
                    trigger_workflow "$workflow" "$branch"
                    ;;
                5)
                    read -p "Workflow name (leave empty for all): " workflow
                    cancel_workflows "$workflow"
                    ;;
                6)
                    read -p "Run ID: " run_id
                    download_artifacts "$run_id"
                    ;;
                7)
                    read -p "Run ID: " run_id
                    view_logs "$run_id"
                    ;;
                8) optimize_cache ;;
                9)
                    read -p "Delete runs older than (days, default 30): " days
                    days=${days:-30}
                    clean_old_runs "$days"
                    ;;
                10) validate_workflows ;;
                11) generate_report ;;
                12)
                    print_status "$GREEN" "👋 Goodbye!"
                    exit 0
                    ;;
                *)
                    print_status "$RED" "❌ Invalid option"
                    ;;
            esac

            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            list) list_workflows ;;
            status) show_workflow_status ;;
            analyze) analyze_workflow_performance "${2:-}" "${3:-7}" ;;
            trigger) trigger_workflow "$2" "${3:-main}" "${4:-{}}" ;;
            cancel) cancel_workflows "${2:-}" ;;
            artifacts) download_artifacts "$2" "${3:-artifacts}" ;;
            logs) view_logs "$2" "${3:-}" ;;
            cache) optimize_cache ;;
            clean) clean_old_runs "${2:-30}" ;;
            validate) validate_workflows ;;
            report) generate_report ;;
            *)
                echo "Usage: $0 [command] [args...]"
                echo "Commands:"
                echo "  list                    - List all workflows"
                echo "  status                  - Show workflow run status"
                echo "  analyze [workflow] [days] - Analyze workflow performance"
                echo "  trigger <workflow> [branch] [inputs] - Trigger a workflow"
                echo "  cancel [workflow]       - Cancel running workflows"
                echo "  artifacts <run_id> [dir] - Download artifacts"
                echo "  logs <run_id> [job]     - View workflow logs"
                echo "  cache                   - Optimize cache usage"
                echo "  clean [days]           - Clean old workflow runs"
                echo "  validate               - Validate all workflows"
                echo "  report                 - Generate performance report"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"
