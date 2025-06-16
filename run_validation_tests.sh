#!/bin/bash
"""
Validation Test Runner

This script provides different modes for running validation tests:
- Quick check of current state
- Comprehensive validation after fixes
- Debug mode with detailed output

Usage:
  ./run_validation_tests.sh quick    # Quick current state check
  ./run_validation_tests.sh full     # Full comprehensive validation
  ./run_validation_tests.sh debug    # Full validation with debug output
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
check_environment() {
    if [[ ! -d "bolt" ]] || [[ ! -d "einstein" ]] || [[ ! -d "src" ]]; then
        print_error "Must be run from wheel-trading project root directory"
        print_error "Expected directories: bolt/, einstein/, src/"
        exit 1
    fi
    
    if [[ ! -f "comprehensive_validation_test.py" ]]; then
        print_error "comprehensive_validation_test.py not found"
        exit 1
    fi
    
    if [[ ! -f "quick_validation_check.py" ]]; then
        print_error "quick_validation_check.py not found"
        exit 1
    fi
}

# Run quick validation check
run_quick_check() {
    print_status "Running quick validation check..."
    echo "This checks the current state before other agents make fixes"
    echo "=================================================="
    
    python3 quick_validation_check.py
    exit_code=$?
    
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        print_success "Quick validation completed successfully"
    else
        print_warning "Quick validation found issues (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Run comprehensive validation
run_comprehensive_validation() {
    local debug_mode=$1
    
    print_status "Running comprehensive validation tests..."
    echo "This validates that all fixes have been properly implemented"
    echo "==========================================================="
    
    if [[ "$debug_mode" == "debug" ]]; then
        print_status "Debug mode enabled - detailed error output"
        python3 comprehensive_validation_test.py --debug
    else
        python3 comprehensive_validation_test.py
    fi
    
    exit_code=$?
    
    echo ""
    case $exit_code in
        0)
            print_success "All tests passed! System is ready for production."
            ;;
        1)
            print_warning "Minor issues detected. System mostly functional."
            ;;
        2)
            print_warning "Several issues need attention before production use."
            ;;
        3)
            print_error "Major issues detected. System needs significant fixes."
            ;;
        *)
            print_error "Unexpected exit code: $exit_code"
            ;;
    esac
    
    return $exit_code
}

# Show usage information
show_usage() {
    echo "Validation Test Runner"
    echo "====================="
    echo ""
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  quick    - Quick check of current state (before fixes)"
    echo "  full     - Comprehensive validation (after fixes)"
    echo "  debug    - Comprehensive validation with debug output"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick    # Check what needs to be fixed"
    echo "  $0 full     # Validate all fixes are working"
    echo "  $0 debug    # Get detailed error information"
    echo ""
    echo "Test Files:"
    echo "  quick_validation_check.py      - Quick current state assessment"
    echo "  comprehensive_validation_test.py - Full validation test suite"
}

# Main execution
main() {
    local mode=${1:-help}
    
    print_status "Validation Test Runner - Mode: $mode"
    
    # Check environment first
    check_environment
    
    case $mode in
        "quick")
            run_quick_check
            ;;
        "full")
            run_comprehensive_validation
            ;;
        "debug")
            run_comprehensive_validation "debug"
            ;;
        "help"|"--help"|"-h")
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown mode: $mode"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"