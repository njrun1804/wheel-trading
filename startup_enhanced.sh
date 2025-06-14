#!/bin/bash
# Unity Wheel Enhanced Startup Script
# Comprehensive initialization with all necessary checks and setup

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${GREEN}${BOLD}🚀 UNITY WHEEL TRADING SYSTEM${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}Starting comprehensive system initialization...${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display step status
step() {
    echo -e "\n${YELLOW}▶ $1${NC}"
}

# Function to display success
success() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

# Function to display error
error() {
    echo -e "${RED}  ❌ $1${NC}"
}

# Function to display warning
warning() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

# Check Python version
step "Checking Python installation"
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    success "Python $PYTHON_VERSION found"
else
    error "Python 3 not found. Please install Python 3.11 or higher."
    exit 1
fi

# Check if in correct directory
step "Verifying project directory"
if [ -f "pyproject.toml" ] && [ -d "src/unity_wheel" ]; then
    success "In Unity Wheel project directory"
else
    error "Not in Unity Wheel project root. Please cd to the project directory."
    exit 1
fi

# Create necessary directories
step "Creating required directories"
mkdir -p logs data/cache ~/.wheel_trading/secrets ~/.wheel_trading/cache
success "Directories created"

# Check environment file
step "Checking environment configuration"
if [ -f ".env" ]; then
    success ".env file found"
    # Source it for this session
    export $(grep -v '^#' .env | xargs)
else
    if [ -f ".env.example" ]; then
        warning ".env file not found. Creating from .env.example"
        cp .env.example .env
        echo ""
        echo -e "${YELLOW}Please edit .env file and add your API keys:${NC}"
        echo "  - DATABENTO_API_KEY"
        echo "  - FRED_API_KEY or OFRED_API_KEY"
        echo ""
    else
        warning "No .env file found. API connections may fail."
    fi
fi

# Check virtual environment
step "Checking Python environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    success "Virtual environment active: $VIRTUAL_ENV"
else
    warning "No virtual environment active"
    if [ -d "venv" ]; then
        echo "  Activating existing venv..."
        source venv/bin/activate
        success "Virtual environment activated"
    elif [ -d ".venv" ]; then
        echo "  Activating existing .venv..."
        source .venv/bin/activate
        success "Virtual environment activated"
    else
        echo "  Creating new virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        success "Virtual environment created and activated"
    fi
fi

# Install dependencies if needed
step "Checking Python dependencies"
if python3 -c "import unity_wheel" 2>/dev/null; then
    success "Unity Wheel package already installed"
else
    echo "  Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        success "Dependencies installed from requirements.txt"
    else
        pip install -e .
        success "Package installed in development mode"
    fi
fi

# Check database
step "Checking database"
DB_PATH="data/wheel_trading_optimized.duckdb"
if [ -f "$DB_PATH" ]; then
    # Get file size in MB
    DB_SIZE=$(du -m "$DB_PATH" | cut -f1)
    success "Database found: ${DB_PATH} (${DB_SIZE}MB)"
else
    warning "Primary database not found at $DB_PATH"
    # Check for alternatives
    if [ -f "data/wheel_trading_master.duckdb" ]; then
        success "Alternative database found: data/wheel_trading_master.duckdb"
    else
        error "No database found. Data collection may be needed."
    fi
fi

# Run API validation
step "Validating API connections"
if [ -f "validate_api_connections.py" ]; then
    echo "  Running API validation..."
    python3 validate_api_connections.py 2>&1 | tail -20
else
    warning "API validation script not found"
fi

# Check MCP servers (if Claude Code is being used)
step "Checking MCP server configuration"
if [ -f "mcp-servers.json" ]; then
    success "MCP server configuration found"
    # Check if key servers are configured
    if command_exists npx; then
        success "Node.js/npx available for MCP servers"
    else
        warning "Node.js not found. Some MCP features may be unavailable."
    fi
else
    warning "No mcp-servers.json found. MCP features disabled."
fi

# Hardware optimization check
step "Checking hardware optimization"
CPU_COUNT=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")
MEM_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().total / (1024**3):.1f}')" 2>/dev/null || echo "Unknown")
success "Hardware detected: ${CPU_COUNT} CPU cores, ${MEM_GB}GB RAM"

# Set performance environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$CPU_COUNT
export MKL_NUM_THREADS=$CPU_COUNT
export NUMEXPR_NUM_THREADS=$CPU_COUNT
export VECLIB_MAXIMUM_THREADS=$CPU_COUNT

# Run system diagnostics
step "Running system diagnostics"
echo ""
python3 -m unity_wheel.cli.run --diagnose || warning "Diagnostics reported issues"

# Final summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}✅ SYSTEM READY${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Available commands:"
echo ""
echo -e "  ${BOLD}${BLUE}python run.py${NC}                       - Get trading recommendation"
echo -e "  ${BLUE}python run.py --portfolio 100000${NC}    - Specify portfolio value"
echo -e "  ${BLUE}python run.py --diagnose${NC}            - Run system diagnostics"
echo -e "  ${BLUE}python run.py --performance${NC}         - Show performance metrics"
echo ""
echo -e "  ${BOLD}${BLUE}./orchestrate_turbo.py${NC} '<command>'  - ${BOLD}TURBO MODE${NC} (all cores)"
echo -e "  ${BLUE}./orchestrate${NC} '<command>'           - Standard orchestrator"
echo ""
echo "Data collection:"
echo -e "  ${BLUE}python -m unity_wheel.cli.databento_integration collect${NC}"
echo ""
echo "API validation:"
echo -e "  ${BLUE}python validate_api_connections.py${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# If command provided, run it
if [ $# -gt 0 ]; then
    echo ""
    echo -e "${BOLD}${BLUE}Executing: $@${NC}"
    echo ""
    "$@"
fi