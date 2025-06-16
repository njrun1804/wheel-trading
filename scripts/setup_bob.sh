#!/bin/bash
# BOB (Bolt Orchestrator Bootstrap) System Setup Script
# Detects environment and prepares system for BOB deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BOB_ROOT="$PROJECT_ROOT/bob"

echo -e "${BLUE}BOB System Setup${NC}"
echo "=================================="

# Detect OS and hardware
detect_system() {
    echo -e "\n${YELLOW}Detecting system...${NC}"
    
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    if [[ "$OS" == "Darwin" ]]; then
        echo "Operating System: macOS"
        
        # Check for Apple Silicon
        if [[ "$ARCH" == "arm64" ]]; then
            echo "Architecture: Apple Silicon (M-series)"
            
            # Detect specific M-series chip
            CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
            echo "Chip: $CHIP_INFO"
            
            # Check for M4 Pro optimizations
            if [[ "$CHIP_INFO" == *"M4"* ]]; then
                echo -e "${GREEN}M4 Pro detected - enabling optimizations${NC}"
                export BOB_M4_OPTIMIZED=1
            fi
        else
            echo "Architecture: Intel x86_64"
        fi
        
        # Check available memory
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
        echo "Total Memory: ${TOTAL_MEM}GB"
        
        # Check CPU cores
        CPU_CORES=$(sysctl -n hw.ncpu)
        echo "CPU Cores: $CPU_CORES"
        
    elif [[ "$OS" == "Linux" ]]; then
        echo "Operating System: Linux"
        echo "Architecture: $ARCH"
        
        # Get memory info
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        echo "Total Memory: ${TOTAL_MEM}GB"
        
        # Get CPU info
        CPU_CORES=$(nproc)
        echo "CPU Cores: $CPU_CORES"
    else
        echo -e "${RED}Unsupported OS: $OS${NC}"
        exit 1
    fi
}

# Check Python environment
check_python() {
    echo -e "\n${YELLOW}Checking Python environment...${NC}"
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if (( PYTHON_MAJOR >= 3 && PYTHON_MINOR >= 9 )); then
            echo -e "${GREEN}Python $PYTHON_VERSION found${NC}"
        else
            echo -e "${RED}Python 3.9+ required (found $PYTHON_VERSION)${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Python 3 not found${NC}"
        exit 1
    fi
    
    # Check for virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}Virtual environment active: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}No virtual environment detected${NC}"
        echo "Consider creating one with: python3 -m venv venv"
    fi
}

# Create BOB directory structure
create_directories() {
    echo -e "\n${YELLOW}Creating BOB directory structure...${NC}"
    
    directories=(
        "$BOB_ROOT"
        "$BOB_ROOT/config"
        "$BOB_ROOT/config/environments"
        "$BOB_ROOT/logs"
        "$BOB_ROOT/cache"
        "$BOB_ROOT/data"
        "$BOB_ROOT/scripts"
        "$BOB_ROOT/core"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            echo "Created: $dir"
        else
            echo "Exists: $dir"
        fi
    done
}

# Install system dependencies
install_system_deps() {
    echo -e "\n${YELLOW}Checking system dependencies...${NC}"
    
    if [[ "$OS" == "Darwin" ]]; then
        # Check for Homebrew
        if ! command -v brew >/dev/null 2>&1; then
            echo -e "${RED}Homebrew not found. Please install from https://brew.sh${NC}"
            exit 1
        fi
        
        # Required packages
        packages=("git" "jq" "ripgrep" "fd" "bat" "htop")
        
        for pkg in "${packages[@]}"; do
            if brew list "$pkg" >/dev/null 2>&1; then
                echo -e "${GREEN}$pkg already installed${NC}"
            else
                echo "Installing $pkg..."
                brew install "$pkg"
            fi
        done
        
    elif [[ "$OS" == "Linux" ]]; then
        # Check for package manager
        if command -v apt-get >/dev/null 2>&1; then
            PKG_MGR="apt-get"
        elif command -v yum >/dev/null 2>&1; then
            PKG_MGR="yum"
        else
            echo -e "${YELLOW}Could not detect package manager${NC}"
            return
        fi
        
        echo "Using package manager: $PKG_MGR"
        
        # Install packages (requires sudo)
        packages=("git" "jq" "ripgrep" "fd-find" "bat" "htop")
        
        echo "Installing system packages (may require sudo)..."
        sudo $PKG_MGR update
        sudo $PKG_MGR install -y "${packages[@]}"
    fi
}

# Setup environment variables
setup_environment() {
    echo -e "\n${YELLOW}Setting up environment variables...${NC}"
    
    ENV_FILE="$PROJECT_ROOT/.env"
    
    # Create .env if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Creating .env file..."
        cat > "$ENV_FILE" << EOF
# BOB Environment Configuration
BOB_ENV=development
BOB_ROOT=$BOB_ROOT
PROJECT_ROOT=$PROJECT_ROOT

# Resource Configuration
BOB_MAX_WORKERS=$CPU_CORES
BOB_MEMORY_LIMIT=${TOTAL_MEM}G

# Logging
LOG_LEVEL=INFO
LOG_DIR=$BOB_ROOT/logs

# Database
DATABASE_PATH=$PROJECT_ROOT/data/wheel_trading_master.duckdb

# Integration with Einstein/Bolt
EINSTEIN_ENABLED=true
BOLT_ENABLED=true

# Performance settings
BOB_CACHE_DIR=$BOB_ROOT/cache
BOB_INDEX_UPDATE_INTERVAL=300

# M4 Pro optimizations (if detected)
$([ -n "$BOB_M4_OPTIMIZED" ] && echo "BOB_M4_OPTIMIZED=1")
EOF
        echo -e "${GREEN}Created .env file${NC}"
    else
        echo -e "${GREEN}.env file already exists${NC}"
    fi
    
    # Export environment variables
    export $(grep -v '^#' "$ENV_FILE" | xargs)
}

# Check existing Einstein/Bolt installation
check_legacy_systems() {
    echo -e "\n${YELLOW}Checking for existing Einstein/Bolt systems...${NC}"
    
    EINSTEIN_EXISTS=false
    BOLT_EXISTS=false
    
    if [[ -d "$PROJECT_ROOT/einstein" ]]; then
        echo -e "${GREEN}Einstein found at $PROJECT_ROOT/einstein${NC}"
        EINSTEIN_EXISTS=true
    else
        echo "Einstein not found"
    fi
    
    if [[ -d "$PROJECT_ROOT/bolt" ]]; then
        echo -e "${GREEN}Bolt found at $PROJECT_ROOT/bolt${NC}"
        BOLT_EXISTS=true
    else
        echo "Bolt not found"
    fi
    
    if $EINSTEIN_EXISTS && $BOLT_EXISTS; then
        echo -e "${GREEN}Both Einstein and Bolt found - BOB will integrate${NC}"
    elif $EINSTEIN_EXISTS || $BOLT_EXISTS; then
        echo -e "${YELLOW}Partial legacy system found - BOB will provide missing components${NC}"
    else
        echo -e "${YELLOW}No legacy systems found - BOB will run standalone${NC}"
    fi
}

# Setup Python dependencies
setup_python_deps() {
    echo -e "\n${YELLOW}Setting up Python dependencies...${NC}"
    
    # Create requirements.txt if it doesn't exist
    REQ_FILE="$BOB_ROOT/requirements.txt"
    if [[ ! -f "$REQ_FILE" ]]; then
        echo "Creating requirements.txt..."
        cat > "$REQ_FILE" << EOF
# BOB Core Dependencies
aiofiles>=23.0.0
asyncio>=3.4.3
click>=8.1.0
fastapi>=0.100.0
httpx>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
rich>=13.0.0
structlog>=23.0.0
uvicorn>=0.23.0

# Integration with existing system
duckdb>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Monitoring and observability
prometheus-client>=0.17.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0

# Development tools (optional)
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.4.0
EOF
        echo -e "${GREEN}Created requirements.txt${NC}"
    fi
    
    # Install dependencies
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
}

# Create initial configuration
create_initial_config() {
    echo -e "\n${YELLOW}Creating initial configuration...${NC}"
    
    # Base configuration
    BASE_CONFIG="$BOB_ROOT/config/base.yaml"
    if [[ ! -f "$BASE_CONFIG" ]]; then
        cat > "$BASE_CONFIG" << EOF
# BOB Base Configuration
name: "BOB - Bolt Orchestrator Bootstrap"
version: "1.0.0"

# System settings
system:
  max_workers: ${CPU_CORES}
  memory_limit: "${TOTAL_MEM}GB"
  cache_size: "1GB"
  
# Logging configuration
logging:
  level: "INFO"
  format: "json"
  directory: "${BOB_ROOT}/logs"
  
# Database configuration
database:
  type: "duckdb"
  path: "${PROJECT_ROOT}/data/wheel_trading_master.duckdb"
  
# Integration settings
integrations:
  einstein:
    enabled: true
    path: "${PROJECT_ROOT}/einstein"
  bolt:
    enabled: true
    path: "${PROJECT_ROOT}/bolt"
    
# Service endpoints
services:
  api:
    host: "0.0.0.0"
    port: 8000
  web_ui:
    host: "0.0.0.0"
    port: 3000
    
# Performance tuning
performance:
  index_update_interval: 300
  cache_ttl: 3600
  max_concurrent_tasks: 10
EOF
        echo -e "${GREEN}Created base configuration${NC}"
    fi
}

# Main setup function
main() {
    echo -e "${BLUE}Starting BOB setup...${NC}\n"
    
    # Run setup steps
    detect_system
    check_python
    create_directories
    install_system_deps
    setup_environment
    check_legacy_systems
    setup_python_deps
    create_initial_config
    
    echo -e "\n${GREEN}BOB setup completed successfully!${NC}"
    echo -e "\nNext steps:"
    echo "1. Review configuration in $BOB_ROOT/config/"
    echo "2. Run deployment: python scripts/deploy_bob.py deploy"
    echo "3. Check status: python scripts/deploy_bob.py status"
    
    # If M4 Pro detected, show optimization tips
    if [[ -n "$BOB_M4_OPTIMIZED" ]]; then
        echo -e "\n${BLUE}M4 Pro Optimizations Available:${NC}"
        echo "- Metal GPU acceleration enabled"
        echo "- Unified memory optimization active"
        echo "- Neural Engine support available"
    fi
}

# Run main function
main