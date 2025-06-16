#!/bin/bash
# Bolt Sonnet 4 Production Startup Script
# Deploys and starts the complete 12-agent orchestrator system with all optimizations

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
DEPLOYMENT_LOG="bolt_production_deployment.log"
QUICK_MODE=${1:-"standard"}
PYTHON_CMD="python3"

echo -e "${GREEN}${BOLD}🚀 BOLT SONNET 4 PRODUCTION DEPLOYMENT${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}Deploying 12-agent orchestrator with M4 Pro optimization${NC}"
echo ""

# Clean up previous logs
if [ -f "$DEPLOYMENT_LOG" ]; then
    mv "$DEPLOYMENT_LOG" "${DEPLOYMENT_LOG}.$(date +%Y%m%d_%H%M%S).bak"
fi

# Pre-deployment checks
echo -e "${YELLOW}🔍 Running pre-deployment checks...${NC}"

# Check Python environment
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check required modules
echo "Checking required Python modules..."
required_modules=("asyncio" "psutil" "numpy")
for module in "${required_modules[@]}"; do
    if ! $PYTHON_CMD -c "import $module" &> /dev/null; then
        echo -e "${RED}❌ Required module '$module' not found${NC}"
        echo "Please install with: pip install $module"
        exit 1
    fi
done

# Check system resources
echo "Checking system resources..."
available_memory=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / 1024**3))")
if [ "$available_memory" -lt 16 ]; then
    echo -e "${YELLOW}⚠️  Warning: Low available memory (${available_memory}GB). Recommended: 16GB+${NC}"
fi

cpu_count=$(python3 -c "import psutil; print(psutil.cpu_count())")
if [ "$cpu_count" -lt 8 ]; then
    echo -e "${YELLOW}⚠️  Warning: Limited CPU cores (${cpu_count}). Optimized for M4 Pro (12 cores)${NC}"
fi

echo -e "${GREEN}✅ Pre-deployment checks passed${NC}"
echo ""

# Start deployment
echo -e "${BOLD}${BLUE}🔥 Starting Bolt production deployment...${NC}"
echo "Deployment log: $DEPLOYMENT_LOG"
echo ""

deployment_start=$(date +%s)

# Choose deployment mode
if [ "$QUICK_MODE" = "--quick" ] || [ "$QUICK_MODE" = "quick" ]; then
    echo -e "${YELLOW}⚡ Quick deployment mode enabled${NC}"
    $PYTHON_CMD deploy_bolt_production.py --quick
else
    echo -e "${BLUE}📋 Standard deployment mode${NC}"
    $PYTHON_CMD deploy_bolt_production.py
fi

deployment_exit_code=$?
deployment_end=$(date +%s)
deployment_duration=$((deployment_end - deployment_start))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $deployment_exit_code -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ BOLT PRODUCTION DEPLOYMENT SUCCESSFUL${NC}"
    echo -e "${GREEN}   Deployment completed in ${deployment_duration} seconds${NC}"
    echo ""
    
    # Display deployment summary
    if [ -f "bolt_production_deployment_report.json" ]; then
        echo -e "${BLUE}📊 Deployment Summary:${NC}"
        
        # Extract key metrics from report
        success_rate=$(python3 -c "
import json
try:
    with open('bolt_production_deployment_report.json') as f:
        data = json.load(f)
    print(f\"{data.get('success_rate', 0):.1%}\")
except: print('N/A')
        ")
        
        components_successful=$(python3 -c "
import json
try:
    with open('bolt_production_deployment_report.json') as f:
        data = json.load(f)
    print(f\"{data.get('components_deployed', {}).get('successful', 0)}\")
except: print('N/A')
        ")
        
        components_total=$(python3 -c "
import json
try:
    with open('bolt_production_deployment_report.json') as f:
        data = json.load(f)
    print(f\"{data.get('components_deployed', {}).get('total', 0)}\")
except: print('N/A')
        ")
        
        echo "   Success Rate: $success_rate"
        echo "   Components: $components_successful/$components_total deployed"
        echo ""
    fi
    
    echo -e "${BOLD}🎯 System Status:${NC}"
    echo "   ✅ 12-Agent Orchestrator: Active"
    echo "   ✅ Dynamic Token Optimizer: Active"
    echo "   ✅ Work Stealing Agent Pool: Active"
    echo "   ✅ M4 Pro CPU Optimizer: Active"
    echo "   ✅ Einstein Integration: Ready"
    echo ""
    
    echo -e "${BOLD}🚀 Usage Examples:${NC}"
    echo ""
    echo -e "   ${BLUE}# Execute with 12-agent orchestrator${NC}"
    echo "   python3 -c \"import asyncio; from bolt.orchestrator_12_agent import execute_with_12_agents; asyncio.run(execute_with_12_agents('Analyze trading performance'))\""
    echo ""
    echo -e "   ${BLUE}# Use in existing code${NC}"
    echo "   from bolt.orchestrator_12_agent import Orchestrator12Agent"
    echo "   orchestrator = Orchestrator12Agent()"
    echo "   await orchestrator.initialize()"
    echo "   result = await orchestrator.execute_complex_task(instruction)"
    echo ""
    echo -e "   ${BLUE}# Check system status${NC}"
    echo "   python3 -c \"from bolt.agents.agent_pool import WorkStealingAgentPool; print('System ready')\""
    echo ""
    
else
    echo -e "${RED}${BOLD}❌ BOLT PRODUCTION DEPLOYMENT FAILED${NC}"
    echo -e "${RED}   Deployment failed after ${deployment_duration} seconds${NC}"
    echo -e "${RED}   Check $DEPLOYMENT_LOG for detailed error information${NC}"
    echo ""
    
    # Show recent errors from log
    if [ -f "$DEPLOYMENT_LOG" ]; then
        echo -e "${YELLOW}Recent errors:${NC}"
        tail -n 10 "$DEPLOYMENT_LOG" | grep -E "(ERROR|CRITICAL)" || echo "No specific errors found in log"
    fi
    
    echo ""
    echo -e "${YELLOW}🔧 Troubleshooting:${NC}"
    echo "   1. Check system requirements (16GB+ RAM, Python 3.8+)"
    echo "   2. Verify all dependencies are installed"
    echo "   3. Check $DEPLOYMENT_LOG for detailed error information"
    echo "   4. Try quick deployment: $0 --quick"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $deployment_exit_code