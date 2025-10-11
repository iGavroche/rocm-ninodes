#!/bin/bash
# Test runner for ROCM Ninodes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}ROCM Ninodes Test Suite${NC}"
echo "=========================="

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "test_fixtures.py" ]; then
    echo -e "${RED}Error: Must run from tests directory${NC}"
    echo "Usage: cd tests && ./run_tests.sh"
    exit 1
fi

echo -e "${YELLOW}Running standalone tests (no ComfyUI required)...${NC}"
echo ""

# Run performance tests
echo -e "${BLUE}1. Performance Tests${NC}"
echo "-------------------"
if pytest test_performance.py -v; then
    echo -e "${GREEN}✓ Performance tests passed${NC}"
else
    echo -e "${RED}✗ Performance tests failed${NC}"
    exit 1
fi

echo ""

# Run correctness tests
echo -e "${BLUE}2. Correctness Tests${NC}"
echo "-------------------"
if pytest test_correctness.py -v; then
    echo -e "${GREEN}✓ Correctness tests passed${NC}"
else
    echo -e "${RED}✗ Correctness tests failed${NC}"
    exit 1
fi

echo ""

# Check for captured data
echo -e "${BLUE}3. Data Availability Check${NC}"
echo "---------------------------"
if [ -d "../test_data/captured" ]; then
    echo -e "${GREEN}✓ Test data directory exists${NC}"
    
    # Count captured files
    flux_files=$(find ../test_data/captured/flux_1024x1024 -name "*.pkl" 2>/dev/null | wc -l)
    wan_files=$(find ../test_data/captured/wan_320x320_17frames -name "*.pkl" 2>/dev/null | wc -l)
    timing_files=$(find ../test_data/captured/timing -name "*.pkl" 2>/dev/null | wc -l)
    memory_files=$(find ../test_data/captured/memory -name "*.pkl" 2>/dev/null | wc -l)
    
    echo "  Flux data files: $flux_files"
    echo "  WAN data files: $wan_files"
    echo "  Timing files: $timing_files"
    echo "  Memory files: $memory_files"
    
    if [ $flux_files -eq 0 ] && [ $wan_files -eq 0 ]; then
        echo -e "${YELLOW}⚠ No captured data found. Tests will use mock data.${NC}"
        echo "  To capture real data, run ComfyUI with ROCM_NINODES_DEBUG=1"
    fi
else
    echo -e "${YELLOW}⚠ Test data directory not found. Tests will use mock data.${NC}"
    echo "  To capture real data, run ComfyUI with ROCM_NINODES_DEBUG=1"
fi

echo ""

# Optional ComfyUI integration tests
if [ -n "$WITH_COMFYUI" ]; then
    echo -e "${BLUE}4. ComfyUI Integration Tests${NC}"
    echo "----------------------------"
    
    if [ -z "$COMFYUI_PATH" ]; then
        COMFYUI_PATH="/home/nino/ComfyUI"
    fi
    
    if [ -d "$COMFYUI_PATH" ]; then
        echo "Running ComfyUI integration tests..."
        PYTHONPATH="$COMFYUI_PATH:$PYTHONPATH" python test_integration.py
    else
        echo -e "${RED}Error: ComfyUI path not found: $COMFYUI_PATH${NC}"
        echo "Set COMFYUI_PATH environment variable to correct path"
        exit 1
    fi
else
    echo -e "${YELLOW}4. ComfyUI Integration Tests (skipped)${NC}"
    echo "To run integration tests: WITH_COMFYUI=1 ./run_tests.sh"
fi

echo ""
echo -e "${GREEN}All tests completed successfully!${NC}"
echo ""
echo "To capture real data for testing:"
echo "  export ROCM_NINODES_DEBUG=1"
echo "  # Run your ComfyUI workflows"
echo "  # Data will be saved to test_data/captured/"
echo ""
echo "To run with ComfyUI integration:"
echo "  WITH_COMFYUI=1 ./run_tests.sh"
