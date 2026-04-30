#!/bin/bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

# ST Test Runner Script
# Supports: single module, full execution, coverage, parallel, dual-mode

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ST_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$ST_DIR")

# Default settings
EXEC_MODE="cpu_mock"
COVERAGE=false
PARALLEL=""
MODULE=""
VERBOSE=false
FAIL_UNDER=80
MARKER=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exec-mode)
            EXEC_MODE="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --module)
            MODULE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --fail-under)
            FAIL_UNDER="$2"
            shift 2
            ;;
        -m|--marker)
            MARKER="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest -c $ST_DIR/pytest.ini"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -sv"
fi

# Add execution mode
PYTEST_CMD="$PYTEST_CMD --exec-mode=$EXEC_MODE"

# Add parallel execution (pytest-xdist)
if [ -n "$PARALLEL" ]; then
    PYTEST_CMD="$PYTEST_CMD -n $PARALLEL --dist=loadscope"
fi

# Add marker filter
if [ -n "$MARKER" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKER"
fi

# Add coverage (pytest-cov)
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=vllm_ascend \
        --cov-config=$ST_DIR/.coveragerc \
        --cov-report=term-missing \
        --cov-report=xml:$ST_DIR/st-coverage.xml \
        --cov-report=html:$ST_DIR/htmlcov_st \
        --cov-fail-under=$FAIL_UNDER \
        --cov-branch"
fi

# Determine test path
if [ -n "$MODULE" ]; then
    TEST_PATH="$ST_DIR/$MODULE/"
else
    TEST_PATH="$ST_DIR/"
fi

# Add extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    PYTEST_CMD="$PYTEST_CMD $EXTRA_ARGS"
fi

# Execute tests
echo "======================================"
echo "ST Test Runner"
echo "======================================"
echo "Execution mode: $EXEC_MODE"
echo "Test path: $TEST_PATH"
if [ -n "$PARALLEL" ]; then
    echo "Parallel processes: $PARALLEL"
fi
if [ "$COVERAGE" = true ]; then
    echo "Coverage enabled (threshold: $FAIL_UNDER%)"
fi
if [ -n "$MARKER" ]; then
    echo "Marker filter: $MARKER"
fi
echo ""
echo "Command:"
echo "$PYTEST_CMD $TEST_PATH"
echo ""

cd "$PROJECT_ROOT"
$PYTEST_CMD $TEST_PATH

echo ""
echo "======================================"
echo "ST tests completed successfully!"
echo "======================================"

# Show coverage report if enabled
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "Coverage report saved to:"
    echo "  - XML: $ST_DIR/st-coverage.xml"
    echo "  - HTML: $ST_DIR/htmlcov_st/index.html"
fi