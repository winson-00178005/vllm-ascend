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
# Supports single module execution, full execution, coverage, and dual-mode execution

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ST_DIR=$(dirname "$SCRIPT_DIR")

# Default settings
EXEC_MODE="auto"
COVERAGE=false
MODULE=""
VERBOSE=false

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
        --module)
            MODULE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -sv"
fi

# Add execution mode
PYTEST_CMD="$PYTEST_CMD --exec-mode=$EXEC_MODE"

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=vllm_ascend --cov-report=term-missing --cov-report=xml:st-coverage.xml"
fi

# Determine test path
if [ -n "$MODULE" ]; then
    TEST_PATH="$ST_DIR/$MODULE/"
else
    TEST_PATH="$ST_DIR/"
fi

# Execute tests
echo "Running ST tests with command:"
echo "$PYTEST_CMD $TEST_PATH"
echo ""

$PYTEST_CMD $TEST_PATH

echo ""
echo "ST tests completed successfully!"