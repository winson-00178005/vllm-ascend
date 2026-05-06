#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# 统一的测试基础设施
# 参考 tests/e2e/common.sh 的风格，为UT/ST/E2E提供通用功能

set -eo errexit

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 颜色定义
cyan='\e[96m'
yellow='\e[33m'
red='\e[31m'
green='\e[32m'
blue='\e[34m'
none='\e[0m'

# 输出函数
_cyan() { echo -e "${cyan}$*${none}"; }
_yellow() { echo -e "${yellow}$*${none}"; }
_red() { echo -e "${red}$*${none}"; }
_green() { echo -e "${green}$*${none}"; }
_blue() { echo -e "${blue}$*${none}"; }

_info() { _cyan "Info: $*"; }
_warn() { _yellow "Warn: $*"; }
_err() { _red "Error: $*" && exit 1; }
_success() { _green "Success: $*"; }

# 环境检查函数
check_torch_npu() {
    if python3 -c "import torch_npu" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

check_cann_env() {
    if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
        return 0
    else
        return 1
    fi
}

get_exec_mode() {
    # 自动检测执行模式
    if check_torch_npu && check_cann_env; then
        echo "npu_real"
    else
        echo "cpu_mock"
    fi
}

# 测试执行函数
run_ut_tests() {
    local coverage="${1:-false}"
    local verbose="${2:-false}"
    
    _info "====> Running UT tests"
    
    PYTEST_CMD="pytest"
    if [ "$verbose" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -sv"
    fi
    
    if [ "$coverage" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov --cov-report=xml:unittests-coverage.xml --cov-report=term-missing"
    fi
    
    PYTEST_CMD="$PYTEST_CMD tests/ut"
    
    _info "Command: $PYTEST_CMD"
    cd "$PROJECT_ROOT"
    $PYTEST_CMD
    
    _success "====> UT tests passed"
}

run_st_tests() {
    local exec_mode="${1:-auto}"
    local coverage="${2:-false}"
    local verbose="${3:-false}"
    local module="${4:-}"
    
    _info "====> Running ST tests"
    _info "Execution mode: $exec_mode"
    
    # 自动检测模式
    if [ "$exec_mode" = "auto" ]; then
        exec_mode=$(get_exec_mode)
        _info "Auto-detected mode: $exec_mode"
    fi
    
    PYTEST_CMD="pytest -c tests/st/pytest.ini"
    
    if [ "$verbose" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -sv"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --exec-mode=$exec_mode"
    
    if [ "$coverage" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov=vllm_ascend \
            --cov-config=tests/st/.coveragerc \
            --cov-report=xml:st-tests-coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=80 \
            --cov-branch"
    fi
    
    # 确定测试路径
    if [ -n "$module" ]; then
        TEST_PATH="tests/st/$module/"
        _info "Module: $module"
    else
        TEST_PATH="tests/st/"
    fi
    
    PYTEST_CMD="$PYTEST_CMD $TEST_PATH"
    
    _info "Command: $PYTEST_CMD"
    cd "$PROJECT_ROOT"
    $PYTEST_CMD
    
    _success "====> ST tests passed"
}

run_e2e_tests() {
    local test_type="${1:-singlecard}"
    local verbose="${2:-false}"
    
    _info "====> Running E2E tests"
    _info "Test type: $test_type"
    
    if [ "$test_type" = "singlecard" ]; then
        TEST_PATH="tests/e2e/singlecard/"
    elif [ "$test_type" = "multicard" ]; then
        TEST_PATH="tests/e2e/multicard/"
    else
        TEST_PATH="tests/e2e/"
    fi
    
    PYTEST_CMD="pytest"
    if [ "$verbose" = true ]; then
        PYTEST_CMD="$PYTEST_CMD -sv"
    fi
    
    PYTEST_CMD="$PYTEST_CMD $TEST_PATH"
    
    _info "Command: $PYTEST_CMD"
    cd "$PROJECT_ROOT"
    $PYTEST_CMD
    
    _success "====> E2E tests passed"
}

# 环境设置函数
setup_vllm_env() {
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
    export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
    
    if check_cann_env; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib"
    fi
    
    _info "Environment setup:"
    _info "  VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"
    _info "  VLLM_USE_MODELSCOPE=$VLLM_USE_MODELSCOPE"
    _info "  VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
    _info "  TORCH_DEVICE_BACKEND_AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"
}

# 打印测试摘要
print_test_summary() {
    local ut_result="${1:-unknown}"
    local st_result="${2:-unknown}"
    local e2e_result="${3:-unknown}"
    
    _blue "======================================"
    _blue "Test Summary"
    _blue "======================================"
    _info "UT tests: $ut_result"
    _info "ST tests: $st_result"
    _info "E2E tests: $e2e_result"
    _blue "======================================"
}