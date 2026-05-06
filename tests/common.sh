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
_fail() { _red "Failed: $*"; return 1; }

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

# WSL环境特殊处理：自动创建fake torch_npu
setup_fake_torch_npu() {
    _warning "torch_npu未安装，创建临时fake模块（仅验证框架）"
    
    FAKE_DIR="/tmp/fake_torch_npu_$$"
    mkdir -p "$FAKE_DIR"
    
    cat > "$FAKE_DIR/__init__.py" << 'EOF'
import torch

def npu_current_stream(device=None):
    try: return torch.cuda.current_stream()
    except: return None

def npu_stream(stream):
    try: return torch.cuda.stream(stream)
    except: return stream

def npu_device_count(): return 0
def npu_set_device(device): pass
def npu_synchronize():
    try: torch.cuda.synchronize()
    except: pass

torch.npu = type('obj', (object,), {
    'is_available': lambda: False,
    'device_count': npu_device_count,
    'current_stream': npu_current_stream,
    'stream': npu_stream,
})

__version__ = "fake-0.1.0"
EOF
    
    export PYTHONPATH="$FAKE_DIR:$PYTHONPATH"
    _info "fake torch_npu已创建: $FAKE_DIR"
    
    # 验证fake模块
    if python3 -c "import torch_npu" 2>/dev/null; then
        _success "fake torch_npu验证成功"
        return 0
    else
        _err "fake torch_npu创建失败"
        return 1
    fi
}

# 自动处理torch_npu依赖
auto_handle_torch_npu() {
    local use_fake="${1:-false}"
    
    if check_torch_npu; then
        _success "torch_npu环境正常"
        return 0
    fi
    
    # torch_npu缺失时的处理
    if [ "$use_fake" = true ]; then
        _warning "使用fake torch_npu模式（仅验证框架）"
        setup_fake_torch_npu || return 1
        
        # 设置标记，告知后续测试使用fake环境
        export VLLM_USE_FAKE_TORCH_NPU=true
        export VLLM_EXEC_MODE=cpu_mock
        
        _warning "⚠️  重要提醒:"
        _warning "  • 使用fake torch_npu环境"
        _warning "  • 仅验证测试框架设计"
        _warning "  • 不验证真实NPU功能"
        _warning "  • 某些测试可能失败"
        
        return 0
    else
        _err "torch_npu缺失，测试无法执行"
        _info "解决方案:"
        _info "  1. 安装CANN环境: https://www.hiascend.com/document"
        _info "  2. 使用Docker容器: quay.io/ascend/cann:8.2.rc1"
        _info "  3. 使用 --use-fake 参数快速验证框架"
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
    
    # 先验证环境，避免ImportError导致脚本提前退出
    if ! python3 -c "import vllm_ascend" 2>/dev/null; then
        _warning "vllm_ascend导入失败，请检查环境依赖"
        _info "提示: 使用 --use-fake 参数创建fake torch_npu环境"
        return 1
    fi
    
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
    
    # 执行pytest并捕获退出状态
    $PYTEST_CMD || {
        local pytest_status=$?
        _fail "====> UT tests failed (exit code: $pytest_status)"
        return $pytest_status
    }
    
    _success "====> UT tests passed"
    return 0
}

run_st_tests() {
    local exec_mode="${1:-auto}"
    local coverage="${2:-false}"
    local verbose="${3:-false}"
    local module="${4:-}"
    
    _info "====> Running ST tests"
    _info "Execution mode: $exec_mode"
    
    # 先验证环境，避免ImportError导致脚本提前退出
    if ! python3 -c "import vllm_ascend" 2>/dev/null; then
        _warning "vllm_ascend导入失败，请检查环境依赖"
        _info "提示: 使用 --use-fake 参数创建fake torch_npu环境"
        return 1
    fi
    
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
    
    # 执行pytest并捕获退出状态
    $PYTEST_CMD || {
        local pytest_status=$?
        _fail "====> ST tests failed (exit code: $pytest_status)"
        return $pytest_status
    }
    
    _success "====> ST tests passed"
    return 0
}

run_e2e_tests() {
    local test_type="${1:-singlecard}"
    local verbose="${2:-false}"
    
    _info "====> Running E2E tests"
    _info "Test type: $test_type"
    
    # 先验证环境，避免ImportError导致脚本提前退出
    if ! python3 -c "import vllm_ascend" 2>/dev/null; then
        _warning "vllm_ascend导入失败，请检查环境依赖"
        _info "提示: E2E测试需要真实NPU环境，请在Docker容器或CANN环境中执行"
        return 1
    fi
    
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
    
    # 执行pytest并捕获退出状态
    $PYTEST_CMD || {
        local pytest_status=$?
        _fail "====> E2E tests failed (exit code: $pytest_status)"
        return $pytest_status
    }
    
    _success "====> E2E tests passed"
    return 0
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