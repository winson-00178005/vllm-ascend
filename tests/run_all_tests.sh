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

# 统一测试执行脚本
# 用途: 按顺序执行UT → ST → E2E测试，模拟CI流程

set -eo errexit

. $(dirname "$0")/common.sh

# 默认参数
RUN_UT=true
RUN_ST=true
RUN_E2E=false  # E2E默认不运行(需要真实NPU)
COVERAGE=false
VERBOSE=false
ST_EXEC_MODE="auto"
ST_MODULE=""
E2E_TYPE="singlecard"
USE_FAKE=false  # WSL环境：使用fake torch_npu

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ut)
            RUN_UT=false
            shift
            ;;
        --skip-st)
            RUN_ST=false
            shift
            ;;
        --run-e2e)
            RUN_E2E=true
            shift
            ;;
        --use-fake)
            # WSL环境：使用fake torch_npu快速验证框架
            USE_FAKE=true
            ST_EXEC_MODE="cpu_mock"  # 强制CPU Mock模式
            _warning "启用fake torch_npu模式（仅验证框架）"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --st-exec-mode)
            ST_EXEC_MODE="$2"
            shift 2
            ;;
        --st-module)
            ST_MODULE="$2"
            shift 2
            ;;
        --e2e-type)
            E2E_TYPE="$2"
            shift 2
            ;;
        *)
            _err "Unknown parameter: $1"
            ;;
    esac
done

# 环境设置
setup_vllm_env

# WSL环境：自动处理torch_npu依赖
auto_handle_torch_npu "$USE_FAKE"

_cyan "======================================"
_cyan "Running All Tests"
_cyan "======================================"
_info "UT tests: $RUN_UT"
_info "ST tests: $RUN_ST"
_info "E2E tests: $RUN_E2E"
_info "Coverage: $COVERAGE"
_info "Verbose: $VERBOSE"
_cyan "======================================"

UT_RESULT="skipped"
ST_RESULT="skipped"
E2E_RESULT="skipped"

# 执行UT测试
if [ "$RUN_UT" = true ]; then
    if run_ut_tests "$COVERAGE" "$VERBOSE"; then
        UT_RESULT="passed"
    else
        UT_RESULT="failed"
        _err "UT tests failed"
    fi
fi

# 执行ST测试
if [ "$RUN_ST" = true ]; then
    if run_st_tests "$ST_EXEC_MODE" "$COVERAGE" "$VERBOSE" "$ST_MODULE"; then
        ST_RESULT="passed"
    else
        ST_RESULT="failed"
        _err "ST tests failed"
    fi
fi

# 执行E2E测试(可选)
if [ "$RUN_E2E" = true ]; then
    _warn "E2E tests require real NPU hardware"
    if run_e2e_tests "$E2E_TYPE" "$VERBOSE"; then
        E2E_RESULT="passed"
    else
        E2E_RESULT="failed"
        _err "E2E tests failed"
    fi
fi

# 打印测试摘要
print_test_summary "$UT_RESULT" "$ST_RESULT" "$E2E_RESULT"

# 显示覆盖率报告
if [ "$COVERAGE" = true ]; then
    _cyan "======================================"
    _cyan "Coverage Reports"
    _cyan "======================================"
    if [ "$RUN_UT" = true ]; then
        _info "UT coverage: unittests-coverage.xml"
    fi
    if [ "$RUN_ST" = true ]; then
        _info "ST coverage: st-tests-coverage.xml"
        _info "ST HTML report: tests/st/htmlcov_st/index.html"
    fi
fi

_success "All tests completed successfully!"