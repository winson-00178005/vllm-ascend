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

# UT+ST联合测试执行脚本
# 用途: 快速执行UT和ST测试，适合本地开发和CI快速门禁
# 执行顺序: UT → ST (模拟GitHub Actions workflow)

set -eo errexit

. $(dirname "$0")/common.sh

# 默认参数
COVERAGE=false
VERBOSE=false
ST_EXEC_MODE="cpu_mock"  # 默认使用CPU Mock模式
ST_MODULE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --quick)
            # 快速模式: 仅运行核心模块测试
            ST_MODULE="worker"
            shift
            ;;
        *)
            _err "Unknown parameter: $1"
            ;;
    esac
done

# 环境设置
setup_vllm_env

_cyan "======================================"
_cyan "Running UT + ST Tests"
_cyan "======================================"
_info "Mode: Quick validation (UT + ST)"
_info "Coverage: $COVERAGE"
_info "Verbose: $VERBOSE"
_info "ST execution mode: $ST_EXEC_MODE"
if [ -n "$ST_MODULE" ]; then
    _info "ST module: $ST_MODULE"
fi
_cyan "======================================"

# 执行顺序: UT → ST (模拟workflow: lint → ut → st-cpu-mock → e2e)
UT_RESULT="unknown"
ST_RESULT="unknown"

# 1. 执行UT测试
_info "===> [Step 1/2] Running UT tests"
if run_ut_tests "$COVERAGE" "$VERBOSE"; then
    UT_RESULT="passed"
    _success "===> UT tests passed ✓"
else
    UT_RESULT="failed"
    _err "===> UT tests failed ✗"
fi

# 2. 执行ST测试
_info "===> [Step 2/2] Running ST tests"
if run_st_tests "$ST_EXEC_MODE" "$COVERAGE" "$VERBOSE" "$ST_MODULE"; then
    ST_RESULT="passed"
    _success "===> ST tests passed ✓"
else
    ST_RESULT="failed"
    _err "===> ST tests failed ✗"
fi

# 打印测试摘要
print_test_summary "$UT_RESULT" "$ST_RESULT" "skipped"

# 显示覆盖率信息
if [ "$COVERAGE" = true ]; then
    _cyan "======================================"
    _cyan "Coverage Summary"
    _cyan "======================================"
    
    if [ -f "unittests-coverage.xml" ]; then
        UT_COV=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('unittests-coverage.xml')
root = tree.getroot()
print(f'{float(root.attrib.get('line-rate', 0)) * 100:.2f}%')
" 2>/dev/null || echo "unknown")
        _info "UT coverage: $UT_COV"
    fi
    
    if [ -f "st-tests-coverage.xml" ]; then
        ST_COV=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('st-tests-coverage.xml')
root = tree.getroot()
print(f'{float(root.attrib.get('line-rate', 0)) * 100:.2f}%')
" 2>/dev/null || echo "unknown")
        _info "ST coverage: $ST_COV"
    fi
    
    _info "HTML reports:"
    _info "  UT: (check terminal output)"
    _info "  ST: tests/st/htmlcov_st/index.html"
fi

_success "All UT+ST tests completed successfully!"
_info "Next step: Run E2E tests (requires real NPU hardware)"