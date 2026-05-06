# WSL环境ST测试完整验证指南

本文档提供在WSL环境中验证ST测试框架的完整指南。

## 目录

- [环境要求](#环境要求)
- [快速验证](#快速验证)
- [完整验证](#完整验证)
- [脚本修改说明](#脚本修改说明)
- [常见问题](#常见问题)

## 环境要求

### 必需依赖

```
┌─────────────────────────────────────┐
│  Python 3.8+                        │
│  pytest + 插件                      │
│  ├─ pytest-mock                     │
│  ├─ pytest-cov                      │
│  ├─ pytest-xdist (可选)             │
│  └─ pytest-timeout                  │
│  torch                              │
│  torch_npu (关键依赖)                │
└─────────────────────────────────────┘
```

### ⚠️ 重要提示

ST测试强依赖`torch_npu`，即使CPU Mock模式也需要安装：
- 源代码import torch_npu
- Mock框架需要torch_npu模块存在
- 无真实NPU硬件可使用fake torch_npu

### 环境检查

```bash
# 进入项目目录
cd /mnt/d/vllm/vllm-ascend  # WSL路径示例

# 添加脚本执行权限
chmod +x tests/*.sh

# 快速检查环境
python3 --version
python3 -c "import pytest; print(pytest.__version__)"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch_npu; print('torch_npu OK')" || echo "需要torch_npu"
```

## 快速验证

### 方案1: 使用统一测试脚本（推荐）

```bash
# WSL环境：使用fake torch_npu快速验证（自动处理依赖）
./tests/run_ut_st.sh --use-fake --quick --coverage -v

# 快速模式（仅Worker模块）
./tests/run_ut_st.sh --use-fake --quick

# 指定ST模块测试
./tests/run_ut_st.sh --use-fake --st-module=worker --coverage

# 指定ST模块测试（其他模块）
./tests/run_ut_st.sh --use-fake --st-module=attention --coverage
```

### 方案2: 使用ST测试脚本

```bash
# 仅运行ST测试（需要先处理torch_npu依赖）
# 方式A: 使用fake torch_npu
export PYTHONPATH="/tmp/fake_torch_npu:$PYTHONPATH"
./tests/st/scripts/run_st_tests.sh --exec-mode=cpu_mock --coverage

# 方式B: 快速模式（仅Worker模块）
./tests/st/scripts/run_st_tests.sh --module=worker --quick

# 方式C: 并行执行（加速）
./tests/st/scripts/run_st_tests.sh --parallel=4 --coverage

# 方式D: 自动检测模式
./tests/st/scripts/run_st_tests.sh --exec-mode=auto --coverage
```

### 方案3: 直接使用pytest

```bash
# 运行单个测试文件（需先处理torch_npu依赖）
pytest tests/st/worker/test_worker_model_runner_integration.py \
    --exec-mode=cpu_mock -sv

# 运行单个模块
pytest tests/st/worker/ --exec-mode=cpu_mock -sv

# 运行全量ST测试
pytest tests/st/ --exec-mode=cpu_mock -sv --cov=vllm_ascend \
    --cov-config=tests/st/.coveragerc \
    --cov-report=term-missing \
    --cov-fail-under=80
```

## 完整验证

### 模拟CI流程

```bash
# 模拟完整CI流程: UT → ST → (E2E可选)
./tests/run_all_tests.sh --use-fake --coverage -v --st-exec-mode=cpu_mock

# 跳过UT，仅运行ST
./tests/run_all_tests.sh --use-fake --skip-ut --coverage -v

# 包含E2E测试（需要真实NPU）
./tests/run_all_tests.sh --run-e2e --coverage -v --e2e-type=singlecard
```

### 测试执行顺序

```
┌────────────┐
│ 1. UT测试  │ ← 单元测试，验证单个函数/类
│            │   需要torch_npu环境
└─────┬──────┘
      │
      ▼
┌────────────┐
│ 2. ST测试  │ ← 系统集成测试，验证模块协作
│ (CPU Mock)│   使用Mock模拟NPU环境
│            │   需要torch_npu环境
└─────┬──────┘
      │
      ▼
┌────────────┐
│ 3. E2E测试 │ ← 端到端测试，完整推理流程
│  (可选)    │   需要真实NPU硬件
│            │   需要真实torch_npu
└────────────┘
```

### 覆盖率验证

```bash
# 生成覆盖率报告
pytest tests/st/ --exec-mode=cpu_mock \
    --cov=vllm_ascend \
    --cov-config=tests/st/.coveragerc \
    --cov-report=term-missing \
    --cov-report=xml:st-coverage.xml \
    --cov-report=html:tests/st/htmlcov_st \
    --cov-fail-under=80

# 查看HTML报告
# 在浏览器中打开: tests/st/htmlcov_st/index.html
```

## 脚本修改说明

### 问题1: 错误报告"Success"

**原始问题**:
- pytest失败后，脚本继续执行
- 错误显示"Success"（误导性）

**修改内容**:
```bash
# 修改前（错误）
$PYTEST_CMD
_success "====> UT tests passed"  # 总是执行

# 修改后（正确）
# 先验证环境
if ! python3 -c "import vllm_ascend" 2>/dev/null; then
    _warning "vllm_ascend导入失败，请检查环境依赖"
    return 1
fi

# 执行pytest并捕获失败
$PYTEST_CMD || {
    local pytest_status=$?
    _fail "====> UT tests failed (exit code: $pytest_status)"
    return $pytest_status
}
_success "====> UT tests passed"  # 仅成功时执行
```

**关键改进**:
- ✓ pytest执行前验证环境
- ✓ 使用 `|| { ... }` 捕获失败
- ✓ 失败时立即返回，停止执行
- ✓ 成功时才显示"Success"

### 问题2: torch_npu依赖处理

**原始问题**:
- 测试脚本未处理torch_npu缺失
- 用户需要手动处理依赖

**修改内容**:
```bash
# 添加 --use-fake 参数
./tests/run_ut_st.sh --use-fake --quick

# 脚本自动处理：
auto_handle_torch_npu "$USE_FAKE"

# 如果use_fake=true:
setup_fake_torch_npu() {
    # 自动创建fake torch_npu模块
    FAKE_DIR="/tmp/fake_torch_npu_$$"
    mkdir -p "$FAKE_DIR"
    
    cat > "$FAKE_DIR/__init__.py" << 'EOF'
    import torch
    # Mock关键函数...
    EOF
    
    export PYTHONPATH="$FAKE_DIR:$PYTHONPATH"
}
```

**关键改进**:
- ✓ 自动检测torch_npu缺失
- ✓ 自动创建fake模块
- ✓ 自动设置PYTHONPATH
- ✓ 无需手动操作

### 问题3: 删除文档不当

**原始问题**:
- 删除了详细指导文档
- 用户无法了解完整信息

**修正内容**:
- ✓ 恢复详细指导文档
- ✓ 保留所有重要信息
- ✓ 提供完整使用说明

## 常见问题

### Q1: ImportError: torch_npu not found

**问题**: WSL环境缺少torch_npu模块

**解决方案**:

```bash
# 方案1: 使用 --use-fake 参数（推荐，快速验证）
./tests/run_ut_st.sh --use-fake --quick --coverage -v

# 方案2: 安装CANN环境（完整环境）
# 参考: https://www.hiascend.com/document

# 方案3: 使用Docker容器（完整环境）
docker pull quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11
docker run -it --rm \
  -v /mnt/d/vllm/vllm-ascend:/workspace \
  quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11 \
  bash

# 在容器内执行（无需--use-fake）
cd /workspace
./tests/run_ut_st.sh --coverage -v
```

### Q2: 测试执行时间过长

**解决方案**: 使用快速模式或并行执行

```bash
# 快速模式（仅核心模块）
./tests/run_ut_st.sh --use-fake --quick

# 并行执行（4进程）
./tests/st/scripts/run_st_tests.sh --parallel=4

# 指定模块测试
./tests/run_ut_st.sh --use-fake --st-module=worker
```

### Q3: 覆盖率未达到阈值

**解决方案**: 检查覆盖率报告，补充测试

```bash
# 查看覆盖率HTML报告
# tests/st/htmlcov_st/index.html

# 降低阈值（临时）
./tests/st/scripts/run_st_tests.sh --coverage --fail-under=60

# 运行覆盖率并查看缺失行
pytest tests/st/ --cov=vllm_ascend --cov-report=term-missing
```

### Q4: fake torch_npu有什么限制？

**限制说明**:
```
┌──────────────────────────────────────┐
│  fake torch_npu仅用于验证框架设计     │
├──────────────────────────────────────┤
│                                      │
│  限制:                                │
│  • 不提供真实NPU功能                  │
│  • 不验证NPU相关逻辑                  │
│  • 某些测试可能因Mock行为失败          │
│                                      │
│  适用场景:                            │
│  ✓ 验证测试框架结构                   │
│  ✓ 检查Mock工厂设计                   │
│  ✓ 验证环境检测机制                   │
│  ✓ 快速原型验证                       │
│                                      │
│  不适用:                              │
│  ✗ 实际功能测试                       │
│  ✗ 性能测试                           │
│  ✗ 精度测试                           │
│  ✗ 生产环境                           │
│                                      │
└──────────────────────────────────────┘
```

### Q5: 如何验证真实NPU功能？

**解决方案**: 使用Docker容器或安装CANN环境

```bash
# 方案A: Docker容器（推荐）
docker pull quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11
docker run -it --rm \
  -v /mnt/d/vllm/vllm-ascend:/workspace \
  quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11 \
  bash

cd /workspace
# 无需--use-fake，容器内有真实torch_npu
./tests/run_ut_st.sh --coverage -v

# 方案B: 安装CANN环境（生产环境）
# 参考: https://www.hiascend.com/document
```

### Q6: 为什么删除文档？

**原因说明**:
- 第一次删除文档是**错误决定**
- 应该保留详细说明
- 现已恢复完整文档

## 测试脚本参数说明

### run_ut_st.sh

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --use-fake | 使用fake torch_npu（WSL环境） | false |
| --coverage | 启用覆盖率报告 | false |
| -v/--verbose | 详细输出 | false |
| --st-exec-mode | ST执行模式 | cpu_mock |
| --st-module | 指定ST模块 | (all) |
| --quick | 快速模式（仅Worker） | false |

### run_st_tests.sh

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --exec-mode | 执行模式(cpu_mock/npu_real/auto) | cpu_mock |
| --coverage | 启用覆盖率报告 | false |
| --module | 指定测试模块 | (all) |
| --parallel | 并行进程数 | (无) |
| --fail-under | 覆盖率阈值 | 80 |
| --quick | 快速模式 | false |
| -m/--marker | pytest marker过滤 | (无) |

### run_all_tests.sh

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --use-fake | 使用fake torch_npu（WSL环境） | false |
| --skip-ut | 跳过UT测试 | false |
| --skip-st | 跳过ST测试 | false |
| --run-e2e | 运行E2E测试 | false |
| --coverage | 启用覆盖率报告 | false |
| -v/--verbose | 详细输出 | false |
| --st-exec-mode | ST执行模式 | auto |
| --st-module | 指定ST模块 | (all) |
| --e2e-type | E2E类型(singlecard/multicard) | singlecard |

## 测试模块列表

| 模块 | 测试内容 | 路径 |
|------|---------|------|
| worker | Worker-ModelRunner协作 | tests/st/worker/ |
| core | Scheduler-Worker协作 | tests/st/core/ |
| attention | Attention-Mask-MLA协作 | tests/st/attention/ |
| quantization | Quantizer-模型加载协作 | tests/st/quantization/ |
| distributed | Communicator-TP-EP协作 | tests/st/distributed/ |
| sample | Sampler-RejectionSampler协作 | tests/st/sample/ |

## 下一步

验证成功后，您可以：
- 修改ST测试代码，验证框架可用性
- 运行覆盖率报告，评估测试质量
- 在真实NPU环境运行npu_real模式测试
- 提交代码，触发CI自动化测试

## 参考文档

- ST测试框架: `tests/st/README.md`
- 测试编写指南: `tests/st/ST_TEST_WRITING_GUIDE.md`
- 双模式执行指南: `tests/st/DUAL_MODE_EXECUTION_GUIDE.md`
- Mock工厂指南: `tests/st/MOCK_FACTORY_GUIDE.md`
- 插件架构测试策略: `tests/st/PLUGIN_ARCHITECTURE_TEST_STRATEGY.md`

---

**最后更新**: 已修正脚本错误处理逻辑，恢复完整文档