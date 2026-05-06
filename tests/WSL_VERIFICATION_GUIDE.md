# WSL环境ST测试验证指南

本文档提供在WSL环境中验证ST测试框架的完整指南。

## 目录

- [环境要求](#环境要求)
- [快速验证](#快速验证)
- [完整验证](#完整验证)
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
chmod +x tests/st/scripts/*.sh

# 快速检查环境
bash tests/common.sh
```

## 快速验证

### 方案1: 使用统一测试脚本（推荐）

```bash
# 快速验证UT+ST测试（推荐）
./tests/run_ut_st.sh --coverage -v --st-exec-mode=cpu_mock

# 快速模式（仅Worker模块）
./tests/run_ut_st.sh --quick

# 指定ST模块测试
./tests/run_ut_st.sh --st-module=worker --coverage

# 指定ST模块测试（其他模块）
./tests/run_ut_st.sh --st-module=attention --coverage
```

### 方案2: 使用ST测试脚本

```bash
# 仅运行ST测试
./tests/st/scripts/run_st_tests.sh --exec-mode=cpu_mock --coverage

# 快速模式（仅Worker模块）
./tests/st/scripts/run_st_tests.sh --module=worker --quick

# 并行执行（加速）
./tests/st/scripts/run_st_tests.sh --parallel=4 --coverage

# 自动检测模式
./tests/st/scripts/run_st_tests.sh --exec-mode=auto --coverage
```

### 方案3: 直接使用pytest

```bash
# 运行单个测试文件
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
./tests/run_all_tests.sh --coverage -v --st-exec-mode=cpu_mock

# 跳过UT，仅运行ST
./tests/run_all_tests.sh --skip-ut --coverage -v

# 包含E2E测试（需要真实NPU）
./tests/run_all_tests.sh --run-e2e --coverage -v --e2e-type=singlecard
```

### 测试执行顺序

```
┌────────────┐
│ 1. UT测试  │ ← 单元测试，验证单个函数/类
└─────┬──────┘
      │
      ▼
┌────────────┐
│ 2. ST测试  │ ← 系统集成测试，验证模块协作
│ (CPU Mock)│   使用Mock模拟NPU环境
└─────┬──────┘
      │
      ▼
┌────────────┐
│ 3. E2E测试 │ ← 端到端测试，完整推理流程
│  (可选)    │   需要真实NPU硬件
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

## 常见问题

### Q1: ImportError: torch_npu not found

**问题**: WSL环境缺少torch_npu模块

**解决方案**:

```bash
# 方案1: 安装CANN环境（推荐）
# 参考: https://www.hiascend.com/document

# 方案2: 使用fake torch_npu（仅测试框架）
mkdir -p /tmp/fake_torch_npu
cat > /tmp/fake_torch_npu/__init__.py << 'EOF'
import torch

def npu_current_stream(device=None):
    return torch.cuda.current_stream()

def npu_stream(stream):
    return torch.cuda.stream(stream)

def npu_device_count():
    return 0

torch.npu = type('obj', (object,), {
    'is_available': lambda: False,
    'device_count': npu_device_count,
    'current_stream': npu_current_stream,
    'stream': npu_stream,
})

__version__ = "fake-0.1.0"
EOF

export PYTHONPATH="/tmp/fake_torch_npu:$PYTHONPATH"
```

### Q2: 测试执行时间过长

**解决方案**: 使用快速模式或并行执行

```bash
# 快速模式（仅核心模块）
./tests/run_ut_st.sh --quick

# 并行执行（4进程）
./tests/st/scripts/run_st_tests.sh --parallel=4

# 指定模块测试
./tests/run_ut_st.sh --st-module=worker
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

### Q4: Mock对象行为异常

**解决方案**: 检查spec限制

```python
# 查看测试代码中的Mock定义
# tests/st/utils/mock_utils.py

# 验证vllm接口spec
# tests/st/utils/mock_utils.py:VLLM_WORKER_SPEC
```

## 测试脚本参数说明

### run_ut_st.sh

| 参数 | 说明 | 默认值 |
|------|------|--------|
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