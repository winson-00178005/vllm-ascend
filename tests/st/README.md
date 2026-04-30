# ST测试框架

System Integration Test (ST) 框架用于验证vllm-ascend插件内部模块协作。

## 概述

ST测试位于测试金字塔中间层：
- **UT (tests/ut/)**: 单个函数/类级别测试
- **ST (tests/st/)**: 插件内部模块协作测试
- **E2E (tests/e2e/)**: 完整推理流程测试

## 核心特性

### 双模式执行

ST测试支持两种执行模式：

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| CPU Mock | Mock NPU硬件，快速执行 | CI快速门禁，开发调试 |
| NPU Real | 真实NPU硬件执行 | 性能测试，精度验证 |

执行模式选择：
```bash
# 自动检测模式
pytest tests/st/ --exec-mode=auto

# 强制CPU Mock模式
pytest tests/st/ --exec-mode=cpu_mock

# NPU Real模式（需要真实NPU）
pytest tests/st/ --exec-mode=npu_real
```

### STRunner资源管理

STRunner提供上下文管理器模式管理测试资源：

```python
from tests.st.utils.runner_factory import STRunner

with STRunner("worker", config) as runner:
    runner.register_resource("worker", mock_worker)
    runner.register_resource("model_runner", mock_model_runner)
    
    # 测试代码
    
# 资源自动清理
```

### Mock工厂函数

提供统一的Mock创建接口：

```python
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_scheduler,
    create_mock_attention,
    verify_mock_calls,
)

# 使用spec限制Mock
worker = create_mock_worker(spec_attrs=['execute_model', 'load_model'])

# 验证Mock调用
verify_mock_calls(worker.execute_model, expected_calls=1)
```

## 测试模块

| 模块 | 测试内容 | 测试文件 |
|------|---------|---------|
| Worker | Worker-ModelRunner协作 | tests/st/worker/ |
| Scheduler | Scheduler-Worker协作 | tests/st/core/ |
| Attention | Attention-Mask-MLA协作 | tests/st/attention/ |
| Quantization | Quantizer-模型加载协作 | tests/st/quantization/ |
| Distributed | Communicator-TP-EP协作 | tests/st/distributed/ |
| Sample | Sampler-RejectionSampler协作 | tests/st/sample/ |

## 运行测试

### 单模块测试

```bash
# Worker模块测试
./tests/st/scripts/run_st_tests.sh --module worker

# Scheduler模块测试
./tests/st/scripts/run_st_tests.sh --module core
```

### 全量测试

```bash
# 全量ST测试
./tests/st/scripts/run_st_tests.sh

# 带覆盖率
./tests/st/scripts/run_st_tests.sh --coverage
```

### 并行执行

```bash
# 使用pytest-xdist并行执行（4进程）
pytest tests/st/ -n 4 --exec-mode=cpu_mock

# 自动检测进程数
pytest tests/st/ -n auto --exec-mode=cpu_mock
```

### 覆盖率报告

```bash
# 生成覆盖率报告
pytest tests/st/ --cov=vllm_ascend \
    --cov-report=term-missing \
    --cov-report=xml:coverage.xml \
    --cov-report=html:htmlcov \
    --cov-fail-under=80
```

## 插件测试职责

ST测试严格遵循职责边界：

| 应该测试 | 不应该测试 |
|----------|-----------|
| 插件内部模块协作 | vllm核心逻辑 |
| 插件接口实现验证 | vllm调度算法 |
| 插件替换逻辑 | 完整推理流程 |
| 插件数据流处理 | vllm.AttentionBackend核心 |

详细职责边界见：tests/st/PLUGIN_TEST_RESPONSIBILITY.md

## 测试编写规范

### Docstring模板

每个测试方法必须包含完整Docstring：

```python
def test_worker_schedule_to_runner(self, batch_size):
    """Test Worker调度请求到ModelRunner执行
    
    验证：
    - Worker正确传递请求参数到ModelRunner
    - ModelRunner执行推理并返回输出tensor
    - 异常情况下Worker正确处理错误
    
    场景：正常执行流程和异常处理
    
    预期结果：执行成功或错误正确传递
    
    执行模式：CPU Mock / NPU Real
    """
```

### 参数化指南

```python
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_worker_parameterized(self, batch_size, dtype):
    """参数化测试"""
```

详细规范见：tests/st/ST_TEST_WRITING_GUIDE.md

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --exec-mode | 执行模式 (auto/cpu_mock/npu_real) | auto |
| --enable-mock | 强制启用Mock | False |
| --performance-test | 启用性能测试 | False |
| --precision-test | 启用精度测试 | False |
| --npu-device | NPU设备ID | 0 |

详细参数说明见：tests/st/PYTEST_CLI_GUIDE.md

## 目录结构

```
tests/st/
├── base.py                  # 测试基类
├── conftest.py              # 全局fixtures
├── utils/                   # 工具类
│   ├── env_detector.py      # 环境检测
│   ├── env_factory.py       # 环境工厂
│   ├── mock_utils.py        # Mock工具
│   ├── data_generator.py    # 数据生成器
│   ├── config_factory.py    # 配置工厂
│   └── runner_factory.py    # STRunner
├── scripts/
│   └ run_st_tests.sh        # 测试运行脚本
├── worker/                  # Worker模块测试
├── core/                    # Scheduler模块测试
├── attention/               # Attention模块测试
├── quantization/            # Quantization模块测试
├── distributed/             # Distributed模块测试
├── sample/                  # Sample模块测试
└── *.md                     # 文档
```

## 参考文档

- [ST测试编写规范](ST_TEST_WRITING_GUIDE.md)
- [插件测试职责边界](PLUGIN_TEST_RESPONSIBILITY.md)
- [Pytest命令行参数](PYTEST_CLI_GUIDE.md)
- [双模式执行策略](DUAL_MODE_EXECUTION_GUIDE.md)