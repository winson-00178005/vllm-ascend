# pytest命令行参数使用指南

本文档详细说明ST测试的pytest命令行参数及其使用场景。

## 1. 执行模式参数

### --exec-mode

控制测试执行模式，支持三种模式：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| auto | 自动检测环境并选择模式 | 默认模式，CI环境推荐 |
| cpu_mock | CPU Mock执行（快速） | CI快速门禁、无NPU环境 |
| npu_real | NPU真实执行（深度） | 性能测试、精度测试 |

**命令示例**：
```bash
pytest tests/st/ --exec-mode=auto          # 自动检测
pytest tests/st/ --exec-mode=cpu_mock       # CPU Mock
pytest tests/st/ --exec-mode=npu_real       # NPU真实
```

### --enable-mock / --disable-mock

强制启用或禁用Mock：

```bash
pytest tests/st/ --enable-mock              # 强制启用Mock
pytest tests/st/ --disable-mock             # 强制禁用Mock
pytest tests/st/ --exec-mode=npu_real --enable-mock  # NPU环境强制Mock（调试用）
```

## 2. 测试类型参数

### --performance-test

启用性能测试，仅在NPU真实模式下有效：

```bash
pytest tests/st/ --exec-mode=npu_real --performance-test
```

性能测试会测量：
- 执行时间
- 内存占用
- 吞吐量

### --precision-test

启用精度测试，仅在NPU真实模式下有效：

```bash
pytest tests/st/ --exec-mode=npu_real --precision-test
```

精度测试会对比：
- CPU Mock结果 vs NPU结果
- 验证数值精度（rtol=1e-3, atol=1e-5）

## 3. 设备选择参数

### --npu-device

指定NPU设备ID：

```bash
pytest tests/st/ --exec-mode=npu_real --npu-device=0
pytest tests/st/ --exec-mode=npu_real --npu-device=1
```

### --performance-baseline

指定性能基准文件路径：

```bash
pytest tests/st/ --exec-mode=npu_real \
    --performance-test \
    --performance-baseline=./baseline/qwen2.json
```

## 4. 配置文件参数

### --config-file

指定测试配置文件：

```bash
pytest tests/st/ --config-file=tests/st/configs/default.yaml
pytest tests/st/ --config-file=tests/st/configs/ci_fast.yaml
pytest tests/st/ --config-file=tests/st/configs/npu_deep.yaml
```

### --model

指定测试模型名称：

```bash
pytest tests/st/ --model=Qwen-0.5B
pytest tests/st/ --model=Qwen-7B
```

### --test-config

指定测试配置名称：

```bash
pytest tests/st/ --test-config=worker_test
pytest tests/st/ --test-config=attention_test
```

## 5. 组合使用示例

### 5.1 CI快速门禁

```bash
pytest tests/st/ --exec-mode=cpu_mock \
    --config-file=tests/st/configs/ci_fast.yaml \
    --cov=vllm_ascend \
    --cov-fail-under=80
```

### 5.2 NPU性能测试

```bash
pytest tests/st/ --exec-mode=npu_real \
    --performance-test \
    --performance-baseline=./baseline/qwen2.json \
    --npu-device=0
```

### 5.3 NPU精度测试

```bash
pytest tests/st/ --exec-mode=npu_real \
    --precision-test \
    --model=Qwen-7B
```

### 5.4 单模块测试

```bash
pytest tests/st/worker/ --exec-mode=cpu_mock
pytest tests/st/attention/ --exec-mode=npu_real --precision-test
```

## 6. pytest marker使用

### 6.1 按marker执行

```bash
pytest -m cpu_mock tests/st/          # 执行CPU Mock测试
pytest -m npu_real tests/st/          # 执行NPU真实测试
pytest -m "cpu_mock or npu_real"      # 双模式执行
pytest -m npu_performance tests/st/   # 性能测试
pytest -m npu_precision tests/st/     # 精度测试
```

### 6.2 排除marker

```bash
pytest -m "not npu_real" tests/st/    # 排除NPU真实测试
pytest -m "cpu_mock and not npu_precision" tests/st/
```

## 7. 覆盖率参数

### 7.1 基本覆盖率

```bash
pytest tests/st/ --cov=vllm_ascend
pytest tests/st/ --cov=vllm_ascend.worker
pytest tests/st/ --cov-branch          # 分支覆盖率
```

### 7.2 覆盖率报告

```bash
pytest tests/st/ --cov-report=term
pytest tests/st/ --cov-report=term-missing
pytest tests/st/ --cov-report=html
pytest tests/st/ --cov-report=xml
pytest tests/st/ --cov-report=json
```

### 7.3 覆盖率门禁

```bash
pytest tests/st/ --cov-fail-under=80   # 覆盖率≥80%
pytest tests/st/ --cov-fail-under=90   # 覆盖率≥90%
```

## 8. 并行执行

使用pytest-xdist进行并行执行：

```bash
pytest tests/st/ -n auto               # 自动检测CPU核心数
pytest tests/st/ -n 4                  # 4个进程并行
pytest tests/st/ --exec-mode=cpu_mock -n 4
```

## 9. 失败重试

使用pytest-rerunfailures进行失败重试：

```bash
pytest tests/st/ --reruns 3            # 失败重试3次
pytest tests/st/ --reruns-delay 1      # 重试间隔1秒
```

## 10. 详细输出

```bash
pytest tests/st/ -v                    # 详细输出
pytest tests/st/ -sv                   # 详细输出 + 显示print
pytest tests/st/ --tb=long             # 详细错误traceback
pytest tests/st/ --tb=short            # 简短错误traceback
```

## 11. 参考文档

- pytest官方文档: https://docs.pytest.org/
- pytest-cov文档: https://pytest-cov.readthedocs.io/
- pytest-xdist文档: https://pytest-xdist.readthedocs.io/