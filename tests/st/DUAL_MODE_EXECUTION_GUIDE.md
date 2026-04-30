# 双模式执行策略指南

本文档说明ST测试的CPU Mock和NPU真实双模式执行策略，以及CI配置指南。

## 1. 双模式概述

ST测试支持两种执行模式：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         双模式执行策略                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU Mock模式                                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Mock策略：Mock torch_npu.npu_*、torch.npu.*、is_310p                       │
│  执行环境：x86/arm服务器、MacBook、无torch_npu环境                            │
│  执行时间：快速（秒级），≤5分钟                                               │
│  测试目的：验证模块协作、验证接口兼容、快速反馈、CI门禁                        │
│                                                                             │
│  NPU真实模式                                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Mock策略：不Mock，使用真实torch_npu                                         │
│  执行环境：Ascend NPU、Atlas 800、Atlas A2                                   │
│  执行时间：较长（分钟级），可选执行                                            │
│  测试目的：验证真实性能、验证真实精度、深度集成测试                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. CPU Mock模式详细说明

### 2.1 Mock边界

CPU Mock模式Mock以下内容：

| Mock内容 | 说明 |
|---------|------|
| torch_npu.npu_quantize | 量化算子 |
| torch_npu.npu_swiglu | SwiGLU算子 |
| torch_npu.npu_flash_attention | Flash Attention算子 |
| torch.npu.set_device | 设备设置 |
| torch.npu.empty_cache | 内存清理 |
| torch.npu.synchronize | 设备同步 |
| vllm_ascend.utils.is_310p | NPU类型检测 |

### 2.2 不Mock内容

保留插件模块协作：

| 不Mock内容 | 说明 |
|-----------|------|
| NPUWorker | 插件Worker组件 |
| AscendModelRunner | 插件ModelRunner组件 |
| AscendScheduler | 插件Scheduler组件 |
| AscendAttention | 插件Attention组件 |

### 2.3 使用场景

- CI快速门禁（lint → ut → st-cpu-mock → e2e）
- 开发环境无NPU硬件
- 快速验证模块协作逻辑
- 接口兼容性验证

### 2.4 命令示例

```bash
pytest tests/st/ --exec-mode=cpu_mock
pytest tests/st/ --exec-mode=cpu_mock --cov=vllm_ascend
pytest tests/st/worker/ --exec-mode=cpu_mock -v
```

## 3. NPU真实模式详细说明

### 3.1 环境要求

| 要求 | 说明 |
|------|------|
| torch_npu | 必须安装 |
| NPU硬件 | Ascend NPU可用 |
| CANN环境 | CANN container环境 |

### 3.2 使用场景

- 性能基准测试
- 精度验证测试
- 周边组件集成测试（CANN、ACL Graph、HCCL）
- 深度集成测试

### 3.3 命令示例

```bash
pytest tests/st/ --exec-mode=npu_real
pytest tests/st/ --exec-mode=npu_real --performance-test
pytest tests/st/ --exec-mode=npu_real --precision-test
pytest tests/st/ --exec-mode=npu_real --npu-device=0
```

## 4. 自动检测模式

`--exec-mode=auto`自动检测环境并选择模式：

```
检测流程：
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Level 1: torch可用性                                                       │
│  ├─ 无torch → skip（跳过测试）                                               │
│  └─ 有torch → 继续检测                                                       │
│                                                                             │
│  Level 2: torch_npu可用性                                                   │
│  ├─ 无torch_npu → cpu_mock                                                  │
│  └─ 有torch_npu → 继续检测                                                   │
│                                                                             │
│  Level 3: NPU硬件可用性                                                      │
│  ├─ NPU不可用 → cpu_mock                                                    │
│  └─ NPU可用 → npu_real                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. CI配置指南

### 5.1 CI快速门禁（CPU Mock）

```yaml
st-cpu-mock:
  runs-on: ubuntu-latest
  container: quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11
  steps:
    - name: Run ST test
      run: |
        pytest tests/st/ --exec-mode=cpu_mock \
          --cov=vllm_ascend \
          --cov-fail-under=80 \
          -n 4
```

**执行时间要求**：≤5分钟

### 5.2 NPU真实测试（可选）

```yaml
st-npu-real:
  runs-on: npu-runner
  if: ${{ optional_trigger }}
  steps:
    - name: Run ST test
      run: |
        pytest tests/st/ --exec-mode=npu_real \
          --performance-test \
          --precision-test
```

**执行时间**：分钟级，可选执行

### 5.3 执行顺序

```
lint → ut → st-cpu-mock → e2e

┌────────┐   ┌────────┐   ┌────────────┐   ┌───────┐
│  lint  │──▶│   ut   │──▶│st-cpu-mock │──▶│  e2e  │
│ ubuntu │   │container│   │  container │   │  NPU  │
│ ~1min  │   │ ~2min   │   │   ~5min    │   │~30min │
└────────┘   └────────┘   └────────────┘   └───────┘
```

### 5.4 coverage配置

```yaml
codecov:
  flags:
    st_tests:
      paths:
        - "tests/st/"
```

## 6. 开发环境使用

### 6.1 无NPU环境

```bash
pytest tests/st/ --exec-mode=cpu_mock
pytest tests/st/ --exec-mode=auto  # 自动选择cpu_mock
```

### 6.2 有NPU环境

```bash
pytest tests/st/ --exec-mode=auto  # 自动选择npu_real
pytest tests/st/ --exec-mode=cpu_mock  # 强制CPU Mock
pytest tests/st/ --exec-mode=npu_real  # 强制NPU真实
```

### 6.3 调试模式

```bash
pytest tests/st/ --exec-mode=npu_real --enable-mock  # NPU环境强制Mock
pytest tests/st/ -sv --tb=long  # 详细输出
```

## 7. 测试结果对比

| 维度 | CPU Mock | NPU真实 |
|------|---------|---------|
| 执行速度 | 秒级 | 分钟级 |
| 硬件依赖 | 无 | NPU |
| 测试覆盖 | 模块协作 | 性能+精度 |
| CI门禁 | 必须 | 可选 |
| 适用场景 | 快速反馈 | 深度验证 |

## 8. 参考文档

- tests/st/utils/env_detector.py：环境检测机制
- tests/st/utils/env_factory.py：执行环境工厂
- tests/st/conftest.py：pytest_addoption参数定义