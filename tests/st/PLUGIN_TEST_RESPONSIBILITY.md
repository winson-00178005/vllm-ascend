# 插件测试职责边界说明

本文档明确vllm-ascend ST测试的职责边界，确保测试不越界到其他测试层职责。

## 1. 测试分层职责

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         测试分层职责划分                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UT（tests/ut/）                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  职责：验证单个函数/类级别的功能                                              │
│  测试内容：插件类、函数、配置                                                 │
│  Mock策略：Mock torch_npu硬件                                               │
│  不测试：模块间协作、完整推理流程                                             │
│                                                                             │
│  ST（tests/st/）                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  职责：验证插件内部模块间协作                                                 │
│  测试内容：Worker+ModelRunner协作、Scheduler+Worker协作等                    │
│  Mock策略：Mock vllm Core和torch_npu                                        │
│  不测试：vllm核心逻辑、完整推理流程                                           │
│                                                                             │
│  E2E（tests/e2e/）                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  职责：验证插件与vllm完整集成                                                 │
│  测试内容：插件注入、完整推理流程                                             │
│  Mock策略：不Mock，使用真实vllm和NPU                                         │
│  不测试：单个函数逻辑、模块内部实现                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. ST测试职责边界

### 2.1 应该测试的内容

| 测试内容 | 说明 |
|---------|------|
| NPUWorker与AscendModelRunner协作 | Worker调度到ModelRunner执行 |
| AscendScheduler与NPUWorker协作 | 调度流程正确性 |
| AscendAttention与AscendAttentionMask协作 | Attention组件协作 |
| AscendQuantizer与模型加载协作 | 量化配置集成 |
| NPUCommunicator通信协作 | 分布式通信正确性 |
| AscendSampler与AscendRejectionSampler协作 | 采样流程正确性 |

### 2.2 不应该测试的内容

| 不测试内容 | 说明 | 归属 |
|-----------|------|------|
| vllm.Scheduler调度算法 | vllm核心逻辑 | vllm上游测试 |
| vllm.Worker核心实现 | vllm核心组件 | vllm上游测试 |
| vllm.AttentionBackend核心逻辑 | vllm核心逻辑 | vllm上游测试 |
| 完整推理流程 | 需要真实vllm环境 | E2E测试 |
| 插件注入vllm流程 | 需要真实vllm集成 | E2E测试 |

## 3. 接口兼容性验证

ST测试职责包括验证插件组件实现了vllm接口规范：

```python
from tests.st.utils.mock_utils import (
    verify_plugin_interface_compatibility,
    VLLM_WORKER_SPEC
)

def test_npu_worker_interface_compatibility():
    """验证NPUWorker实现了vllm.Worker接口
    
    这是ST测试职责，因为：
    - 验证的是插件组件自身的接口实现
    - 不依赖vllm运行时环境
    - 使用spec定义接口规范
    """
    npu_worker = create_npu_worker(mock_config)
    verify_plugin_interface_compatibility(npu_worker, VLLM_WORKER_SPEC)
```

### 3.1 为什么接口兼容性验证属于ST职责

| 理由 | 说明 |
|------|------|
| 验证插件组件自身 | 检查插件是否实现了接口 |
| 不依赖vllm运行时 | 使用spec静态验证，无需真实vllm |
| 插件质量保障 | 确保插件与vllm版本兼容 |

## 4. 职责边界判断规则

```
判断流程：
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  测试是否依赖真实vllm？                                                       │
│  ├─ 是 → E2E测试职责                                                         │
│  └─ 否 → 继续判断                                                            │
│                                                                             │
│  测试是否验证完整推理流程？                                                    │
│  ├─ 是 → E2E测试职责                                                         │
│  └─ 否 → 继续判断                                                            │
│                                                                             │
│  测试是否只涉及插件组件？                                                      │
│  ├─ 是 → ST测试职责                                                          │
│  └─ 否 → 继续判断                                                            │
│                                                                             │
│  测试是否验证静态接口规范？                                                    │
│  ├─ 是 → ST测试职责（接口兼容性）                                              │
│  └─ 否 → 可能是UT测试职责                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. 测试用例示例

### 5.1 正确的ST测试用例

```python
class Test_NPUWorker_AscendModelRunner_Integration(TestSTBase):
    """测试插件内部：NPUWorker与AscendModelRunner协作
    
    这是ST测试职责，因为：
    - 测试的是插件内部模块间协作
    - 不依赖真实vllm环境
    - 不测试vllm.Worker核心逻辑
    """
    
    def test_worker_schedule_to_model_runner(self):
        """验证NPUWorker调度请求到AscendModelRunner执行
        
        验证：Worker正确传递参数到ModelRunner
        场景：正常执行流程
        预期结果：ModelRunner被正确调用
        执行模式：CPU Mock
        """
        with STRunner("worker", mock_config) as runner:
            result = runner.worker.execute_model(input_batch)
            runner.model_runner.execute.assert_called_once()
```

### 5.2 错误的ST测试用例（越界到E2E）

```python
# ❌ 错误示例：这是E2E测试职责
class Test_Vllm_Plugin_Integration:
    """测试插件注入vllm后的完整推理
    
    这是E2E测试职责，因为：
    - 需要真实vllm环境
    - 验证完整推理流程
    - 不应该在tests/st/目录
    """
    
    def test_plugin_injection(self):
        from vllm import LLM
        llm = LLM(model="Qwen/Qwen2.5-0.5B")
        ...
```

## 6. 参考设计文档

详细职责边界定义见：
- openspec/changes/add-developer-test-suite/design.md 第26节
- tests/st/utils/mock_utils.py VLLM接口spec定义