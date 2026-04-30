# 插件架构测试策略文档

本文档说明如何测试vllm-ascend插件的继承、替换、注入机制。

## 插件架构概述

vllm-ascend插件通过以下机制集成到vllm：

| 机制 | 说明 | 测试重点 |
|------|------|----------|
| 继承 | 组件继承vllm基类 | 接口兼容性验证 |
| 替换 | 组件替换vllm默认实现 | 替换逻辑正确性 |
| 注入 | 组件注入到vllm运行时 | 注入流程验证 |

## 继承测试

### 测试目标

验证插件组件正确继承vllm基类并实现所有必需接口。

### 测试方法

使用verify_plugin_interface_compatibility验证接口：

```python
from tests.st.utils.mock_utils import (
    VLLM_WORKER_SPEC,
    verify_plugin_interface_compatibility,
)

def test_worker_interface_compatibility():
    """验证NPUWorker继承vllm.Worker
    
    验证：
    - NPUWorker包含所有vllm.Worker接口方法
    - 每个接口方法可调用
    - 接口签名符合vllm规范
    
    场景：插件组件继承vllm基类
    
    预期结果：接口兼容性验证通过
    
    执行模式：CPU Mock
    """
    # 创建符合接口的Mock
    worker = create_mock_worker(spec_attrs=VLLM_WORKER_SPEC)
    
    # 验证接口兼容性
    verify_plugin_interface_compatibility(worker, VLLM_WORKER_SPEC)
```

### vllm接口Spec

| Spec常量 | 说明 | 接口方法 |
|----------|------|---------|
| VLLM_WORKER_SPEC | Worker接口 | execute_model, initialize_model, get_model, profile_run, start_worker, stop_worker |
| VLLM_SCHEDULER_SPEC | Scheduler接口 | schedule, get_num_unfinished_seqs, has_unfinished_seqs, add_seq_group, abort_seq_group |
| VLLM_ATTENTION_BACKEND_SPEC | AttentionBackend接口 | get_name, get_impl_cls, get_metadata_cls, get_state_cls, get_kv_cache_shape, swap_blocks, copy_blocks |

## 替换测试

### 测试目标

验证插件组件正确替换vllm默认实现，保持功能一致性。

### 测试方法

测试替换逻辑正确性：

```python
def test_worker_replacement_logic():
    """验证Worker替换vllm.Worker逻辑
    
    验证：
    - Worker正确初始化ModelRunner
    - Worker调度协作正确
    - Worker接口兼容
    
    场景：插件组件替换vllm默认实现
    
    预期结果：替换逻辑正确，功能一致
    
    执行模式：CPU Mock
    """
    worker = create_mock_worker(spec_attrs=['execute_model', 'model_runner'])
    model_runner = create_mock_model_runner()
    
    # Worker正确初始化ModelRunner
    worker.model_runner = model_runner
    
    # Worker调度协作
    scheduler_output = MagicMock()
    worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
    
    result = worker.execute_model(scheduler_output)
    assert result is not None
```

### 替换场景测试

| 场景 | 测试内容 |
|------|---------|
| Worker替换 | Worker初始化ModelRunner、调度协作 |
| Scheduler替换 | Scheduler调度策略、优先级调度 |
| AttentionBackend替换 | Attention与Mask协作、KV cache管理 |
| Quantizer替换 | Quantizer与模型加载协作 |

## 注入测试

### 测试目标

验证插件组件正确注入到vllm运行时，不影响vllm核心逻辑。

### 测试方法

测试组件注入流程：

```python
def test_worker_injection_flow():
    """验证Worker注入到vllm运行时
    
    验证：
    - Worker正确接收vllm配置
    - Worker正确初始化
    - Worker不影响vllm核心流程
    
    场景：插件组件注入vllm运行时
    
    预期结果：注入正确，vllm核心流程不受影响
    
    执行模式：CPU Mock
    """
    from tests.st.utils.config_factory import create_vllm_config
    
    # 创建vllm配置
    vllm_config = create_vllm_config(model="Qwen/Qwen2.5-0.5B")
    
    # Worker接收配置
    worker = create_mock_worker()
    worker.vllm_config = vllm_config
    
    # Worker不影响vllm核心流程
    assert worker.vllm_config.model == "Qwen/Qwen2.5-0.5B"
```

## 测试策略矩阵

| 架构机制 | 测试类型 | 测试方法 | 负责测试层 |
|----------|---------|---------|-----------|
| 继承 | 接口兼容性 | verify_plugin_interface_compatibility | ST |
| 替换 | 替换逻辑 | 模块协作测试 | ST |
| 注入 | 注入流程 | 配置传递测试 | ST/E2E |
| 功能一致性 | 行为验证 | 真实NPU测试 | E2E |

## 测试边界

### ST测试职责

ST测试负责验证：
- 插件组件继承vllm接口
- 插件组件替换逻辑
- 插件组件内部协作

### 不属于ST职责

以下不属于ST测试范围：
- vllm核心逻辑功能
- 插件注入vllm完整流程（E2E职责）
- vllm配置管理核心逻辑

## 参考

- tests/st/PLUGIN_TEST_RESPONSIBILITY.md - 测试职责边界
- tests/st/utils/mock_utils.py - vllm接口Spec定义
- tests/st/ST_TEST_WRITING_GUIDE.md - 测试编写规范