# Mock工厂函数使用指南

本文档说明如何使用tests/st/utils/mock_utils.py中的Mock工厂函数。

## 核心原则

### spec限制Mock

所有Mock工厂函数支持spec限制，避免过度Mock：

```python
# 使用默认spec
worker = create_mock_worker()

# 自定义spec
worker = create_mock_worker(spec_attrs=['execute_model', 'load_model'])

# 使用vllm接口spec
from tests.st.utils.mock_utils import VLLM_WORKER_SPEC
worker = create_mock_worker(spec_attrs=VLLM_WORKER_SPEC)
```

### Mock工厂函数命名规范

所有工厂函数遵循`create_mock_*`命名：

| 函数 | 说明 | 默认spec |
|------|------|---------|
| create_mock_worker | 创建Worker Mock | ['execute_model', 'input_batch', 'device'] |
| create_mock_model_runner | 创建ModelRunner Mock | ['execute', 'load_model', 'device'] |
| create_mock_scheduler | 创建Scheduler Mock | ['schedule', 'get_num_unfinished_seqs', 'device'] |
| create_mock_attention | 创建Attention Mock | ['forward', 'get_kv_cache_shape', 'device'] |
| create_mock_attention_mask | 创建AttentionMask Mock | 无固定spec |
| create_mock_input_batch | 创建InputBatch Mock | 无固定spec |
| create_mock_distribution_env | 创建分布式环境Mock | 无固定spec |

## 使用示例

### Worker Mock

```python
from tests.st.utils.mock_utils import create_mock_worker, verify_mock_calls

# 创建Worker Mock
worker = create_mock_worker(spec_attrs=['execute_model', 'load_model'])

# 配置Mock行为
import torch
worker.execute_model = MagicMock(return_value=torch.randn(16, 128))

# 使用Mock
result = worker.execute_model()

# 验证调用
verify_mock_calls(worker.execute_model, expected_calls=1)
```

### Scheduler Mock

```python
from tests.st.utils.mock_utils import create_mock_scheduler

scheduler = create_mock_scheduler()

# 配置schedule返回值
from unittest.mock import MagicMock
scheduler.schedule = MagicMock(return_value=MagicMock(num_scheduled_tokens=1024))

output = scheduler.schedule()
assert output.num_scheduled_tokens == 1024
```

### Attention Mock

```python
from tests.st.utils.mock_utils import create_mock_attention, create_mock_attention_mask

attention = create_mock_attention()
attention_mask = create_mock_attention_mask()

# 关联Mock
attention.attention_mask = attention_mask

# 配置forward行为
import torch
attention.forward = MagicMock(return_value=torch.randn(16, 128, 4096))
```

## verify_mock_calls使用

验证Mock调用次数和参数：

```python
from tests.st.utils.mock_utils import verify_mock_calls

# 仅验证调用次数
verify_mock_calls(worker.execute_model, expected_calls=1)

# 验证调用次数和参数
verify_mock_calls(
    worker.execute_model,
    expected_calls=1,
    expected_args={'scheduler_output': mock_output}
)
```

## CPUMockEnvironment使用

CPUMockEnvironment用于CPU Mock模式：

```python
from tests.st.utils.env_factory import create_environment

# 创建CPU Mock环境
env = create_environment("cpu_mock")

# 进入环境
env.enter()

# 使用Mock执行测试
worker = create_mock_worker()
...

# 退出环境（自动清理）
env.exit()
```

## vllm接口Spec常量

定义vllm接口规范，用于验证插件兼容性：

```python
from tests.st.utils.mock_utils import (
    VLLM_WORKER_SPEC,
    VLLM_SCHEDULER_SPEC,
    VLLM_ATTENTION_BACKEND_SPEC,
    verify_plugin_interface_compatibility,
)

# 创建符合vllm接口的Mock
worker = create_mock_worker(spec_attrs=VLLM_WORKER_SPEC)

# 验证插件组件实现vllm接口
verify_plugin_interface_compatibility(worker, VLLM_WORKER_SPEC)
```

## 最佳实践

### 1. 优先使用工厂函数

避免直接创建MagicMock：

```python
# 推荐
worker = create_mock_worker()

# 不推荐
worker = MagicMock()
```

### 2. 使用spec限制

```python
# 推荐 - spec限制
worker = create_mock_worker(spec_attrs=['execute_model', 'load_model'])

# 不推荐 - 无限制，可能过度Mock
worker = MagicMock()
```

### 3. 使用验证函数

```python
# 推荐 - 使用verify_mock_calls
verify_mock_calls(worker.execute_model, expected_calls=1)

# 不推荐 - 手动验证
assert worker.execute_model.call_count == 1
```

### 4. fixture复用

在conftest.py中定义fixture复用Mock：

```python
# tests/st/worker/conftest.py
@pytest.fixture
def mock_worker():
    return create_mock_worker()

@pytest.fixture
def mock_model_runner():
    return create_mock_model_runner()
```

## 参考

- tests/st/utils/mock_utils.py - Mock工厂函数实现
- tests/st/ST_TEST_WRITING_GUIDE.md - 测试编写规范
- tests/ut/quantization/test_w8a8.py - spec限制Mock参考模式