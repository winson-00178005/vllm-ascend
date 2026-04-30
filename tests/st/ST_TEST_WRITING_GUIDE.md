# ST测试编写规范

本文档定义vllm-ascend ST测试的编写规范，确保测试质量和一致性。

## 1. Docstring模板（强制）

每个测试用例必须包含完整的Docstring，格式如下：

```python
def test_worker_schedule_to_runner(self, batch_size, scenario):
    """Test Worker调度请求到ModelRunner执行
    
    验证：
    - Worker正确传递请求参数到ModelRunner
    - ModelRunner执行推理并返回输出tensor
    - 异常情况下Worker正确处理错误
    
    场景：正常执行流程和异常处理
    
    预期结果：执行成功或错误正确传递
    
    执行模式：CPU Mock / NPU Real
    """
    ...
```

Docstring必须包含四个部分：
1. **验证**：列出验证点
2. **场景**：描述测试场景
3. **预期结果**：期望的输出或行为
4. **执行模式**：适用的执行模式

## 2. 参数化指南

### 2.1 多维度参数化

使用@pytest.mark.parametrize进行多维度测试：

```python
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("scenario", ["normal", "error"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_worker_model_runner_integration(self, batch_size, scenario, dtype):
    """Test Worker与ModelRunner集成"""
    ...
```

### 2.2 参数化命名

参数名称应清晰表达含义：
- batch_size：批次大小
- scenario：场景类型
- dtype：数据类型
- seq_len：序列长度

### 2.3 参数值选择

参考tests/st/utils/data_generator.py中的标准值：
- batch_size: [1, 16, 32]
- dtype: ["float16", "bfloat16"]
- seq_len: [128, 512, 1024]

## 3. 异常测试模式

### 3.1 assertRaises模式

```python
def test_worker_handle_runner_error(self):
    """Test Worker处理ModelRunner执行失败
    
    验证：异常正确传递，资源正确清理
    """
    with self.assertRaises(RuntimeError) as cm:
        worker.execute_model(error_input)
    
    assert "memory allocation failed" in str(cm.exception)
    assert worker.resources_released
```

### 3.2 异常消息验证

验证异常消息包含关键信息：
```python
assert "error_msg" in str(cm.exception)
```

## 4. Mock使用规范

### 4.1 使用Mock工厂函数

优先使用tests/st/utils/mock_utils.py中的工厂函数：

```python
from tests.st.utils.mock_utils import create_mock_worker

def test_worker_integration(self):
    worker = create_mock_worker()
    ...
```

### 4.2 spec限制Mock

使用spec限制避免过度Mock：

```python
worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
```

### 4.3 Mock验证

使用verify_mock_calls验证Mock调用：

```python
from tests.st.utils.mock_utils import verify_mock_calls

verify_mock_calls(worker.execute_model, expected_calls=1)
```

## 5. 双模式测试标记

### 5.1 标记执行模式

使用pytest marker标记测试执行模式：

```python
@pytest.mark.cpu_mock
def test_worker_schedule_cpu():
    """CPU Mock模式测试"""
    ...

@pytest.mark.npu_real
def test_worker_schedule_npu():
    """NPU真实模式测试"""
    ...

@pytest.mark.npu_precision
def test_attention_precision():
    """NPU精度测试"""
    ...
```

### 5.2 参数化双模式

```python
@pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
def test_worker_execute_model(exec_mode):
    """参数化双模式测试"""
    ...
```

## 6. 测试命名规范

测试类命名：`Test_<模块集成特性>_<场景描述>`

测试方法命名：`test_<具体测试场景>`

示例：
- `Test_WorkerModelRunner_Integration`
- `test_worker_schedule_to_runner`
- `test_worker_handle_runner_error`

## 7. 参考

参考现有优秀测试模式：
- tests/ut/quantization/test_w8a8.py：spec限制Mock
- tests/ut/sample/test_rejection_sampler.py：全局常量定义
- tests/e2e/singlecard/test_offline_inference.py：多维度参数化