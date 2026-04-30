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

"""Worker完整测试模式示例

本文件演示完整的ST测试编写模式，包括：
1. Docstring强制模板（验证、场景、预期结果、执行模式）
2. 异常场景测试（assertRaises验证错误传递）
3. Mock验证策略（verify_mock_calls验证调用次数和参数）

参考：tests/st/ST_TEST_WRITING_GUIDE.md
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_model_runner,
    verify_mock_calls,
)


class TestWorkerDocstringPattern(PytestSTBase):
    """Worker Docstring模式测试类 - 演示完整Docstring模板"""

    @pytest.mark.cpu_mock
    def test_worker_docstring_example_normal(self):
        """Test Worker正常执行流程 - Docstring模式示例

        验证：
        - Worker正确初始化并持有ModelRunner引用
        - Worker execute_model方法正确调用
        - execute_model返回正确形状的tensor

        场景：Worker正常初始化和执行流程

        预期结果：Worker正确初始化，execute_model返回预期tensor

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'model_runner'])
        model_runner = create_mock_model_runner(spec_attrs=['execute'])
        worker.model_runner = model_runner
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128, dtype=torch.float16)
        )
        
        result = worker.execute_model()
        
        assert result is not None
        assert result.shape == (16, 128)
        assert result.dtype == torch.float16

    @pytest.mark.cpu_mock
    def test_worker_docstring_example_batch(self):
        """Test Worker处理不同batch_size - Docstring模式示例

        验证：
        - Worker正确处理batch_size=1的小批次
        - Worker正确处理batch_size=16的中等批次
        - Worker正确处理batch_size=32的大批次

        场景：不同batch_size下的Worker执行

        预期结果：所有batch_size正确处理

        执行模式：CPU Mock
        """
        for batch_size in [1, 16, 32]:
            worker = create_mock_worker(spec_attrs=['execute_model'])
            worker.execute_model = MagicMock(
                return_value=torch.randn(batch_size, 128)
            )
            
            result = worker.execute_model()
            assert result.shape[0] == batch_size


class TestWorkerExceptionPattern(PytestSTBase):
    """Worker异常测试模式类 - 演示assertRaises模式"""

    @pytest.mark.cpu_mock
    def test_worker_memory_error_propagation(self):
        """Test Worker正确传递内存分配错误

        验证：
        - ModelRunner内存分配失败时正确抛出RuntimeError
        - Worker捕获错误并正确传递
        - 异常消息包含"memory allocation"关键信息

        场景：ModelRunner执行推理时发生内存不足错误

        预期结果：错误正确传递，异常消息包含关键信息

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'model_runner'])
        
        worker.execute_model = MagicMock(
            side_effect=RuntimeError("memory allocation failed: NPU OOM")
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_model()
        
        assert "memory allocation" in str(cm.exception)
        assert "OOM" in str(cm.exception)

    @pytest.mark.cpu_mock
    def test_worker_device_error_propagation(self):
        """Test Worker正确传递设备错误

        验证：
        - NPU设备不可用时正确抛出RuntimeError
        - Worker捕获错误并正确传递
        - 异常消息包含"device"关键信息

        场景：NPU设备初始化失败时的错误处理

        预期结果：设备错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        worker.execute_model = MagicMock(
            side_effect=RuntimeError("device not available: NPU:0 offline")
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_model()
        
        assert "device" in str(cm.exception)
        assert "offline" in str(cm.exception)

    @pytest.mark.cpu_mock
    def test_worker_config_error_propagation(self):
        """Test Worker正确传递配置错误

        验证：
        - 配置参数无效时正确抛出ValueError
        - Worker捕获错误并正确传递
        - 异常消息包含配置参数名称

        场景：配置参数无效时的错误处理

        预期结果：配置错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['initialize_model'])
        
        worker.initialize_model = MagicMock(
            side_effect=ValueError("invalid config: dtype must be float16 or bfloat16")
        )
        
        with pytest.raises(ValueError) as cm:
            worker.initialize_model()
        
        assert "invalid config" in str(cm.exception)
        assert "dtype" in str(cm.exception)

    @pytest.mark.cpu_mock
    def test_worker_runtime_error_with_context(self):
        """Test Worker RuntimeError包含完整上下文信息

        验证：
        - Worker执行失败时抛出RuntimeError
        - 异常消息包含batch_size上下文
        - 异常消息包含执行阶段信息

        场景：Worker执行失败时的完整错误上下文

        预期结果：异常消息包含完整上下文信息

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        error_msg = "execution failed at batch_size=16, stage=prefill, error=timeout"
        worker.execute_model = MagicMock(
            side_effect=RuntimeError(error_msg)
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_model()
        
        assert "batch_size=16" in str(cm.exception)
        assert "stage=prefill" in str(cm.exception)
        assert "error=timeout" in str(cm.exception)


class TestWorkerMockVerificationPattern(PytestSTBase):
    """Worker Mock验证模式类 - 演示verify_mock_calls模式"""

    @pytest.mark.cpu_mock
    def test_worker_verify_execute_model_calls(self):
        """Test Worker verify_mock_calls验证execute_model调用

        验证：
        - execute_model被调用1次
        - execute_model被正确调用
        - 调用次数验证通过

        场景：使用verify_mock_calls验证Mock调用次数

        预期结果：Mock调用次数验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        worker.execute_model()
        
        verify_mock_calls(worker.execute_model, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_worker_verify_load_model_calls(self):
        """Test Worker verify_mock_calls验证load_model调用

        验证：
        - load_model被调用1次
        - load_model调用参数正确
        - Mock验证策略正确应用

        场景：使用verify_mock_calls验证load_model调用

        预期结果：Mock调用验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['load_model'])
        worker.load_model = MagicMock()
        
        worker.load_model()
        
        verify_mock_calls(worker.load_model, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_worker_verify_multiple_calls(self):
        """Test Worker verify_mock_calls验证多次调用

        验证：
        - execute_model被调用多次
        - 每次调用返回结果正确
        - 调用次数验证准确

        场景：验证Worker多次调用场景

        预期结果：多次调用验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        for _ in range(3):
            worker.execute_model()
        
        verify_mock_calls(worker.execute_model, expected_calls=3)

    @pytest.mark.cpu_mock
    def test_worker_verify_call_with_args(self):
        """Test Worker verify_mock_calls验证调用参数

        验证：
        - execute_model被调用时参数正确
        - 参数包含scheduler_output
        - verify_mock_calls验证参数通过

        场景：验证Worker调用参数正确性

        预期结果：调用参数验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_seqs = 16
        
        worker.execute_model(scheduler_output)
        
        verify_mock_calls(worker.execute_model, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_worker_verify_no_call(self):
        """Test Worker verify_mock_calls验证未调用

        验证：
        - 未调用的方法call_count为0
        - verify_mock_calls验证未调用通过
        - 异常路径下Mock未被调用

        场景：验证异常路径下Mock未被调用

        预期结果：未调用验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'load_model'])
        worker.execute_model = MagicMock()
        worker.load_model = MagicMock()
        
        worker.execute_model()
        
        verify_mock_calls(worker.execute_model, expected_calls=1)
        verify_mock_calls(worker.load_model, expected_calls=0)


class TestWorkerCombinedPatterns(PytestSTBase):
    """Worker组合模式类 - 演示所有模式组合使用"""

    @pytest.mark.cpu_mock
    def test_worker_combined_patterns_example(self):
        """Test Worker组合模式示例 - Docstring + Exception + Mock验证

        验证：
        - Worker正确初始化（Docstring模式）
        - Worker正确处理异常（Exception模式）
        - Worker Mock调用验证（Mock验证模式）

        场景：组合使用所有测试模式

        预期结果：所有模式验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'load_model', 'get_model']
        )
        
        worker.load_model = MagicMock()
        worker.load_model()
        verify_mock_calls(worker.load_model, expected_calls=1)
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        result = worker.execute_model()
        assert result.shape == (16, 128)
        verify_mock_calls(worker.execute_model, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_worker_exception_with_mock_verification(self):
        """Test Worker异常路径与Mock验证组合

        验证：
        - 异常路径正确触发
        - 异常消息包含关键信息
        - 其他方法未被错误调用

        场景：异常路径下的Mock验证组合

        预期结果：异常正确传递，Mock验证通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'load_model']
        )
        
        worker.execute_model = MagicMock(
            side_effect=RuntimeError("execution failed")
        )
        worker.load_model = MagicMock()
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_model()
        
        assert "execution failed" in str(cm.exception)
        verify_mock_calls(worker.load_model, expected_calls=0)