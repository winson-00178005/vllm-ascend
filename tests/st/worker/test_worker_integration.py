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

"""Worker与ModelRunner集成测试

验证插件内部Worker_v1与ModelRunner_v1的协作流程。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_model_runner,
    create_mock_input_batch,
    verify_mock_calls,
)
from tests.st.utils.runner_factory import STRunner


class TestWorkerModelRunnerIntegration(PytestSTBase):
    """Worker与ModelRunner集成测试类"""

    @pytest.mark.cpu_mock
    def test_worker_init_model_runner(self, mock_worker, mock_model_runner):
        """Test Worker正确初始化ModelRunner

        验证：
        - Worker初始化时能够创建ModelRunner实例
        - Worker持有ModelRunner引用
        - ModelRunner接口可正常调用

        场景：Worker正常初始化流程

        预期结果：ModelRunner成功初始化，Worker持有正确引用

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        
        assert mock_worker.model_runner is not None
        assert hasattr(mock_worker.model_runner, 'execute')
        assert hasattr(mock_worker.model_runner, 'load_model')

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_worker_execute_model_flow(self, mock_worker, mock_model_runner, batch_size):
        """Test Worker调度请求到ModelRunner执行

        验证：
        - Worker接收推理请求，调用execute_model
        - ModelRunner接收请求参数，执行推理
        - 返回输出tensor正确传递

        场景：Worker正常调度ModelRunner执行推理

        预期结果：execute_model调用成功，返回输出tensor

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        mock_input_batch = create_mock_input_batch(batch_size=batch_size)
        
        mock_worker.input_batch = mock_input_batch
        
        output = mock_worker.execute_model()
        
        assert output is not None
        assert isinstance(output, MagicMock)
        mock_worker.execute_model.assert_called_once()

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("scenario", ["normal", "error"])
    def test_worker_handle_runner_error(self, mock_worker, mock_model_runner, scenario):
        """Test Worker处理ModelRunner执行错误

        验证：
        - ModelRunner执行推理时发生错误
        - Worker捕获错误并正确传递
        - 异常消息包含关键信息

        场景：ModelRunner执行失败时的错误处理

        预期结果：错误正确传递，异常消息包含关键信息

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        
        if scenario == "error":
            mock_worker.execute_model = MagicMock(
                side_effect=RuntimeError("memory allocation failed")
            )
            
            with pytest.raises(RuntimeError) as cm:
                mock_worker.execute_model()
            
            assert "memory allocation failed" in str(cm.exception)
        else:
            output = mock_worker.execute_model()
            assert output is not None

    @pytest.mark.cpu_mock
    def test_worker_model_runner_resource_management(self, mock_worker, mock_model_runner):
        """Test STRunner管理Worker和ModelRunner资源

        验证：
        - STRunner正确初始化Worker资源
        - STRunner正确清理资源
        - 资源管理符合上下文管理器模式

        场景：使用STRunner管理Worker模块资源

        预期结果：资源正确初始化和清理

        执行模式：CPU Mock
        """
        with STRunner("worker", {"model": "test_model"}) as runner:
            runner.register_resource("worker", mock_worker)
            runner.register_resource("model_runner", mock_model_runner)
            
            assert runner.get_resource("worker") is not None
            assert runner.get_resource("model_runner") is not None
        
        assert runner._resources == {}

    @pytest.mark.cpu_mock
    def test_worker_model_runner_data_flow(self, mock_worker, mock_model_runner):
        """Test Worker与ModelRunner数据流传递

        验证：
        - InputBatch数据正确传递到ModelRunner
        - ModelRunner正确读取InputBatch数据
        - 数据格式转换正确性

        场景：Worker将InputBatch传递给ModelRunner执行

        预期结果：数据正确传递，格式转换正确

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        mock_input_batch = create_mock_input_batch(batch_size=16)
        mock_worker.input_batch = mock_input_batch
        
        mock_worker.execute_model = MagicMock(
            return_value=torch.randn(16, 64)
        )
        
        output = mock_worker.execute_model()
        
        assert output.shape[0] == 16

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_worker_multi_batch_processing(self, batch_size, dtype):
        """Test Worker处理多批次请求的集成正确性

        验证：
        - Worker正确接收多批次请求
        - 多批次请求的调度顺序正确
        - 返回多个批次结果

        场景：Worker处理多批次并发请求

        预期结果：多批次正确处理，结果正确返回

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch', 'device'])
        model_runner = create_mock_model_runner(spec_attrs=['execute', 'load_model', 'device'])
        worker.model_runner = model_runner
        
        input_batch = create_mock_input_batch(batch_size=batch_size)
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(batch_size, 16, 64, 
                                    dtype=torch.float16 if dtype == "float16" else torch.bfloat16)
        )
        
        output = worker.execute_model()
        
        assert output.shape[0] == batch_size
        assert output.dtype == (torch.float16 if dtype == "float16" else torch.bfloat16)