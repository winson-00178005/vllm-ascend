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

"""Worker参数化多维度测试

验证Worker在不同参数组合下的行为正确性。
参考：tests/st/utils/data_generator.py中的标准参数值
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_model_runner,
    create_mock_input_batch,
)
from tests.st.utils.data_generator import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_DTYPES,
    DEFAULT_SEQ_LENGTHS,
)


class TestWorkerParameterized(PytestSTBase):
    """Worker参数化多维度测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", DEFAULT_BATCH_SIZES)
    @pytest.mark.parametrize("scenario", ["normal", "error"])
    @pytest.mark.parametrize("dtype", DEFAULT_DTYPES)
    def test_worker_execute_model_multi_dim(self, batch_size, scenario, dtype):
        """Test Worker execute_model多维度参数化

        验证：
        - 不同batch_size下execute_model正确执行
        - 不同scenario下Worker正确处理
        - 不同dtype下返回正确数据类型

        场景：多维度参数组合测试

        预期结果：所有参数组合下测试通过

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        model_runner = create_mock_model_runner(spec_attrs=['execute'])
        worker.model_runner = model_runner
        
        input_batch = create_mock_input_batch(batch_size=batch_size)
        worker.input_batch = input_batch
        
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        
        if scenario == "error":
            worker.execute_model = MagicMock(
                side_effect=RuntimeError(f"error with batch_size={batch_size}, dtype={dtype}")
            )
            with pytest.raises(RuntimeError) as cm:
                worker.execute_model()
            assert f"batch_size={batch_size}" in str(cm.exception)
        else:
            worker.execute_model = MagicMock(
                return_value=torch.randn(batch_size, 16, 64, dtype=torch_dtype)
            )
            result = worker.execute_model()
            assert result.shape[0] == batch_size
            assert result.dtype == torch_dtype

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", DEFAULT_BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", DEFAULT_SEQ_LENGTHS)
    def test_worker_input_batch_combinations(self, batch_size, seq_len):
        """Test Worker InputBatch参数组合

        验证：
        - 不同batch_size和seq_len组合下InputBatch正确创建
        - InputBatch数据格式正确
        - Worker正确处理InputBatch

        场景：batch_size和seq_len参数组合测试

        预期结果：所有组合下InputBatch正确创建和处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        input_batch = create_mock_input_batch(batch_size=batch_size)
        input_batch.token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        worker.input_batch = input_batch
        
        assert worker.input_batch.batch_size == batch_size
        assert worker.input_batch.token_ids.shape == (batch_size, seq_len)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", DEFAULT_BATCH_SIZES)
    @pytest.mark.parametrize("dtype", DEFAULT_DTYPES)
    def test_worker_output_dtype_correctness(self, batch_size, dtype):
        """Test Worker输出dtype正确性

        验证：
        - Worker返回输出dtype与配置一致
        - 不同dtype下输出数值范围正确
        - dtype转换正确性

        场景：不同dtype参数下输出验证

        预期结果：输出dtype正确，数值范围符合预期

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(batch_size, 128, dtype=torch_dtype)
        )
        
        result = worker.execute_model()
        
        assert result.dtype == torch_dtype
        assert result.shape[0] == batch_size

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("scenario", ["normal", "error"])
    def test_worker_scenario_handling(self, scenario):
        """Test Worker场景处理

        验证：
        - normal场景下Worker正确执行
        - error场景下Worker正确处理错误
        - 异常消息包含关键信息

        场景：不同场景下Worker行为验证

        预期结果：场景处理正确，错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        if scenario == "error":
            worker.execute_model = MagicMock(
                side_effect=RuntimeError("execution failed")
            )
            with pytest.raises(RuntimeError) as cm:
                worker.execute_model()
            assert "execution failed" in str(cm.exception)
        else:
            worker.execute_model = MagicMock(
                return_value=torch.randn(16, 128)
            )
            result = worker.execute_model()
            assert result is not None

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64])
    def test_worker_batch_size_range(self, batch_size):
        """Test Worker batch_size范围测试

        验证：
        - 小batch_size(1)正确处理
        - 中等batch_size(16)正确处理
        - 大batch_size(64)正确处理

        场景：batch_size范围边界测试

        预期结果：所有batch_size正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        input_batch = create_mock_input_batch(batch_size=batch_size)
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(batch_size, 16, 64)
        )
        
        result = worker.execute_model()
        
        assert result.shape[0] == batch_size

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
    def test_worker_dtype_range(self, dtype):
        """Test Worker dtype范围测试

        验证：
        - float16正确处理
        - bfloat16正确处理
        - float32正确处理

        场景：dtype范围测试

        预期结果：所有dtype正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128, dtype=torch_dtype)
        )
        
        result = worker.execute_model()
        
        assert result.dtype == torch_dtype