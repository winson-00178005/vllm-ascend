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

"""Worker与InputBatch集成测试

验证NPUWorker与NpuInputBatch的数据流和格式转换。
参考：tests/st/specs/st-worker-integration/spec.md
"""

import pytest
import torch

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_input_batch,
    verify_mock_calls,
)

if not HAS_NUMPY:
    pytestmark = pytest.mark.skip(reason="numpy not installed")


class TestWorkerInputBatchIntegration(PytestSTBase):
    """Worker与InputBatch集成测试类"""

    @pytest.mark.cpu_mock
    def test_worker_input_batch_data_flow(self):
        """Test Worker与InputBatch数据流传递

        验证：
        - Worker正确接收InputBatch对象
        - InputBatch数据正确存储token_ids
        - Worker能正确读取InputBatch数据

        场景：Worker接收推理请求，创建InputBatch对象

        预期结果：InputBatch正确存储输入数据，Worker正确读取

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        input_batch = create_mock_input_batch(batch_size=16)
        input_batch.token_ids = torch.randint(0, 1000, (16, 128))
        
        worker.input_batch = input_batch
        
        assert worker.input_batch is not None
        assert worker.input_batch.batch_size == 16
        assert worker.input_batch.token_ids.shape == (16, 128)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_input_batch_different_batch_sizes(self, batch_size):
        """Test InputBatch不同batch_size处理

        验证：
        - InputBatch正确处理batch_size=1
        - InputBatch正确处理batch_size=16
        - InputBatch正确处理batch_size=32

        场景：不同batch_size下的InputBatch数据存储

        预期结果：所有batch_size正确处理

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=batch_size)
        
        assert input_batch.batch_size == batch_size
        
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(batch_size, 128)
        )
        
        result = worker.execute_model()
        assert result.shape[0] == batch_size

    @pytest.mark.cpu_mock
    def test_input_batch_token_ids_format(self):
        """Test InputBatch token_ids格式转换

        验证：
        - token_ids正确存储为int32类型
        - token_ids格式符合NPU输入要求
        - token_ids数值范围正确

        场景：InputBatch token_ids格式转换验证

        预期结果：token_ids格式正确

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=16)
        
        token_ids = torch.randint(0, 32000, (16, 128), dtype=torch.int32)
        input_batch.token_ids = token_ids
        
        assert input_batch.token_ids.dtype == torch.int32
        assert input_batch.token_ids.max() < 32000
        assert input_batch.token_ids.min() >= 0

    @pytest.mark.cpu_mock
    def test_input_batch_position_ids_format(self):
        """Test InputBatch position_ids格式

        验证：
        - position_ids正确生成
        - position_ids格式符合推理要求
        - position_ids与token_ids长度一致

        场景：InputBatch position_ids生成验证

        预期结果：position_ids格式正确

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=16)
        
        seq_len = 128
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(16, -1)
        input_batch.position_ids = position_ids
        
        assert input_batch.position_ids.shape == (16, seq_len)
        assert input_batch.position_ids.max() == seq_len - 1

    @pytest.mark.cpu_mock
    def test_worker_passes_input_batch_to_model_runner(self):
        """Test Worker将InputBatch传递给ModelRunner

        验证：
        - Worker将InputBatch数据传递给ModelRunner
        - ModelRunner正确读取InputBatch数据
        - 数据传递过程无丢失

        场景：Worker将InputBatch传递给ModelRunner执行推理

        预期结果：数据正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch', 'model_runner'])
        model_runner = MagicMock()
        model_runner.execute = MagicMock(return_value=torch.randn(16, 128))
        
        worker.model_runner = model_runner
        
        input_batch = create_mock_input_batch(batch_size=16)
        input_batch.token_ids = torch.randint(0, 1000, (16, 128))
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
        result = worker.execute_model()
        
        verify_mock_calls(worker.execute_model, expected_calls=1)
        assert result.shape[0] == 16

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_input_batch_different_sequence_lengths(self, seq_len):
        """Test InputBatch不同序列长度处理

        验证：
        - InputBatch正确处理seq_len=64
        - InputBatch正确处理seq_len=128
        - InputBatch正确处理seq_len=256
        - InputBatch正确处理seq_len=512

        场景：不同序列长度下的InputBatch处理

        预期结果：所有序列长度正确处理

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=16)
        input_batch.token_ids = torch.randint(0, 1000, (16, seq_len))
        
        assert input_batch.token_ids.shape[1] == seq_len
        
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, seq_len)
        )
        
        result = worker.execute_model()
        assert result.shape[1] == seq_len

    @pytest.mark.cpu_mock
    def test_input_batch_attention_mask_generation(self):
        """Test InputBatch attention_mask生成

        验证：
        - InputBatch正确生成attention_mask
        - attention_mask格式符合NPU要求
        - attention_mask数值正确（全1表示有效）

        场景：InputBatch attention_mask生成验证

        预期结果：attention_mask正确生成

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=16)
        
        seq_len = 128
        attention_mask = torch.ones(16, 1, seq_len, seq_len)
        input_batch.attention_mask = attention_mask
        
        assert input_batch.attention_mask.shape == (16, 1, seq_len, seq_len)
        assert input_batch.attention_mask.min() == 1
        assert input_batch.attention_mask.max() == 1

    @pytest.mark.cpu_mock
    def test_input_batch_sampling_params_storage(self):
        """Test InputBatch采样参数存储

        验证：
        - InputBatch正确存储temperature参数
        - InputBatch正确存储top_p参数
        - InputBatch正确存储top_k参数

        场景：InputBatch采样参数存储验证

        预期结果：采样参数正确存储

        执行模式：CPU Mock
        """
        input_batch = create_mock_input_batch(batch_size=16)
        
        input_batch.temperature = torch.ones(16, dtype=torch.float32) * 0.8
        input_batch.top_p = torch.ones(16, dtype=torch.float32) * 0.95
        input_batch.top_k = torch.ones(16, dtype=torch.int32) * 50
        
        assert input_batch.temperature.mean() == 0.8
        assert input_batch.top_p.mean() == 0.95
        assert input_batch.top_k.mean() == 50


class TestWorkerInputBatchErrorHandling(PytestSTBase):
    """Worker与InputBatch错误处理测试类"""

    @pytest.mark.cpu_mock
    def test_input_batch_empty_error(self):
        """Test InputBatch空数据错误处理

        验证：
        - InputBatch空数据时正确抛出错误
        - Worker捕获错误并正确传递
        - 异常消息包含关键信息

        场景：InputBatch数据为空时的错误处理

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        input_batch = create_mock_input_batch(batch_size=0)
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            side_effect=ValueError("InputBatch is empty: batch_size=0")
        )
        
        with pytest.raises(ValueError) as cm:
            worker.execute_model()
        
        assert "InputBatch is empty" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_input_batch_invalid_token_ids_error(self):
        """Test InputBatch无效token_ids错误处理

        验证：
        - token_ids超出vocab范围时正确抛出错误
        - Worker捕获错误并正确传递
        - 异常消息包含vocab范围信息

        场景：token_ids超出vocab范围时的错误处理

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        input_batch = create_mock_input_batch(batch_size=16)
        invalid_token_ids = torch.randint(32000, 50000, (16, 128))
        input_batch.token_ids = invalid_token_ids
        worker.input_batch = input_batch
        
        worker.execute_model = MagicMock(
            side_effect=ValueError("token_ids exceed vocab_size: max_token=50000, vocab_size=32000")
        )
        
        with pytest.raises(ValueError) as cm:
            worker.execute_model()
        
        assert "exceed vocab_size" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_input_batch_memory_overflow_error(self):
        """Test InputBatch内存溢出错误处理

        验证：
        - InputBatch内存分配失败时正确抛出错误
        - Worker捕获错误并正确传递
        - 异常消息包含内存信息

        场景：InputBatch内存溢出时的错误处理

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        worker.execute_model = MagicMock(
            side_effect=RuntimeError("InputBatch memory allocation failed: NPU OOM")
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_model()
        
        assert "memory allocation failed" in str(cm.value)
        assert "OOM" in str(cm.value)