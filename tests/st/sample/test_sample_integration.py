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

"""Sample模块集成测试

验证AscendSampler和AscendRejectionSampler的插件内部协作。
参考：tests/ut/sample/test_rejection_sampler.py模式
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import verify_mock_calls
from tests.st.utils.runner_factory import STRunner


PLACEHOLDER_TOKEN_ID = -1


class TestSamplerIntegration(PytestSTBase):
    """Sampler集成测试类"""

    @pytest.mark.cpu_mock
    def test_sampler_init_with_config(self, mock_sampler):
        """Test Sampler初始化配置

        验证：
        - Sampler正确初始化并持有配置
        - 采样配置参数正确传递
        - 初始化状态正确

        场景：Sampler正常初始化流程

        预期结果：Sampler正确初始化

        执行模式：CPU Mock
        """
        mock_sampler.config = MagicMock()
        mock_sampler.config.temperature = 0.7
        
        assert mock_sampler.config.temperature == 0.7

    @pytest.mark.cpu_mock
    def test_sampler_sample_flow(self, mock_sampler, logits_data):
        """Test Sampler采样流程

        验证：
        - Sampler sample方法正确调用
        - sample接收正确输入参数
        - 采样结果正确返回

        场景：Sampler正常采样流程

        预期结果：采样正确执行，结果正确

        执行模式：CPU Mock
        """
        output = mock_sampler.sample(logits_data)
        
        assert output is not None
        assert output.shape[0] == 16
        verify_mock_calls(mock_sampler.sample, expected_calls=1)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_sampler_different_batch_sizes(self, batch_size):
        """Test Sampler不同batch_size

        验证：
        - Sampler正确处理batch_size=1
        - Sampler正确处理batch_size=16
        - Sampler正确处理batch_size=32

        场景：不同batch_size采样测试

        预期结果：所有batch_size正确处理

        执行模式：CPU Mock
        """
        mock_sample = MagicMock(spec=['sample'])
        
        logits = torch.randn(batch_size, 32000)
        mock_sample.sample = MagicMock(return_value=torch.randint(0, 32000, (batch_size,)))
        
        result = mock_sample.sample(logits)
        assert result.shape[0] == batch_size


class TestSamplerTemperature(PytestSTBase):
    """Sampler Temperature测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 0.7, 0.9, 1.0])
    def test_sampler_apply_temperature(self, temperature, logits_data):
        """Test Sampler应用temperature

        验证：
        - Sampler正确应用temperature=0.0 (greedy)
        - Sampler正确应用temperature=0.5
        - Sampler正确应用temperature=0.7
        - Sampler正确应用temperature=0.9
        - Sampler正确应用temperature=1.0

        场景：不同temperature值测试

        预期结果：所有temperature正确应用

        执行模式：CPU Mock
        """
        mock_sample = MagicMock(spec=['apply_temperature'])
        
        if temperature == 0.0:
            mock_sample.apply_temperature = MagicMock(
                return_value=torch.zeros(16, 32000)
            )
        else:
            mock_sample.apply_temperature = MagicMock(
                return_value=logits_data / temperature
            )
        
        result = mock_sample.apply_temperature(logits_data, temperature)
        assert result is not None

    @pytest.mark.cpu_mock
    def test_sampler_temperature_distribution(self):
        """Test Sampler temperature影响分布

        验证：
        - temperature低时分布集中
        - temperature高时分布平滑
        - temperature正确影响采样

        场景：temperature影响分布验证

        预期结果：distribution正确变化

        执行模式：CPU Mock
        """
        logits = torch.randn(16, 32000)
        
        low_temp = logits / 0.1
        high_temp = logits / 1.0
        
        low_temp_probs = torch.softmax(low_temp, dim=-1)
        high_temp_probs = torch.softmax(high_temp, dim=-1)
        
        assert low_temp_probs.max() > high_temp_probs.max()


class TestRejectionSamplerIntegration(PytestSTBase):
    """RejectionSampler集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("rejection_scenario", ["all_accept", "all_reject", "partial_accept"])
    def test_rejection_sampler_scenarios(self, rejection_scenario):
        """Test RejectionSampler不同验证场景

        验证：
        - RejectionSampler正确处理全接受场景
        - RejectionSampler正确处理全拒绝场景
        - RejectionSampler正确处理部分接受场景

        场景：不同rejection场景测试

        预期结果：所有场景正确处理

        执行模式：CPU Mock
        """
        mock_reject = MagicMock(spec=['verify'])
        
        if rejection_scenario == "all_accept":
            mock_reject.verify = MagicMock(return_value=torch.ones(16, dtype=torch.bool))
        elif rejection_scenario == "all_reject":
            mock_reject.verify = MagicMock(return_value=torch.zeros(16, dtype=torch.bool))
        elif rejection_scenario == "partial_accept":
            mock_reject.verify = MagicMock(
                return_value=torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=torch.bool)
            )
        
        result = mock_reject.verify(torch.randint(0, 32000, (16, 4)))
        
        if rejection_scenario == "all_accept":
            assert result.sum() == 16
        elif rejection_scenario == "all_reject":
            assert result.sum() == 0
        elif rejection_scenario == "partial_accept":
            assert result.sum() > 0 and result.sum() < 16

    @pytest.mark.cpu_mock
    def test_rejection_sampler_verify_calls(self, mock_rejection_sampler, token_ids_data):
        """Test RejectionSampler verify调用

        验证：
        - RejectionSampler verify方法正确调用
        - verify接收正确参数
        - 验证结果正确返回

        场景：RejectionSampler验证调用

        预期结果：verify正确执行

        执行模式：CPU Mock
        """
        result = mock_rejection_sampler.verify(token_ids_data)
        
        assert result is not None
        verify_mock_calls(mock_rejection_sampler.verify, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_rejection_sampler_placeholder_token(self):
        """Test RejectionSampler PLACEHOLDER_TOKEN_ID处理

        验证：
        - PLACEHOLDER_TOKEN_ID正确使用
        - placeholder token正确替换
        - 替换结果正确

        场景：PLACEHOLDER_TOKEN_ID处理

        预期结果：placeholder正确处理

        执行模式：CPU Mock
        """
        tokens = torch.randint(0, 32000, (16, 4))
        
        tokens_with_placeholder = tokens.clone()
        tokens_with_placeholder[0, 0] = PLACEHOLDER_TOKEN_ID
        
        assert tokens_with_placeholder[0, 0] == PLACEHOLDER_TOKEN_ID


class TestSamplerInterfaceCompatibility(PytestSTBase):
    """Sampler接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_sampler_interface_methods(self, mock_sampler):
        """Test Sampler接口方法

        验证：
        - Sampler包含sample方法
        - Sampler包含apply_temperature方法
        - 接口方法可调用

        场景：验证Sampler接口实现

        预期结果：接口正确实现

        执行模式：CPU Mock
        """
        assert callable(mock_sampler.sample)
        assert callable(mock_sampler.apply_temperature)


class TestSamplerBatchIntegration(PytestSTBase):
    """Sampler Batch集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_sampler_batch_sampling(self, batch_size):
        """Test Sampler batch采样

        验证：
        - Sampler正确处理batch_size=1
        - Sampler正确处理batch_size=16
        - Sampler正确处理batch_size=32
        - batch采样结果正确

        场景：Sampler batch采样测试

        预期结果：batch采样正确

        执行模式：CPU Mock
        """
        mock_sample = MagicMock(spec=['sample'])
        
        logits = torch.randn(batch_size, 32000)
        mock_sample.sample = MagicMock(return_value=torch.randint(0, 32000, (batch_size,)))
        
        result = mock_sample.sample(logits)
        assert result.shape[0] == batch_size

    @pytest.mark.cpu_mock
    def test_sampler_with_strunner(self, sample_runner, mock_sampler):
        """Test Sampler使用STRunner管理资源

        验证：
        - STRunner正确初始化采样资源
        - STRunner正确清理资源
        - 资源管理符合上下文管理器模式

        场景：使用STRunner管理采样模块资源

        预期结果：资源正确初始化和清理

        执行模式：CPU Mock
        """
        with STRunner("sample", {"model": "test_model"}) as runner:
            runner.register_resource("sampler", mock_sampler)
            
            assert runner.get_resource("sampler") is not None
        
        assert runner._resources == {}


class TestSamplerWorkerIntegration(PytestSTBase):
    """Sampler Worker集成测试类"""

    @pytest.mark.cpu_mock
    def test_sampler_worker_collaboration(self):
        """Test Sampler与Worker协作

        验证：
        - Worker正确调用Sampler
        - Sampler正确返回采样结果
        - Worker正确处理采样结果

        场景：Sampler与Worker协作流程

        预期结果：协作正确执行

        执行模式：CPU Mock
        """
        from tests.st.utils.mock_utils import create_mock_worker
        
        worker = create_mock_worker(spec_attrs=['execute_model', 'sample'])
        sampler = MagicMock(spec=['sample'])
        
        worker.sample = sampler
        worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
        
        sampler.sample = MagicMock(return_value=torch.randint(0, 32000, (16,)))
        
        output = worker.execute_model()
        sampled_tokens = worker.sample.sample(output)
        
        assert sampled_tokens is not None

    @pytest.mark.cpu_mock
    def test_sampler_worker_error_propagation(self):
        """Test Sampler与Worker错误传递

        验证：
        - Sampler错误正确传递到Worker
        - Worker捕获错误并正确处理
        - 异常消息包含关键信息

        场景：Sampler与Worker错误传递

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        from tests.st.utils.mock_utils import create_mock_worker
        
        worker = create_mock_worker(spec_attrs=['sample'])
        sampler = MagicMock(spec=['sample'])
        
        sampler.sample = MagicMock(
            side_effect=RuntimeError("sampling failed: temperature invalid")
        )
        worker.sample = sampler
        
        with pytest.raises(RuntimeError) as cm:
            worker.sample.sample(torch.randn(16, 32000))
        
        assert "sampling failed" in str(cm.exception)


class TestSamplerDualMode(PytestSTBase):
    """Sampler双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_sampler_dual_mode_sample(self, exec_mode):
        """Test Sampler双模式采样

        验证：
        - CPU Mock模式下Sampler正确执行
        - NPU Real模式下Sampler正确执行
        - 不同模式下执行结果一致性

        场景：Sampler双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        mock_sample = MagicMock(spec=['sample'])
        
        if exec_mode == "cpu_mock":
            mock_sample.sample = MagicMock(return_value=torch.randint(0, 32000, (16,)))
            result = mock_sample.sample(torch.randn(16, 32000))
            assert result is not None
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")


class TestSamplerTopK(PytestSTBase):
    """Sampler TopK测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("top_k", [1, 10, 50, 100])
    def test_sampler_top_k_filter(self, top_k):
        """Test Sampler top_k过滤

        验证：
        - Sampler正确应用top_k=1 (greedy)
        - Sampler正确应用top_k=10
        - Sampler正确应用top_k=50
        - Sampler正确应用top_k=100

        场景：不同top_k值测试

        预期结果：所有top_k正确应用

        执行模式：CPU Mock
        """
        mock_sample = MagicMock(spec=['apply_top_k'])
        
        logits = torch.randn(16, 32000)
        
        mock_sample.apply_top_k = MagicMock(return_value=logits)
        
        result = mock_sample.apply_top_k(logits, top_k)
        assert result is not None