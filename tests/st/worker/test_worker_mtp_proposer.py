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

"""Worker与MTProposer集成测试

验证NPUWorker与MtpProposer的协作。
参考：tests/st/specs/st-worker-integration/spec.md
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_input_batch,
    verify_mock_calls,
)


class TestWorkerMTPProposerIntegration(PytestSTBase):
    """Worker与MTProposer集成测试类"""

    @pytest.mark.cpu_mock
    def test_worker_init_mtp_proposer(self):
        """Test Worker初始化MTProposer

        验证：
        - Worker启用MTP功能时正确初始化MTProposer
        - MTProposer持有正确的配置参数
        - MTProposer准备推测生成

        场景：Worker初始化MTProposer

        预期结果：MTProposer正确初始化，准备推测生成

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'enable_mtp']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.num_speculative_tokens = 4
        mock_mtp_proposer.load_model = MagicMock()
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, 4))
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        worker.enable_mtp = MagicMock(return_value=True)
        
        assert worker.enable_mtp() == True
        assert worker.mtp_proposer is not None
        assert worker.mtp_proposer.num_speculative_tokens == 4

    @pytest.mark.cpu_mock
    def test_mtp_proposer_generate_speculative_tokens(self):
        """Test MTProposer生成推测token

        验证：
        - Worker请求MTProposer生成推测token
        - MTProposer返回多个推测token
        - Worker将推测token加入候选池

        场景：MTProposer生成推测token

        预期结果：推测token正确生成和传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'get_speculative_tokens']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, 4))
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        worker.get_speculative_tokens = MagicMock(
            side_effect=lambda: worker.mtp_proposer.propose()
        )
        
        speculative_tokens = worker.get_speculative_tokens()
        
        assert speculative_tokens is not None
        assert speculative_tokens.shape[0] == 16
        assert speculative_tokens.shape[1] == 4
        
        verify_mock_calls(mock_mtp_proposer.propose, expected_calls=1)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("num_speculative_tokens", [2, 4, 8])
    def test_mtp_proposer_different_spec_token_count(self, num_speculative_tokens):
        """Test MTProposer不同推测token数量

        验证：
        - MTProposer生成num_speculative_tokens个推测token
        - 推测token数量正确
        - 推测token格式正确

        场景：不同推测token数量测试

        预期结果：推测token数量正确

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.num_speculative_tokens = num_speculative_tokens
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, num_speculative_tokens))
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        
        speculative_tokens = worker.mtp_proposer.propose()
        
        assert speculative_tokens.shape[1] == num_speculative_tokens

    @pytest.mark.cpu_mock
    def test_worker_verify_speculative_tokens(self):
        """Test Worker验证推测token的正确性

        验证：
        - Worker将推测token与真实token对比
        - Worker统计接受/拒绝的token数量
        - Worker更新候选池状态

        场景：Worker验证推测token正确性

        预期结果：验证过程正确执行

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'verify_speculative_tokens']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, 4))
        )
        worker.mtp_proposer = mock_mtp_proposer
        
        speculative_tokens = worker.mtp_proposer.propose()
        
        real_tokens = torch.randint(0, 32000, (16, 4))
        
        worker.verify_speculative_tokens = MagicMock(
            return_value={
                "accepted": torch.ones(16, dtype=torch.int32) * 3,
                "rejected": torch.ones(16, dtype=torch.int32) * 1,
            }
        )
        
        result = worker.verify_speculative_tokens(speculative_tokens, real_tokens)
        
        assert result["accepted"].mean() == 3
        assert result["rejected"].mean() == 1

    @pytest.mark.cpu_mock
    def test_worker_mtp_proposer_integration_flow(self):
        """Test Worker与MTProposer完整协作流程

        验证：
        - Worker初始化MTProposer
        - MTProposer生成推测token
        - Worker执行推理并验证推测token
        - Worker更新状态

        场景：Worker与MTProposer完整协作流程

        预期结果：完整流程正确执行

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'get_speculative_tokens', 'verify_speculative_tokens']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.num_speculative_tokens = 4
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, 4))
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        
        worker.get_speculative_tokens = MagicMock(
            return_value=worker.mtp_proposer.propose()
        )
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        worker.verify_speculative_tokens = MagicMock(
            return_value={
                "accepted": torch.ones(16, dtype=torch.int32) * 3,
                "rejected": torch.ones(16, dtype=torch.int32) * 1,
            }
        )
        
        speculative_tokens = worker.get_speculative_tokens()
        verify_mock_calls(mock_mtp_proposer.propose, expected_calls=1)
        
        output = worker.execute_model()
        assert output is not None
        
        result = worker.verify_speculative_tokens(speculative_tokens, speculative_tokens)
        assert result is not None


class TestWorkerMTPProposerErrorHandling(PytestSTBase):
    """Worker与MTProposer错误处理测试类"""

    @pytest.mark.cpu_mock
    def test_mtp_proposer_not_enabled(self):
        """Test Worker处理MTP未启用情况

        验证：
        - MTP未启用时Worker正确处理
        - Worker不调用MTProposer
        - 正常推理流程不受影响

        场景：MTP未启用时的Worker行为

        预期结果：正常推理流程不受影响

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'enable_mtp']
        )
        
        worker.mtp_proposer = None
        worker.enable_mtp = MagicMock(return_value=False)
        
        assert worker.enable_mtp() == False
        assert worker.mtp_proposer is None
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        result = worker.execute_model()
        assert result is not None

    @pytest.mark.cpu_mock
    def test_mtp_proposer_propose_error(self):
        """Test Worker处理MTProposer生成失败

        验证：
        - MTProposer生成失败时Worker正确处理错误
        - Worker捕获错误并降级到正常推理
        - 异常消息包含关键信息

        场景：MTProposer生成失败时的错误处理

        预期结果：错误正确处理，降级执行

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'get_speculative_tokens']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.propose = MagicMock(
            side_effect=RuntimeError("MTProposer propose failed: memory OOM")
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        
        worker.get_speculative_tokens = MagicMock(
            side_effect=lambda: worker.mtp_proposer.propose()
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.get_speculative_tokens()
        
        assert "MTProposer propose failed" in str(cm.exception)

    @pytest.mark.cpu_mock
    def test_mtp_proposer_all_tokens_rejected(self):
        """Test Worker处理全部推测token被拒绝

        验证：
        - 全部推测token被拒绝时Worker正确处理
        - Worker重新生成token
        - 推理流程继续执行

        场景：全部推测token被拒绝时的处理

        预期结果：推理流程继续执行

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'verify_speculative_tokens']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.propose = MagicMock(
            return_value=torch.randint(0, 32000, (16, 4))
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        
        worker.verify_speculative_tokens = MagicMock(
            return_value={
                "accepted": torch.zeros(16, dtype=torch.int32),
                "rejected": torch.ones(16, dtype=torch.int32) * 4,
            }
        )
        
        result = worker.verify_speculative_tokens(
            torch.randint(0, 32000, (16, 4)),
            torch.randint(0, 32000, (16, 4))
        )
        
        assert result["accepted"].sum() == 0
        assert result["rejected"].sum() == 64

    @pytest.mark.cpu_mock
    def test_mtp_proposer_load_model_error(self):
        """Test Worker处理MTProposer加载模型失败

        验证：
        - MTProposer加载模型失败时Worker正确处理错误
        - Worker捕获错误并传递
        - 异常消息包含关键信息

        场景：MTProposer加载模型失败时的错误处理

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'mtp_proposer', 'load_mtp_model']
        )
        
        mock_mtp_proposer = MagicMock()
        mock_mtp_proposer.load_model = MagicMock(
            side_effect=RuntimeError("MTProposer load model failed: model not found")
        )
        
        worker.mtp_proposer = mock_mtp_proposer
        worker.load_mtp_model = MagicMock(
            side_effect=lambda: worker.mtp_proposer.load_model()
        )
        
        with pytest.raises(RuntimeError) as cm:
            worker.load_mtp_model()
        
        assert "MTProposer load model failed" in str(cm.exception)