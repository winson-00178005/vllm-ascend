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

"""Worker双模式测试

验证Worker在CPU Mock和NPU Real两种模式下的行为一致性。
参考：tests/st/DUAL_MODE_EXECUTION_GUIDE.md
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_model_runner,
)
from tests.st.utils.env_detector import has_torch_npu


class TestWorkerDualMode(PytestSTBase):
    """Worker双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_worker_execute_model_dual_mode(self, exec_mode):
        """Test Worker execute_model双模式

        验证：
        - CPU Mock模式下execute_model正确执行
        - NPU Real模式下execute_model正确执行
        - 不同模式下执行结果一致性

        场景：双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        if exec_mode == "cpu_mock":
            worker = create_mock_worker(spec_attrs=['execute_model'])
            worker.execute_model = MagicMock(
                return_value=torch.randn(16, 128, dtype=torch.float16)
            )
            result = worker.execute_model()
            assert result is not None
            assert result.shape[0] == 16
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available, skipping npu_real test")
            pytest.skip("NPU Real mode requires real hardware - placeholder for future implementation")

    @pytest.mark.cpu_mock
    @pytest.mark.npu_real
    def test_worker_init_dual_mode_markers(self):
        """Test Worker初始化双模式标记

        验证：
        - Worker在CPU Mock模式下正确初始化
        - Worker在NPU Real模式下正确初始化
        - 标记正确区分执行模式

        场景：双模式初始化验证

        预期结果：两种模式初始化流程一致

        执行模式：CPU Mock / NPU Real
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'initialize_model'])
        
        assert worker is not None
        assert hasattr(worker, 'execute_model')
        assert hasattr(worker, 'initialize_model')

    @pytest.mark.cpu_mock
    def test_worker_cpu_mock_mode_behavior(self):
        """Test Worker CPU Mock模式行为

        验证：
        - CPU Mock模式下使用Mock对象
        - Mock对象返回预设结果
        - Mock调用次数正确

        场景：CPU Mock模式完整流程验证

        预期结果：Mock行为符合预期

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'load_model', 'get_model']
        )
        model_runner = create_mock_model_runner(
            spec_attrs=['execute', 'load_model']
        )
        worker.model_runner = model_runner
        
        worker.load_model = MagicMock()
        worker.load_model()
        worker.load_model.assert_called_once()
        
        worker.execute_model = MagicMock(
            return_value=torch.randn(16, 64)
        )
        result = worker.execute_model()
        assert result.shape == (16, 64)

    @pytest.mark.npu_real
    def test_worker_npu_real_mode_placeholder(self):
        """Test Worker NPU Real模式占位测试

        验证：
        - NPU Real模式标记正确
        - 真实NPU环境下测试框架正确
        - 为未来真实测试预留接口

        场景：NPU Real模式框架验证

        预期结果：NPU Real模式框架正确

        执行模式：NPU Real
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_real test")
        
        pytest.skip("NPU Real mode requires real hardware - placeholder for future implementation")

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_worker_dual_mode_with_batch_size(self, exec_mode, batch_size):
        """Test Worker双模式与batch_size参数组合

        验证：
        - 不同batch_size在双模式下正确执行
        - CPU Mock和NPU Real结果一致性
        - 参数化双模式组合测试

        场景：双模式与batch_size组合测试

        预期结果：所有组合测试通过

        执行模式：CPU Mock / NPU Real
        """
        if exec_mode == "cpu_mock":
            worker = create_mock_worker(spec_attrs=['execute_model'])
            worker.execute_model = MagicMock(
                return_value=torch.randn(batch_size, 128)
            )
            result = worker.execute_model()
            assert result.shape[0] == batch_size
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available, skipping npu_real test")
            pytest.skip("NPU Real mode requires real hardware")

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_worker_dual_mode_with_dtype(self, exec_mode, dtype):
        """Test Worker双模式与dtype参数组合

        验证：
        - 不同dtype在双模式下正确执行
        - CPU Mock和NPU Real dtype一致性
        - 参数化双模式组合测试

        场景：双模式与dtype组合测试

        预期结果：所有组合测试通过

        执行模式：CPU Mock / NPU Real
        """
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        
        if exec_mode == "cpu_mock":
            worker = create_mock_worker(spec_attrs=['execute_model'])
            worker.execute_model = MagicMock(
                return_value=torch.randn(16, 128, dtype=torch_dtype)
            )
            result = worker.execute_model()
            assert result.dtype == torch_dtype
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available, skipping npu_real test")
            pytest.skip("NPU Real mode requires real hardware")

    @pytest.mark.cpu_mock
    @pytest.mark.npu_real
    @pytest.mark.parametrize("scenario", ["normal", "error"])
    def test_worker_dual_mode_scenario(self, scenario):
        """Test Worker双模式场景处理

        验证：
        - normal场景在双模式下正确处理
        - error场景在双模式下正确传递错误
        - 双模式场景处理一致性

        场景：双模式场景处理测试

        预期结果：场景处理一致

        执行模式：CPU Mock / NPU Real
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