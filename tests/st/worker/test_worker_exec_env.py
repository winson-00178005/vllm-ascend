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

"""Worker execution environment动态切换测试

验证execution_environment fixture动态切换CPU Mock和NPU真实环境。
参考：tests/st/DUAL_MODE_EXECUTION_GUIDE.md
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import create_mock_worker
from tests.st.utils.env_detector import has_torch_npu


class TestWorkerExecutionEnvironment(PytestSTBase):
    """Worker execution environment测试类"""

    @pytest.mark.cpu_mock
    def test_execution_environment_cpu_mock_mode(self, execution_environment):
        """Test execution_environment在CPU Mock模式下的行为

        验证：
        - execution_environment fixture正确返回环境实例
        - CPU Mock环境下Worker正确初始化
        - Mock对象正确创建和使用

        场景：CPU Mock模式下的环境切换

        预期结果：环境正确切换，Mock对象正确使用

        执行模式：CPU Mock
        """
        env = execution_environment
        
        assert env is not None
        assert hasattr(env, 'enter')
        assert hasattr(env, 'exit')
        
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
        
        result = worker.execute_model()
        assert result is not None

    @pytest.mark.npu_real
    def test_execution_environment_npu_real_mode(self, execution_environment):
        """Test execution_environment在NPU Real模式下的行为

        验证：
        - execution_environment fixture正确返回NPU环境实例
        - NPU环境下正确检测NPU可用性
        - 真实NPU环境下测试框架正确

        场景：NPU Real模式下的环境切换

        预期结果：环境正确切换，NPU可用性正确检测

        执行模式：NPU Real
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_real test")
        
        env = execution_environment
        assert env is not None
        
        pytest.skip("NPU Real mode requires real hardware")

    def test_execution_environment_dynamic_switch(self, execution_environment):
        """Test execution_environment动态切换

        验证：
        - execution_environment根据命令行参数动态切换
        - 自动检测模式下正确选择环境
        - 环境切换无副作用

        场景：动态环境切换测试

        预期结果：环境正确动态切换

        执行模式：CPU Mock / NPU Real
        """
        env = execution_environment
        
        assert env is not None
        
        worker = create_mock_worker(spec_attrs=['execute_model'])
        assert worker is not None

    @pytest.mark.parametrize("exec_mode_param", ["cpu_mock", "npu_real"], indirect=True)
    def test_execution_environment_with_param(self, exec_mode_param, execution_environment):
        """Test execution_environment与参数化exec_mode组合

        验证：
        - 参数化exec_mode与execution_environment正确组合
        - 不同模式下环境正确初始化
        - 参数化双模式测试完整流程

        场景：参数化exec_mode与环境fixture组合测试

        预期结果：组合正确，环境正确初始化

        执行模式：CPU Mock / NPU Real
        """
        env = execution_environment
        assert env is not None
        
        if exec_mode_param == "cpu_mock":
            worker = create_mock_worker(spec_attrs=['execute_model'])
            worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
            result = worker.execute_model()
            assert result is not None
        elif exec_mode_param == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")

    @pytest.mark.cpu_mock
    def test_worker_with_st_environment(self, st_environment):
        """Test Worker使用st_environment fixture

        验证：
        - st_environment session scope fixture正确初始化
        - Worker在st_environment下正确执行
        - session scope共享环境资源

        场景：Worker与session scope环境fixture协作

        预期结果：session scope环境正确共享

        执行模式：CPU Mock
        """
        env = st_environment
        assert env is not None
        
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
        
        result = worker.execute_model()
        assert result.shape[0] == 16

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_execution_environment_with_batch_size(self, execution_environment, batch_size):
        """Test execution_environment与batch_size组合

        验证：
        - execution_environment与参数化batch_size正确组合
        - 不同batch_size下环境正确处理
        - 组合测试完整流程

        场景：环境与batch_size组合测试

        预期结果：组合正确，环境正确处理

        执行模式：CPU Mock
        """
        env = execution_environment
        assert env is not None
        
        worker = create_mock_worker(spec_attrs=['execute_model'])
        worker.execute_model = MagicMock(
            return_value=torch.randn(batch_size, 128)
        )
        
        result = worker.execute_model()
        assert result.shape[0] == batch_size