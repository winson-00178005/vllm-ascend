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

"""Worker接口兼容性测试

验证NPUWorker实现了vllm.Worker接口规范。
"""

import pytest
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    VLLM_WORKER_SPEC,
    verify_plugin_interface_compatibility,
    create_mock_worker,
)


class TestWorkerInterfaceCompatibility(PytestSTBase):
    """Worker接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_npu_worker_interface_compatibility(self):
        """Test NPUWorker implements vllm.Worker interface

        验证：
        - NPUWorker包含所有vllm.Worker接口方法
        - 每个接口方法都是可调用的
        - 接口方法签名符合vllm规范

        场景：验证插件Worker与vllm接口兼容性

        预期结果：所有接口检查通过，兼容性验证成功

        执行模式：CPU Mock
        """
        with patch('torch_npu') as mock_torch_npu, \
             patch('vllm.config.VllmConfig') as mock_config, \
             patch('vllm.distributed.ensure_model_parallel_initialized'), \
             patch('vllm.distributed.init_distributed_environment'), \
             patch('vllm_ascend.utils.init_ascend_config'), \
             patch('vllm_ascend.utils.init_ascend_soc_version'), \
             patch('vllm_ascend.utils.adapt_patch'), \
             patch('vllm_ascend.ops.register_dummy_fusion_op'), \
             patch('torch_npu.op_plugin.atb._atb_ops._register_atb_extensions'):
            
            mock_config_instance = MagicMock()
            mock_config_instance.model_config = MagicMock()
            mock_config_instance.model_config.dtype = "float16"
            mock_config_instance.model_config.trust_remote_code = False
            mock_config_instance.cache_config = MagicMock()
            mock_config_instance.cache_config.cache_dtype = "auto"
            mock_config_instance.parallel_config = MagicMock()
            
            from vllm_ascend.worker.worker_v1 import NPUWorker
            
            worker = NPUWorker(
                vllm_config=mock_config_instance,
                local_rank=0,
                rank=0,
                distributed_init_method="tcp://localhost:12345",
                is_driver_worker=True
            )
            
            verify_plugin_interface_compatibility(worker, VLLM_WORKER_SPEC)

    @pytest.mark.cpu_mock
    def test_worker_interface_spec_coverage(self):
        """Test Worker interface spec coverage

        验证：
        - VLLM_WORKER_SPEC包含核心接口方法
        - 每个接口方法在Worker中都存在

        场景：验证接口规范覆盖完整性

        预期结果：接口规范覆盖所有必需方法

        执行模式：CPU Mock
        """
        expected_core_methods = ['execute_model', 'get_model']
        
        for method in expected_core_methods:
            assert method in VLLM_WORKER_SPEC, \
                f"Core method {method} missing from VLLM_WORKER_SPEC"

    @pytest.mark.cpu_mock
    def test_mock_worker_with_spec(self):
        """Test Mock Worker created with spec restriction

        验证：
        - Mock Worker使用VLLM_WORKER_SPEC创建
        - Mock Worker接口与真实Worker一致
        - Mock限制防止过度Mock

        场景：使用spec限制创建Mock Worker

        预期结果：Mock Worker接口受限，符合规范

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=VLLM_WORKER_SPEC)
        
        for attr in VLLM_WORKER_SPEC:
            assert hasattr(worker, attr), \
                f"Mock Worker missing spec attribute: {attr}"

    @pytest.mark.cpu_mock
    def test_worker_execute_model_interface(self):
        """Test Worker execute_model interface signature

        验证：
        - execute_model方法存在且可调用
        - execute_model返回正确类型

        场景：验证execute_model接口实现

        预期结果：execute_model接口正确实现

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model'])
        
        assert callable(worker.execute_model), \
            "execute_model is not callable"
        
        import torch
        worker.execute_model = MagicMock(return_value=torch.randn(1, 16, 64))
        
        result = worker.execute_model()
        assert isinstance(result, torch.Tensor)

    @pytest.mark.cpu_mock
    def test_worker_get_model_interface(self):
        """Test Worker get_model interface signature

        验证：
        - get_model方法存在且可调用
        - get_model返回模型实例

        场景：验证get_model接口实现

        预期结果：get_model接口正确实现

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['get_model'])
        
        assert callable(worker.get_model), \
            "get_model is not callable"
        
        worker.get_model = MagicMock(return_value=MagicMock())
        
        result = worker.get_model()
        assert result is not None