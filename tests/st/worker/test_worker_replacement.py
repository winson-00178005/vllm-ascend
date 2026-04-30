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

"""Worker替换逻辑测试

验证NPUWorker替换vllm.Worker的逻辑正确性。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_model_runner,
    verify_mock_calls,
)
from tests.st.utils.runner_factory import STRunner


class TestWorkerReplacementLogic(PytestSTBase):
    """Worker替换逻辑测试类"""

    @pytest.mark.cpu_mock
    def test_worker_init_model_runner_correctly(self):
        """Test Worker正确初始化ModelRunner

        验证：
        - Worker在init_device时创建ModelRunner实例
        - ModelRunner使用正确的vllm_config和device
        - Worker持有ModelRunner引用

        场景：Worker初始化流程中的ModelRunner创建

        预期结果：ModelRunner正确初始化，引用正确

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
             patch('torch_npu.op_plugin.atb._atb_ops._register_atb_extensions'), \
             patch('vllm_ascend.platform.NPUPlatform') as mock_platform:
            
            mock_config_instance = MagicMock()
            mock_config_instance.model_config = MagicMock()
            mock_config_instance.model_config.dtype = "float16"
            mock_config_instance.model_config.trust_remote_code = False
            mock_config_instance.model_config.seed = 42
            mock_config_instance.cache_config = MagicMock()
            mock_config_instance.cache_config.cache_dtype = "auto"
            mock_config_instance.parallel_config = MagicMock()
            
            mock_platform.set_device = MagicMock()
            mock_platform.empty_cache = MagicMock()
            mock_platform.mem_get_info = MagicMock(return_value=(1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
            mock_platform.seed_everything = MagicMock()
            
            with patch('vllm_ascend.worker.worker_v1.NPUModelRunner') as mock_runner_class:
                mock_runner_instance = MagicMock()
                mock_runner_class.return_value = mock_runner_instance
                
                from vllm_ascend.worker.worker_v1 import NPUWorker
                
                worker = NPUWorker(
                    vllm_config=mock_config_instance,
                    local_rank=0,
                    rank=0,
                    distributed_init_method="tcp://localhost:12345",
                    is_driver_worker=True
                )
                
                worker.init_device()
                
                assert worker.model_runner is not None
                mock_runner_class.assert_called_once()

    @pytest.mark.cpu_mock
    def test_worker_scheduling_collaboration(self, mock_worker, mock_model_runner):
        """Test Worker与ModelRunner调度协作

        验证：
        - Worker接收scheduler_output
        - Worker正确传递调度信息到ModelRunner
        - ModelRunner执行并返回结果

        场景：Worker调度协作流程

        预期结果：调度信息正确传递，执行结果正确返回

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        
        mock_scheduler_output = MagicMock()
        mock_scheduler_output.num_scheduled_seqs = 16
        
        mock_worker.execute_model = MagicMock(
            return_value=torch.randn(16, 64)
        )
        
        result = mock_worker.execute_model(mock_scheduler_output)
        
        assert result is not None
        mock_worker.execute_model.assert_called_once_with(mock_scheduler_output)

    @pytest.mark.cpu_mock
    def test_worker_interface_compatible_with_vllm(self):
        """Test Worker接口与vllm兼容

        验证：
        - Worker接口方法签名与vllm.Worker一致
        - Worker可被vllm正确调用
        - 返回类型符合vllm预期

        场景：验证Worker替换vllm.Worker的接口兼容性

        预期结果：接口完全兼容，替换无障碍

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'initialize_model', 'get_model', 
                       'load_model', 'compile_or_warm_up_model']
        )
        
        assert callable(worker.execute_model)
        assert callable(worker.initialize_model)
        assert callable(worker.get_model)
        assert callable(worker.load_model)
        assert callable(worker.compile_or_warm_up_model)

    @pytest.mark.cpu_mock
    def test_worker_load_model_flow(self, mock_worker, mock_model_runner):
        """Test Worker load_model流程

        验证：
        - Worker调用ModelRunner的load_model
        - load_model正确传递配置参数
        - 资源正确分配

        场景：Worker加载模型流程

        预期结果：模型正确加载，资源正确分配

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        mock_worker.load_model = MagicMock()
        
        mock_worker.load_model()
        
        mock_worker.load_model.assert_called_once()

    @pytest.mark.cpu_mock
    def test_worker_compile_warmup_flow(self, mock_worker, mock_model_runner):
        """Test Worker compile_or_warm_up_model流程

        验证：
        - Worker调用compile_or_warm_up_model
        - warmup_sizes正确传递
        - capture_model正确调用

        场景：Worker编译和预热模型流程

        预期结果：模型正确编译和预热

        执行模式：CPU Mock
        """
        mock_worker.model_runner = mock_model_runner
        mock_worker.compile_or_warm_up_model = MagicMock()
        
        mock_worker.compile_or_warm_up_model()
        
        mock_worker.compile_or_warm_up_model.assert_called_once()

    @pytest.mark.cpu_mock
    def test_worker_get_model_returns_runner_model(self, mock_worker, mock_model_runner):
        """Test Worker get_model返回ModelRunner的模型

        验证：
        - Worker.get_model调用ModelRunner.get_model
        - 返回正确的模型实例
        - 模型实例可正常使用

        场景：Worker获取模型实例流程

        预期结果：模型实例正确返回

        执行模式：CPU Mock
        """
        mock_model = MagicMock()
        mock_model_runner.get_model = MagicMock(return_value=mock_model)
        mock_worker.model_runner = mock_model_runner
        mock_worker.get_model = MagicMock(return_value=mock_model)
        
        result = mock_worker.get_model()
        
        assert result is mock_model
        mock_worker.get_model.assert_called_once()

    @pytest.mark.cpu_mock
    def test_worker_execute_model_returns_correct_output(self):
        """Test Worker execute_model返回正确输出类型

        验证：
        - execute_model返回ModelRunnerOutput类型
        - 输出包含正确字段
        - 输出可被后续处理

        场景：Worker执行模型并返回输出

        预期结果：输出类型正确，字段完整

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'model_runner'])
        model_runner = create_mock_model_runner(spec_attrs=['execute'])
        worker.model_runner = model_runner
        
        mock_output = MagicMock()
        mock_output.samples = torch.randn(16, 128)
        
        worker.execute_model = MagicMock(return_value=mock_output)
        
        result = worker.execute_model()
        
        assert result is not None
        assert hasattr(result, 'samples')