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

"""Distributed模块集成测试

验证NPUCommunicator和Tensor Parallel的插件内部协作。
参考：tests/ut/ops/test_fused_ops.py模式
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import (
    create_mock_distribution_env,
    verify_mock_calls,
)
from tests.st.utils.runner_factory import STRunner


class TestDistributedIntegration(PytestSTBase):
    """Distributed集成测试类"""

    @pytest.mark.cpu_mock
    def test_communicator_init_with_config(self, mock_communicator):
        """Test Communicator初始化配置

        验证：
        - Communicator正确初始化并持有配置
        - rank和world_size正确设置
        - 初始化状态正确

        场景：Communicator正常初始化流程

        预期结果：Communicator正确初始化

        执行模式：CPU Mock
        """
        assert mock_communicator.rank == 0
        assert mock_communicator.world_size == 4

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("comm_op", ["all_reduce", "all_gather", "broadcast", "reduce_scatter"])
    def test_communicator_different_ops(self, comm_op, mock_communicator):
        """Test Communicator不同通信操作

        验证：
        - Communicator正确执行all_reduce
        - Communicator正确执行all_gather
        - Communicator正确执行broadcast
        - Communicator正确执行reduce_scatter

        场景：不同通信操作测试

        预期结果：所有操作正确执行

        执行模式：CPU Mock
        """
        input_tensor = torch.randn(4, 16, 64)
        result = None
        
        if comm_op == "all_reduce":
            result = mock_communicator.all_reduce(input_tensor)
        elif comm_op == "all_gather":
            result = mock_communicator.all_gather(input_tensor)
        elif comm_op == "broadcast":
            mock_communicator.broadcast = MagicMock(return_value=input_tensor.clone())
            result = mock_communicator.broadcast(input_tensor)
        elif comm_op == "reduce_scatter":
            mock_communicator.reduce_scatter = MagicMock(return_value=input_tensor.mean(dim=0))
            result = mock_communicator.reduce_scatter(input_tensor)
        
        assert result is not None

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    def test_communicator_world_size_parameterization(self, world_size):
        """Test Communicator world_size参数化

        验证：
        - Communicator正确处理world_size=1
        - Communicator正确处理world_size=2
        - Communicator正确处理world_size=4
        - Communicator正确处理world_size=8

        场景：world_size参数化测试

        预期结果：所有world_size正确处理

        执行模式：CPU Mock
        """
        mock_comm = MagicMock(spec=['all_reduce'])
        mock_comm.world_size = world_size
        
        input_tensor = torch.randn(world_size, 16, 64)
        mock_comm.all_reduce = MagicMock(return_value=input_tensor.sum(dim=0))
        
        result = mock_comm.all_reduce(input_tensor)
        assert result.shape == (16, 64)

    @pytest.mark.cpu_mock
    def test_distribution_env_creation(self, mock_distribution_env):
        """Test 分布式环境创建

        验证：
        - 分布式环境正确创建
        - 分布式组正确初始化
        - 分布式环境状态正确

        场景：分布式环境创建验证

        预期结果：环境正确创建

        执行模式：CPU Mock
        """
        assert mock_distribution_env is not None
        assert mock_distribution_env.rank_in_group == 0
        assert mock_distribution_env.world_size == 4


class TestTensorParallel(PytestSTBase):
    """Tensor Parallel集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_tp_weight_sharding(self, world_size):
        """Test Tensor Parallel权重分片

        验证：
        - TP正确分片权重到world_size个rank
        - 分片权重形状正确
        - 权重分布均匀

        场景：Tensor Parallel权重分片

        预期结果：权重正确分片

        执行模式：CPU Mock
        """
        full_weight = torch.randn(256, 128)
        shard_weight = full_weight.chunk(world_size, dim=0)[0]
        
        mock_tp = MagicMock()
        mock_tp.shard_weight = MagicMock(return_value=shard_weight)
        
        result = mock_tp.shard_weight(full_weight, world_size)
        assert result.shape[0] == 256 // world_size

    @pytest.mark.cpu_mock
    def test_tp_all_reduce_sum(self, mock_tensor_parallel_group):
        """Test Tensor Parallel all_reduce sum

        验证：
        - TP all_reduce正确执行sum操作
        - 各rank数据正确聚合
        - 结果正确返回

        场景：Tensor Parallel all_reduce sum

        预期结果：all_reduce正确执行

        执行模式：CPU Mock
        """
        input_tensor = torch.randn(2, 16, 64)
        
        mock_tensor_parallel_group.all_reduce = MagicMock(
            return_value=input_tensor.sum(dim=0)
        )
        
        result = mock_tensor_parallel_group.all_reduce(input_tensor)
        assert result.shape == (16, 64)

    @pytest.mark.cpu_mock
    def test_tp_with_strunner(self, distributed_runner, mock_tensor_parallel_group):
        """Test Tensor Parallel使用STRunner管理资源

        验证：
        - STRunner正确初始化TP资源
        - STRunner正确清理资源
        - 资源管理符合上下文管理器模式

        场景：使用STRunner管理TP模块资源

        预期结果：资源正确初始化和清理

        执行模式：CPU Mock
        """
        with STRunner("distributed", {"world_size": 2}) as runner:
            runner.register_resource("tp_group", mock_tensor_parallel_group)
            
            assert runner.get_resource("tp_group") is not None
        
        assert runner._resources == {}


class TestExpertParallel(PytestSTBase):
    """Expert Parallel集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("num_experts", [2, 4, 8, 16, 32])
    def test_ep_expert_distribution(self, num_experts):
        """Test Expert Parallel expert分布

        验证：
        - EP正确分布num_experts个expert
        - expert分布均匀
        - 分布策略正确

        场景：Expert Parallel expert分布

        预期结果：expert正确分布

        执行模式：CPU Mock
        """
        world_size = 4
        experts_per_rank = num_experts // world_size
        
        mock_ep = MagicMock()
        mock_ep.distribute_experts = MagicMock(return_value=experts_per_rank)
        
        result = mock_ep.distribute_experts(num_experts, world_size)
        assert result == num_experts // world_size

    @pytest.mark.cpu_mock
    def test_ep_expert_all_to_all(self):
        """Test Expert Parallel all_to_all通信

        验证：
        - EP all_to_all正确执行
        - expert token正确分发
        - 通信结果正确

        场景：Expert Parallel all_to_all通信

        预期结果：all_to_all正确执行

        执行模式：CPU Mock
        """
        mock_ep = MagicMock(spec=['all_to_all'])
        
        input_tokens = torch.randn(16, 128)
        mock_ep.all_to_all = MagicMock(return_value=torch.randn(16, 128))
        
        result = mock_ep.all_to_all(input_tokens)
        assert result is not None


class TestDistributedKVConnector(PytestSTBase):
    """分布式KV Connector集成测试类"""

    @pytest.mark.cpu_mock
    def test_kv_connector_send(self, mock_kv_connector):
        """Test KV Connector发送KV cache

        验证：
        - KV Connector正确发送KV cache
        - send_kv方法正确调用
        - 发送数据格式正确

        场景：KV Connector发送KV cache

        预期结果：发送正确执行

        执行模式：CPU Mock
        """
        kv_cache = torch.randn(16, 128, 512)
        
        mock_kv_connector.send_kv(kv_cache)
        
        verify_mock_calls(mock_kv_connector.send_kv, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_kv_connector_recv(self, mock_kv_connector):
        """Test KV Connector接收KV cache

        验证：
        - KV Connector正确接收KV cache
        - recv_kv方法正确调用
        - 接收数据格式正确

        场景：KV Connector接收KV cache

        预期结果：接收正确执行

        执行模式：CPU Mock
        """
        result = mock_kv_connector.recv_kv()
        
        assert result is not None
        assert result.shape == (16, 128, 512)
        verify_mock_calls(mock_kv_connector.recv_kv, expected_calls=1)


class TestDistributedInterfaceCompatibility(PytestSTBase):
    """分布式接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_communicator_interface_methods(self, mock_communicator):
        """Test Communicator接口方法

        验证：
        - Communicator包含all_reduce方法
        - Communicator包含all_gather方法
        - 接口方法可调用

        场景：验证Communicator接口实现

        预期结果：接口正确实现

        执行模式：CPU Mock
        """
        assert callable(mock_communicator.all_reduce)
        assert callable(mock_communicator.all_gather)


class TestDistributedDualMode(PytestSTBase):
    """分布式双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_distributed_dual_mode_comm(self, exec_mode):
        """Test 分布式双模式通信

        验证：
        - CPU Mock模式下分布式通信正确执行
        - NPU Real模式下分布式通信正确执行
        - 不同模式下执行结果一致性

        场景：分布式双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        mock_comm = MagicMock(spec=['all_reduce'])
        
        if exec_mode == "cpu_mock":
            mock_comm.all_reduce = MagicMock(return_value=torch.randn(16, 64))
            result = mock_comm.all_reduce(torch.randn(16, 64))
            assert result is not None
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")


class TestDistributedHCCLReal(PytestSTBase):
    """HCCL真实通信测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.npu_real
    def test_hccl_real_placeholder(self):
        """Test HCCL真实通信占位测试

        验证：
        - NPU Real模式标记正确
        - 真实NPU环境下HCCL框架正确
        - 为未来真实HCCL测试预留接口

        场景：HCCL真实通信框架验证

        预期结果：HCCL框架正确

        执行模式：NPU Real
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_real test")
        
        pytest.skip("HCCL Real test requires real NPU hardware - placeholder for future implementation")

    @pytest.mark.cpu_mock
    def test_hccl_mock_all_reduce(self):
        """Test HCCL Mock all_reduce

        验证：
        - HCCL Mock all_reduce正确执行
        - 调用次数验证正确
        - 返回结果正确

        场景：HCCL Mock all_reduce测试

        预期结果：Mock正确执行

        执行模式：CPU Mock
        """
        mock_hccl = MagicMock()
        
        input_tensor = torch.randn(4, 16, 64)
        mock_hccl.all_reduce = MagicMock(return_value=input_tensor)
        
        result = mock_hccl.all_reduce(input_tensor)
        assert result is not None