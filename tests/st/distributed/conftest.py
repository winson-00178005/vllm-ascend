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

"""Distributed module fixtures for ST tests.

Reference: tests/ut/ops/test_fused_ops.py pattern
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import create_mock_distribution_env


@pytest.fixture
def distributed_config():
    """Distributed配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
        tensor_parallel_size=2,
    )


@pytest.fixture(params=[1, 2, 4, 8])
def world_size(request):
    """参数化world_size"""
    return request.param


@pytest.fixture(params=[0, 1, 2, 3])
def rank(request):
    """参数化rank"""
    return request.param


@pytest.fixture(params=["all_reduce", "all_gather", "broadcast", "reduce_scatter"])
def comm_op(request):
    """参数化通信操作"""
    return request.param


@pytest.fixture
def mock_communicator():
    """Mock Communicator fixture"""
    mock_comm = MagicMock(spec=['all_reduce', 'all_gather', 'broadcast', 'reduce_scatter'])
    mock_comm.rank = 0
    mock_comm.world_size = 4
    mock_comm.all_reduce = MagicMock(return_value=torch.randn(4, 16, 64))
    mock_comm.all_gather = MagicMock(return_value=torch.randn(4, 16, 64))
    return mock_comm


@pytest.fixture
def mock_distribution_env():
    """Mock分布式环境fixture"""
    return create_mock_distribution_env(MagicMock())


@pytest.fixture
def distributed_runner(st_environment, distributed_config):
    """Distributed STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("distributed", distributed_config)


@pytest.fixture
def execution_environment(st_environment, request):
    """Execution environment fixture for dynamic switching."""
    return st_environment


@pytest.fixture(params=["cpu_mock", "npu_real"])
def exec_mode_param(request):
    """Parameterized execution mode fixture."""
    mode = request.param
    if mode == "npu_real":
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_real test")
    return mode


@pytest.fixture(params=[2, 4, 8, 16, 32])
def num_experts(request):
    """参数化expert数量"""
    return request.param


@pytest.fixture
def mock_tensor_parallel_group():
    """Mock Tensor Parallel group fixture"""
    mock_group = MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 2
    mock_group.all_reduce = MagicMock(return_value=torch.randn(2, 16, 64))
    return mock_group


@pytest.fixture
def mock_kv_connector():
    """Mock KV Connector fixture"""
    mock_connector = MagicMock(spec=['send_kv', 'recv_kv'])
    mock_connector.send_kv = MagicMock()
    mock_connector.recv_kv = MagicMock(return_value=torch.randn(16, 128, 512))
    return mock_connector