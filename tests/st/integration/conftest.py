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

"""Integration test fixtures for ST tests."""

import pytest
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu


@pytest.fixture
def integration_config():
    """Integration测试配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
    )


@pytest.fixture(params=["cann_compat", "acl_graph", "hccl_comm", "torchair"])
def integration_component(request):
    """参数化周边组件"""
    return request.param


@pytest.fixture(params=["8.0.RC1", "8.0.RC2", "8.0.RC3"])
def cann_version(request):
    """参数化CANN版本"""
    return request.param


@pytest.fixture
def acl_graph_compiler():
    """ACL Graph编译器fixture"""
    compiler = MagicMock(spec=['compile', 'execute', 'get_status'])
    compiler.compile = MagicMock(return_value=True)
    compiler.execute = MagicMock(return_value=True)
    compiler.get_status = MagicMock(return_value="success")
    return compiler


@pytest.fixture
def hccl_comm_tester():
    """HCCL通信测试器fixture"""
    tester = MagicMock(spec=['test_all_reduce', 'test_all_gather', 'test_broadcast'])
    tester.test_all_reduce = MagicMock(return_value=True)
    tester.test_all_gather = MagicMock(return_value=True)
    tester.test_broadcast = MagicMock(return_value=True)
    return tester