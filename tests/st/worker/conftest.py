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

import pytest

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.mock_utils import create_mock_worker, create_mock_model_runner


@pytest.fixture
def worker_config():
    """Worker配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
        batch_size=16
    )


@pytest.fixture(params=[1, 16, 32])
def batch_size(request):
    """参数化batch_size"""
    return request.param


@pytest.fixture(params=["normal", "error"])
def scenario(request):
    """参数化测试场景"""
    return request.param


@pytest.fixture(params=["float16", "bfloat16"])
def dtype(request):
    """参数化数据类型"""
    return request.param


@pytest.fixture
def worker_runner(st_environment, worker_config):
    """Worker STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("worker", worker_config)


@pytest.fixture
def mock_worker():
    """Mock Worker fixture"""
    return create_mock_worker()


@pytest.fixture
def mock_model_runner():
    """Mock ModelRunner fixture"""
    return create_mock_model_runner()