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

"""Scheduler module fixtures for ST tests."""

import pytest
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import create_mock_scheduler, create_mock_worker


@pytest.fixture
def scheduler_config():
    """Scheduler配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
        max_num_running_reqs=64,
        max_num_scheduled_tokens=2048,
    )


@pytest.fixture(params=[1, 16, 32, 64])
def num_running_reqs(request):
    """参数化运行请求数"""
    return request.param


@pytest.fixture(params=[256, 512, 1024, 2048])
def token_budget(request):
    """参数化token预算"""
    return request.param


@pytest.fixture(params=["prefill_first", "chunked_prefill"])
def scheduler_strategy(request):
    """参数化调度策略"""
    return request.param


@pytest.fixture
def scheduler_runner(st_environment, scheduler_config):
    """Scheduler STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("scheduler", scheduler_config)


@pytest.fixture
def mock_scheduler():
    """Mock Scheduler fixture"""
    return create_mock_scheduler()


@pytest.fixture
def mock_worker_for_scheduler():
    """Mock Worker fixture for Scheduler tests"""
    return create_mock_worker(spec_attrs=['execute_model', 'get_model'])


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


@pytest.fixture(params=["normal", "preempt", "requeue"])
def scheduling_scenario(request):
    """参数化调度场景"""
    return request.param


@pytest.fixture
def mock_scheduler_output():
    """Mock SchedulerOutput fixture"""
    mock_output = MagicMock()
    mock_output.num_scheduled_tokens = 1024
    mock_output.num_scheduled_seqs = 16
    mock_output.scheduled_new_reqs = []
    mock_output.scheduled_running_reqs = []
    mock_output.preempted_reqs = []
    return mock_output