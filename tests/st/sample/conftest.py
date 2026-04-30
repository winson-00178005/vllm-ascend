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

"""Sample module fixtures for ST tests.

Reference: tests/ut/sample/test_rejection_sampler.py pattern
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.data_generator import (
    DEFAULT_BATCH_SIZES,
    PLACEHOLDER_TOKEN_ID,
)


PLACEHOLDER_TOKEN_ID = -1


@pytest.fixture
def sample_config():
    """Sample配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
    )


@pytest.fixture(params=DEFAULT_BATCH_SIZES)
def batch_size(request):
    """参数化batch_size"""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 0.7, 0.9, 1.0])
def temperature(request):
    """参数化temperature值"""
    return request.param


@pytest.fixture(params=[1, 10, 50, 100])
def top_k(request):
    """参数化top_k值"""
    return request.param


@pytest.fixture(params=["all_accept", "all_reject", "partial_accept"])
def rejection_scenario(request):
    """参数化rejection场景"""
    return request.param


@pytest.fixture
def mock_sampler():
    """Mock Sampler fixture"""
    mock_sample = MagicMock(spec=['sample', 'apply_temperature'])
    mock_sample.sample = MagicMock(return_value=torch.randint(0, 32000, (16,)))
    mock_sample.apply_temperature = MagicMock(return_value=torch.randn(16, 32000))
    return mock_sample


@pytest.fixture
def mock_rejection_sampler():
    """Mock RejectionSampler fixture"""
    mock_reject = MagicMock(spec=['verify', 'accept', 'reject'])
    mock_reject.verify = MagicMock(return_value=torch.ones(16, dtype=torch.bool))
    mock_reject.accept = MagicMock(return_value=torch.ones(16, dtype=torch.bool))
    mock_reject.reject = MagicMock(return_value=torch.zeros(16, dtype=torch.bool))
    return mock_reject


@pytest.fixture
def sample_runner(st_environment, sample_config):
    """Sample STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("sample", sample_config)


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


@pytest.fixture
def logits_data():
    """Logits数据fixture"""
    return torch.randn(16, 32000)


@pytest.fixture
def token_ids_data():
    """Token IDs数据fixture"""
    return torch.randint(0, 32000, (16, 4))