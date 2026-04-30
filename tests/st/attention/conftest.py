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

"""Attention module fixtures for ST tests."""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import create_mock_attention, create_mock_attention_mask
from tests.st.utils.data_generator import DEFAULT_SEQ_LENGTHS


@pytest.fixture
def attention_config():
    """Attention配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
        hidden_size=4096,
    )


@pytest.fixture(params=DEFAULT_SEQ_LENGTHS)
def seq_len(request):
    """参数化序列长度"""
    return request.param


@pytest.fixture(params=["full", "causal", "sliding_window"])
def mask_type(request):
    """参数化mask类型"""
    return request.param


@pytest.fixture
def mock_attention():
    """Mock Attention fixture"""
    return create_mock_attention()


@pytest.fixture
def mock_attention_mask():
    """Mock AttentionMask fixture"""
    return create_mock_attention_mask()


@pytest.fixture
def attention_runner(st_environment, attention_config):
    """Attention STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("attention", attention_config)


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
def attention_input_data(seq_len):
    """Attention输入数据fixture"""
    from tests.st.utils.data_generator import DataGenerator
    return DataGenerator.generate_attention_input(
        batch_size=16,
        seq_len=seq_len,
        hidden_dim=4096,
        dtype=torch.float16
    )


@pytest.fixture(params=["eager", "torchair_graph"])
def compile_mode(request):
    """参数化编译模式"""
    return request.param