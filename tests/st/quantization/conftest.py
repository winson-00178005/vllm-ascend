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

"""Quantization module fixtures for ST tests.

Reference: tests/ut/quantization/test_w8a8.py pattern
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.data_generator import DEFAULT_DTYPES


@pytest.fixture
def quant_config():
    """Quantization配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
        quantization="w8a8",
    )


@pytest.fixture(params=["w8a8", "w8a8_dynamic", "w4a8"])
def quant_method(request):
    """参数化量化方法"""
    return request.param


@pytest.fixture(params=DEFAULT_DTYPES)
def quant_dtype(request):
    """参数化量化数据类型"""
    return request.param


@pytest.fixture(params=["per_tensor", "per_channel"])
def quant_scale_type(request):
    """参数化量化scale类型"""
    return request.param


@pytest.fixture
def mock_quantizer():
    """Mock Quantizer fixture"""
    mock_quant = MagicMock(spec=['quantize', 'get_weight', 'apply'])
    mock_quant.quantize = MagicMock(return_value=torch.randn(128, 256))
    mock_quant.get_weight = MagicMock(return_value={'weight': torch.int8})
    mock_quant.apply = MagicMock(return_value=torch.randn(32, 128))
    return mock_quant


@pytest.fixture
def mock_quant_config():
    """Mock QuantConfig fixture"""
    mock_config = MagicMock()
    mock_config.quant_method = "w8a8"
    mock_config.weight_bits = 8
    mock_config.activation_bits = 8
    mock_config.group_size = 128
    return mock_config


@pytest.fixture
def quant_runner(st_environment, quant_config):
    """Quantization STRunner fixture"""
    from tests.st.utils.runner_factory import STRunner
    return STRunner("quantization", quant_config)


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
def quant_test_weights():
    """Quantization测试权重fixture"""
    weight = torch.randint(-128, 127, (256, 128), dtype=torch.int8)
    scale = torch.tensor(0.1)
    offset = torch.tensor(0.0)
    return {
        'weight': weight,
        'scale': scale,
        'offset': offset,
    }


@pytest.fixture(params=[1, 2, 4])
def world_size(request):
    """参数化分布式world_size"""
    return request.param