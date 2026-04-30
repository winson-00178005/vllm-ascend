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

"""Precision test fixtures for ST tests."""

import pytest
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu


PRECISION_THRESHOLDS = {
    "attention": {"rtol": 1e-3, "atol": 1e-5},
    "quantization": {"rtol": 1e-2, "atol": 1e-4},
    "sample": {"rtol": 1e-4, "atol": 1e-6},
    "default": {"rtol": 1e-3, "atol": 1e-5},
}


@pytest.fixture
def precision_config():
    """Precision测试配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
    )


@pytest.fixture(params=["attention", "quantization", "sample"])
def precision_module(request):
    """参数化精度测试模块"""
    return request.param


@pytest.fixture
def precision_threshold(precision_module):
    """精度误差阈值fixture"""
    return PRECISION_THRESHOLDS.get(precision_module, PRECISION_THRESHOLDS["default"])


@pytest.fixture
def precision_checker():
    """精度检查器fixture"""
    checker = MagicMock(spec=['compare', 'check_threshold', 'report'])
    
    def compare_tensors(cpu_tensor, npu_tensor, rtol, atol):
        import torch
        return torch.allclose(cpu_tensor, npu_tensor, rtol=rtol, atol=atol)
    
    checker.compare = MagicMock(side_effect=compare_tensors)
    checker.check_threshold = MagicMock(return_value=True)
    checker.report = MagicMock(return_value={"max_error": 1e-4, "mean_error": 1e-5})
    
    return checker