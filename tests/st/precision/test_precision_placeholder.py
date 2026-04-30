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

"""Precision placeholder tests (require real NPU hardware)."""

import pytest
import torch

from tests.st.base import PytestSTBase
from tests.st.utils.env_detector import has_torch_npu


class TestPrecisionPlaceholder(PytestSTBase):
    """精度测试占位类"""

    @pytest.mark.npu_precision
    def test_precision_placeholder(self):
        """Test 精度测试占位（需要真实NPU）

        验证：
        - NPU精度测试标记正确
        - 真实NPU环境下精度框架正确
        - 为未来真实精度测试预留接口

        场景：精度测试框架验证

        预期结果：精度测试框架正确

        执行模式：NPU Precision
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping precision test")
        
        pytest.skip("Precision test requires real NPU hardware - placeholder")