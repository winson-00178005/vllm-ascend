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

"""Integration placeholder tests (require real NPU and CANN)."""

import pytest
import torch

from tests.st.base import PytestSTBase
from tests.st.utils.env_detector import has_torch_npu


class TestIntegrationPlaceholder(PytestSTBase):
    """周边组件集成测试占位类"""

    @pytest.mark.npu_real
    def test_cann_compat_placeholder(self):
        """Test CANN版本兼容性占位测试

        验证：
        - NPU Real模式标记正确
        - CANN版本兼容测试框架正确
        - 为未来真实兼容测试预留接口

        场景：CANN版本兼容测试框架验证

        预期结果：框架正确

        执行模式：NPU Real
        """
        if not has_torch_npu():
            pytest.skip("NPU not available")
        
        pytest.skip("CANN compatibility test requires real hardware")

    @pytest.mark.npu_real
    def test_acl_graph_placeholder(self):
        """Test ACL Graph编译执行占位测试

        验证：
        - ACL Graph编译测试框架正确
        - ACL Graph执行测试框架正确
        - 为未来真实ACL Graph测试预留接口

        场景：ACL Graph测试框架验证

        预期结果：框架正确

        执行模式：NPU Real
        """
        if not has_torch_npu():
            pytest.skip("NPU not available")
        
        pytest.skip("ACL Graph test requires real hardware")