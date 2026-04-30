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

import unittest

import pytest

from vllm_ascend.utils import adapt_patch, register_ascend_customop


class TestSTBase(unittest.TestCase):
    """ST测试基类（unittest风格），用于插件模块集成测试。

    继承unittest.TestCase，在__init__中自动初始化Mock环境，
    支持setUp/tearDown资源管理。

    适用场景：
    - 传统unittest测试场景
    - 需要assertRaises等unittest方法
    - 与现有UT测试风格保持一致

    使用示例：
    ```python
    class TestWorkerIntegration(TestSTBase):
        def setUp(self):
            self.worker = create_mock_worker()

        def tearDown(self):
            cleanup_mock_worker(self.worker)

        def test_worker_schedule(self):
            self.worker.schedule(...)
            ...
    ```
    """

    def __init__(self, *args, **kwargs):
        adapt_patch(True)
        adapt_patch()
        register_ascend_customop()
        super().setUp()
        super(TestSTBase, self).__init__(*args, **kwargs)


class PytestSTBase:
    """ST测试基类（pytest风格），用于插件模块集成测试。

    使用pytest fixture autouse自动初始化环境，
    支持pytest mocker和parametrize功能。

    适用场景：
    - 复杂fixture和参数化场景
    - pytest-mock Mock功能
    - @pytest.mark.parametrize多维度测试

    使用示例：
    ```python
    class TestAttentionIntegration(PytestSTBase):
        @pytest.mark.parametrize("batch_size", [1, 16, 32])
        def test_attention_forward(self, batch_size):
            ...
    ```
    """

    @pytest.fixture(autouse=True)
    def setup_st(self):
        adapt_patch(True)
        adapt_patch()
        register_ascend_customop()
        yield