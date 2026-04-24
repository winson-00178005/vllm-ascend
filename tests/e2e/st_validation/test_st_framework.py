#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""
ST 框架验证用例 - 用于验证场景驱动测试框架的功能

这些用例展示了如何使用 ST 框架的各种功能：
1. 场景过滤
2. 模型参数化
3. Fixture 生命周期
4. 分布式测试装饰器
"""

import pytest

from ..st_config.test_utils import large_gpu_test, multi_gpu_test


class TestSTFrameworkValidation:
    """验证 ST 框架基本功能"""

    @pytest.mark.scenario("single_card", "ascend_310p")
    def test_scenario_filtering(self, example_prompts):
        """验证场景过滤功能 - 只在 single_card 和 ascend_310p 场景下执行"""
        # 这个用例在其他场景下会被自动跳过
        assert len(example_prompts) > 0

    @pytest.mark.model("Qwen3-8B-Base")
    def test_model_marker(self, example_prompts):
        """验证模型 marker 功能 - 只针对 Qwen3-8B-Base 模型"""
        assert len(example_prompts) > 0


class TestVllmRunnerSession:
    """验证 session 级 fixture"""

    def test_session_fixture_basic(self, vllm_runner_session, example_prompts):
        """验证 session 级 fixture 可以正常工作"""
        # 注意：这个测试会实际加载模型，需要足够的 GPU 内存
        # 在实际运行时，应该只在有足够资源的机器上执行
        pass  # 占位，实际测试需要 GPU 环境


class TestVllmRunnerModule:
    """验证 module 级 fixture"""

    @pytest.mark.parametrize(
        "vllm_runner_module",
        ["Qwen3-8B-Base"],
        indirect=True,
    )
    def test_module_fixture_parametrized(self, vllm_runner_module, example_prompts):
        """验证 module 级 fixture 参数化"""
        # 这个测试会遍历指定的模型列表
        pass  # 占位，实际测试需要 GPU 环境


class TestMultiGPU:
    """验证多 GPU 测试装饰器"""

    @multi_gpu_test(num_gpus=2)
    def test_multi_gpu_decorator(self, example_prompts):
        """验证多 GPU 装饰器 - 需要至少 2 张 GPU"""
        pass  # 占位，实际测试需要多 GPU 环境


class TestLargeGPU:
    """验证大 GPU 内存测试装饰器"""

    @large_gpu_test(min_gb=32)
    def test_large_gpu_decorator(self, example_prompts):
        """验证大 GPU 内存装饰器 - 需要至少 32GB GPU 内存"""
        pass  # 占位，实际测试需要大内存 GPU
