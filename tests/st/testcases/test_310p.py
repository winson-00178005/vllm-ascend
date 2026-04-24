#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

import os
import pytest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from tests.e2e.conftest import VllmRunner
from tests.st.framework.decorators import require_scene


class Test310PSingleCard:
    """310P 单卡测试"""

    @require_scene("310P_SINGLECARD")
    def test_310p_basic_generation(self):
        """310P 基础生成测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            tensor_parallel_size=1,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("310P_SINGLECARD")
    def test_310p_fp16(self):
        """310P FP16 测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            tensor_parallel_size=1,
            dtype="float16",
            enforce_eager=True,
            max_model_len=1024,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1

    @require_scene("310P_SINGLECARD")
    def test_310p_long_prompt(self):
        """310P �?prompt 测试"""
        prompts = ["This is a test " * 100]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            tensor_parallel_size=1,
            max_model_len=2048,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
