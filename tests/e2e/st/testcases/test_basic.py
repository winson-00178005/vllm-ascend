#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

import os
import pytest
from unittest.mock import patch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from tests.e2e.conftest import VllmRunner
from tests.e2e.st.framework.decorators import require_scene, require_model, require_scene_and_model
from tests.e2e.st.framework.compat import scene_aware_parametrize


class TestBasicInference:
    """基础推理测试"""

    @require_scene("SINGLECARD")
    def test_single_token_generation(self):
        """单步生成测试 - 单卡场景"""
        prompts = ["Hello, my name is"]
        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_2Cards")
    def test_multicard_2cards_basic(self):
        """2卡场景基础测试"""
        prompts = ["Hello, my name is"]
        with VllmRunner("Qwen/Qwen3-30B-A3B", tensor_parallel_size=2, max_model_len=512) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_4Cards")
    def test_multicard_4cards_basic(self):
        """4卡场景基础测试"""
        prompts = ["Hello, my name is"]
        with VllmRunner("Qwen/Qwen3-Next-80B-A3B-Instruct", tensor_parallel_size=4, max_model_len=512) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0


class TestPrefixCaching:
    """Prefix Caching 测试"""

    @require_scene("SINGLECARD")
    def test_prefix_caching_single(self):
        """单卡 Prefix Caching 测试"""
        prompts = [
            "Hello, my name is",
            "Hello, my name is Tom, I am a student",
            "Hello, my name is John, I am a teacher",
        ]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            max_model_len=1024,
            enable_prefix_caching=True,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 3
        for output_ids, output_str in outputs:
            assert len(output_ids) > 0

    @require_scene("MULTICARD_2Cards")
    def test_prefix_caching_multicard(self):
        """多卡 Prefix Caching 测试"""
        prompts = [
            "The capital of France is",
            "The capital of France is Paris, it is",
            "The capital of Germany is Berlin, which is",
        ]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=1024,
            enable_prefix_caching=True,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 3


class TestChunkedPrefill:
    """Chunked Prefill 测试"""

    @require_scene("SINGLECARD")
    def test_chunked_prefill_singlecard(self):
        """单卡 Chunked Prefill 测试"""
        prompts = [
            "This is a very long prompt " * 50,
            "Another long prompt " * 50,
        ]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            max_model_len=2048,
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 2

    @require_scene("MULTICARD_2Cards")
    def test_chunked_prefill_multicard(self):
        """多卡 Chunked Prefill 测试"""
        prompts = [
            "A" * 500,
            "B" * 500,
            "C" * 500,
        ]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=2048,
            enable_chunked_prefill=True,
            max_num_batched_tokens=256,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 3


class TestEagerMode:
    """Eager Mode 测试"""

    @require_scene("SINGLECARD")
    def test_eager_mode_basic(self):
        """Eager 模式基础测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-8B",
            max_model_len=512,
            enforce_eager=True,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_2Cards")
    def test_eager_mode_multicard(self):
        """Eager 模式多卡测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=512,
            enforce_eager=True,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0


class TestMultiInstance:
    """多实例测试"""

    @require_scene("SINGLECARD")
    def test_two_instances_on_single_card(self):
        """单卡双实例测试 - 验证内存管理"""
        model = "Qwen/Qwen3-0.6B"
        prompts = ["Hello, my name is"]
        max_tokens = 5

        with VllmRunner(
            model,
            max_model_len=512,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
        ) as runner1:
            with VllmRunner(
                model,
                max_model_len=512,
                gpu_memory_utilization=0.4,
                enforce_eager=True,
            ) as runner2:
                outputs2 = runner2.generate_greedy(prompts, max_tokens=max_tokens)

            outputs1 = runner1.generate_greedy(prompts, max_tokens=max_tokens)

        assert outputs1, "First instance produced no outputs"
        assert outputs2, "Second instance produced no outputs"
        assert len(outputs1[0][1]) > 0, "First instance output text is empty"
        assert len(outputs2[0][1]) > 0, "Second instance output text is empty"


class TestQuantization:
    """量化测试"""

    @require_scene("SINGLECARD")
    def test_w8a8_quantization_singlecard(self):
        """单卡 W8A8 量化测试"""
        max_tokens = 5
        prompts = ["vLLM is a high-throughput inference engine."]

        with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A8",
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            quantization="ascend",
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_2Cards")
    def test_w8a8_quantization_multicard(self):
        """多卡 W8A8 量化测试"""
        max_tokens = 5
        prompts = ["vLLM is a high-throughput inference engine."]

        with VllmRunner(
            "vllm-ascend/Qwen3-30B-A3B-W8A8",
            tensor_parallel_size=2,
            max_model_len=8192,
            gpu_memory_utilization=0.7,
            quantization="ascend",
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_2Cards")
    def test_w8a16_quantization(self):
        """W8A16 量化测试"""
        max_tokens = 5
        prompts = ["Hello, how are you?"]

        with VllmRunner(
            "vllm-ascend/Qwen3-0.6B-W8A16",
            tensor_parallel_size=2,
            max_model_len=8192,
            enforce_eager=False,
            gpu_memory_utilization=0.7,
            quantization="ascend",
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1


class TestMoEModels:
    """MoE 模型测试"""

    @require_scene("MULTICARD_2Cards")
    def test_moe_tp2_basic(self):
        """MoE 模型 TP2 基础测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_4Cards")
    def test_moe_tp4_basic(self):
        """MoE 模型 TP4 基础测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_2Cards")
    def test_moe_with_quantization(self):
        """MoE 量化模型测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "vllm-ascend/Qwen3-30B-A3B-W8A8",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            max_model_len=512,
            quantization="ascend",
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1


class TestContextParallelism:
    """Context Parallelism 测试"""

    @require_scene("MULTICARD_2Cards")
    def test_cp_basic(self):
        """Context Parallelism 基础测试"""
        prompts = ["Hello, my name is", "The capital of France is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=1024,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 2

    @require_scene("MULTICARD_4Cards")
    def test_cp_tp4(self):
        """Context Parallelism TP4 测试"""
        prompts = ["A" * 200, "B" * 200, "C" * 200]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            max_model_len=1024,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 3


class TestSamplingParams:
    """采样参数测试"""

    @require_scene("SINGLECARD")
    def test_greedy_sampling(self):
        """Greedy 采样测试"""
        prompts = ["Hello, my name is"]
        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1

    @require_scene("SINGLECARD")
    def test_temperature_sampling(self):
        """Temperature 采样测试"""
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=10, top_p=0.9)

        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs = runner.generate(prompts, sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("SINGLECARD")
    def test_beam_search(self):
        """Beam Search 测试"""
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10, beam_width=4)

        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs = runner.generate(prompts, sampling_params)

        assert len(outputs) == 1


class TestAutoFitMaxModelLen:
    """自动适配 max_model_len 测试"""

    @require_scene("SINGLECARD")
    def test_auto_fit_memory_singlecard(self):
        """单卡内存自动适配测试"""
        prompts = ["Hello"]

        with VllmRunner(
            "Qwen/Qwen3-8B",
            max_model_len=-1,
            gpu_memory_utilization=0.5,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1

    @require_scene("MULTICARD_2Cards")
    def test_auto_fit_memory_multicard(self):
        """多卡内存自动适配测试"""
        prompts = ["Hello"]

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=-1,
            gpu_memory_utilization=0.5,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1


class TestModelCombinations:
    """模型组合测试 - 使用 scene_aware_parametrize"""

    @scene_aware_parametrize("model", [
        "Qwen/Qwen3-8B",
        "vllm-ascend/Qwen3-0.6B-W8A8",
    ])
    def test_model_selection_singlecard(self, model):
        """单卡场景下的模型选择测试"""
        tp_size = 1
        quantization = "ascend" if "W8A8" in model else None

        with VllmRunner(
            model,
            tensor_parallel_size=tp_size,
            max_model_len=512,
            quantization=quantization,
        ) as runner:
            outputs = runner.generate_greedy(["Hello"], max_tokens=10)

        assert len(outputs) == 1

    @scene_aware_parametrize("model", [
        "Qwen/Qwen3-30B-A3B",
        "vllm-ascend/Qwen3-30B-A3B-W8A8",
    ])
    def test_model_selection_2cards(self, model):
        """2卡场景下的模型选择测试"""
        tp_size = 2
        quantization = "ascend" if "W8A8" in model else None

        with VllmRunner(
            model,
            tensor_parallel_size=tp_size,
            max_model_len=512,
            quantization=quantization,
        ) as runner:
            outputs = runner.generate_greedy(["Hello"], max_tokens=10)

        assert len(outputs) == 1
