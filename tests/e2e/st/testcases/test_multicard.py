#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

import os
import pytest
from unittest.mock import patch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from tests.e2e.conftest import VllmRunner
from tests.e2e.st.framework.decorators import require_scene


class TestDataParallel:
    """Data Parallel 测试"""

    @require_scene("MULTICARD_2Cards")
    def test_data_parallel_basic(self):
        """Data Parallel 基础测试"""
        prompts = ["Hello, my name is"] * 4
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 4

    @require_scene("MULTICARD_4Cards")
    def test_data_parallel_dp2(self):
        """Data Parallel DP2 测试"""
        prompts = ["Hello, my name is"] * 2
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=2,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 2


class TestExpertParallel:
    """Expert Parallel 测试"""

    @require_scene("MULTICARD_2Cards")
    def test_expert_parallel_tp2(self):
        """Expert Parallel TP2 测试"""
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
    def test_expert_parallel_tp4(self):
        """Expert Parallel TP4 测试"""
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


class TestSequenceParallelism:
    """Sequence Parallelism 测试"""

    @require_scene("MULTICARD_2Cards")
    def test_sequence_parallelism(self):
        """Sequence Parallelism 测试"""
        prompts = ["The capital of France is a beautiful city" * 20]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=2048,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1


class TestDistributedExecution:
    """分布式执行测试"""

    @require_scene("MULTICARD_2Cards")
    @patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
    def test_distributed_mp_backend(self):
        """多进程分布式后端测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1

    @require_scene("MULTICARD_4Cards")
    @patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
    def test_distributed_mp_backend_tp4(self):
        """多进程分布式后端 TP4 测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1


class TestFullGraphMode:
    """Full Graph 模式测试"""

    @require_scene("MULTICARD_2Cards")
    def test_full_graph_mode(self):
        """Full Graph 模式测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enforce_eager=False,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0

    @require_scene("MULTICARD_4Cards")
    def test_full_graph_mode_tp4(self):
        """Full Graph 模式 TP4 测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            enforce_eager=False,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
        assert len(outputs[0][0]) > 0


class TestOfflineInferenceDistributed:
    """分布式离线推理测试"""

    @require_scene("MULTICARD_2Cards")
    def test_offline_distributed_basic(self):
        """分布式离线推理基础测试"""
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
            "Machine learning is",
        ]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 4

    @require_scene("MULTICARD_4Cards")
    def test_offline_distributed_large_batch(self):
        """分布式离线推理大批量测试"""
        prompts = [f"Test prompt number {i}" for i in range(8)]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            tensor_parallel_size=4,
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 8


class TestACLGraphCaptureReplay:
    """ACL Graph Capture/Replay 测试"""

    @require_scene("MULTICARD_2Cards")
    @patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
    def test_aclgraph_capture_replay(self):
        """ACL Graph Capture Replay 测试"""
        prompts = ["Hello, my name is"]
        max_tokens = 10

        with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enforce_eager=False,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_model_len=512,
        ) as runner:
            outputs = runner.generate_greedy(prompts, max_tokens)

        assert len(outputs) == 1
