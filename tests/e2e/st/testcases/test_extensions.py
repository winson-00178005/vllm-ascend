#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 扩展功能示例测试

展示如何使用扩展功能:
1. vllm_version_is 版本跳过
2. PerformanceMonitor 性能监控
3. PrecisionComparator 精度对比
4. require_hardware 硬件要求
"""

import os
import pytest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from tests.e2e.conftest import VllmRunner
from tests.e2e.st.framework.decorators import require_scene
from tests.e2e.st.framework.extensions import (
    vllm_version_is,
    PerformanceMonitor,
    PrecisionComparator,
    require_hardware,
    require_npu_device,
    TestReport,
)


class TestVersionSkipping:
    """版本跳过机制测试"""

    @require_scene("SINGLECARD")
    @pytest.mark.skipif(vllm_version_is("0.19.0"), reason="Not supported in v0.19.0")
    def test_version_specific_feature(self):
        """仅在特定版本执行的测试"""
        assert True

    @require_scene("SINGLECARD")
    def test_normal_after_version_check(self):
        """经过版本检查后正常执行的测试"""
        assert True


class TestPerformanceMonitoring:
    """性能监控测试"""

    @require_scene("SINGLECARD")
    def test_performance_baseline(self, performance_monitor):
        """性能基准测试"""
        prompts = ["Hello, my name is"] * 10
        max_tokens = 20

        with performance_monitor.start("baseline_test"):
            with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
                outputs = runner.generate_greedy(prompts, max_tokens)

        result = performance_monitor.get_result("baseline_test")
        assert result is not None
        print(f"\nPerformance: {result.throughput_tokens_per_sec:.2f} tokens/s")

    @require_scene("SINGLECARD")
    def test_performance_with_assertion(self, performance_monitor):
        """带性能断言的测试"""
        prompts = ["The capital of France is"] * 5
        max_tokens = 10

        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            with performance_monitor.start("throughput_test"):
                outputs = runner.generate_greedy(prompts, max_tokens)

        performance_monitor.assert_performance(
            "throughput_test",
            min_throughput=0.1
        )


class TestPrecisionComparison:
    """精度对比测试"""

    @require_scene("SINGLECARD")
    def test_precision_comparison(self, precision_comparator):
        """NPU 输出精度对比测试"""
        prompts = ["Hello, world"]

        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs1 = runner.generate_greedy(prompts, max_tokens=10)
            outputs2 = runner.generate_greedy(prompts, max_tokens=10)

        output_text1 = outputs1[0][1]
        output_text2 = outputs2[0][1]

        is_close = precision_comparator.compare(output_text1, output_text2)
        precision_comparator.record("same_prompt", output_text1, output_text2, is_close)

        summary = precision_comparator.get_summary()
        print(f"\nPrecision comparison: {summary}")

    @require_scene("SINGLECARD")
    def test_deterministic_output(self):
        """确定性输出测试"""
        prompt = ["1+1="]
        max_tokens = 5

        with VllmRunner("Qwen/Qwen3-8B", max_model_len=512) as runner:
            outputs = [runner.generate_greedy(prompt, max_tokens) for _ in range(3)]

        texts = [out[0][1] for out in outputs]
        assert texts[0] == texts[1] == texts[2], f"Non-deterministic output: {texts}"


class TestHardwareRequirements:
    """硬件要求测试"""

    @require_scene("SINGLECARD")
    @require_hardware("Ascend910B")
    def test_910b_specific(self):
        """仅在 910B 上执行的测试"""
        assert True

    @require_scene("SINGLECARD")
    @require_npu_device(0)
    def test_single_npu(self):
        """要求单卡 NPU"""
        assert True


class TestReportGeneration:
    """测试报告生成"""

    @require_scene("SINGLECARD")
    def test_with_report(self, test_report):
        """生成测试报告"""
        test_report.set_metadata(
            scene="SINGLECARD",
            model="Qwen/Qwen3-8B",
            version="0.9.1"
        )

        test_report.add_test("test_case_1", "PASSED", duration=1.5)
        test_report.add_test("test_case_2", "PASSED", duration=2.0)

        summary = test_report.get_summary()
        print(f"\nTest Summary: {summary}")
        assert summary["total"] == 2
