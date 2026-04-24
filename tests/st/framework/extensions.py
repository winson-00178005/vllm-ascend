#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 扩展功能 - 基于 PR #8557 单元测试框架优化

增强功能:
1. 版本跳过机制 - 支持 vllm_version_is 条件
2. 性能基准测试 - 支持性能回归检�?3. 精度对比测试 - 支持 NPU vs CPU 精度对比
4. 环境变量配置增强 - 支持更多配置选项
"""

import os
import time
import pytest
from typing import Optional, List, Callable, Any, Dict
from dataclasses import dataclass, field


def vllm_version_is(version: str) -> bool:
    """
    检查当�?vLLM 版本是否匹配指定版本

    用法:
    @pytest.mark.skipif(vllm_version_is("0.19.0"), reason="Not supported in this version")
    def test_xxx():
        ...
    """
    try:
        import vllm
        current_version = getattr(vllm, "__version__", "0.0.0")
        return current_version == version
    except ImportError:
        return False


def vllm_version_above(version: str) -> bool:
    """检查当�?vLLM 版本是否高于指定版本"""
    try:
        import vllm
        current_version = getattr(vllm, "__version__", "0.0.0")
        from packaging import version
        return version.parse(current_version) >= version.parse(version)
    except ImportError:
        return False


def vllm_version_below(version: str) -> bool:
    """检查当�?vLLM 版本是否低于指定版本"""
    try:
        import vllm
        current_version = getattr(vllm, "__version__", "0.0.0")
        from packaging import version
        return version.parse(current_version) < version.parse(version)
    except ImportError:
        return False


@dataclass
class PerformanceBenchmark:
    """性能基准测试结果"""
    name: str
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_used_gb: float
    timestamp: str = ""


class PerformanceMonitor:
    """
    性能监控�?- 用于性能基准测试

    用法:
    def test_throughput():
        monitor = PerformanceMonitor()
        with monitor.start("test_name"):
            # 执行推理
            outputs = runner.generate(prompts, max_tokens)
        result = monitor.get_result("test_name")
        assert result.throughput_tokens_per_sec > baseline
    """

    def __init__(self):
        self.results: Dict[str, PerformanceBenchmark] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        """开始监�?""
        self._start_times[name] = time.time()

    def stop(self, name: str, token_count: int = 0):
        """停止监控并记录结�?""
        if name not in self._start_times:
            return

        elapsed = time.time() - self._start_times[name]
        latency_ms = elapsed * 1000
        throughput = token_count / elapsed if elapsed > 0 else 0

        import datetime
        self.results[name] = PerformanceBenchmark(
            name=name,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
            memory_used_gb=self._get_memory_usage(),
            timestamp=datetime.datetime.now().isoformat()
        )
        del self._start_times[name]

    def _get_memory_usage(self) -> float:
        """获取当前内存使用 (GB)"""
        try:
            import torch
            if torch.npu.is_available():
                mem_allocated = torch.npu.memory_allocated() / (1024**3)
                return mem_allocated
        except Exception:
            pass
        return 0.0

    def get_result(self, name: str) -> Optional[PerformanceBenchmark]:
        return self.results.get(name)

    def get_all_results(self) -> Dict[str, PerformanceBenchmark]:
        return self.results.copy()

    def assert_performance(self, name: str, min_throughput: float = None, max_latency_ms: float = None):
        """断言性能指标"""
        result = self.get_result(name)
        if result is None:
            pytest.fail(f"No performance data for {name}")

        if min_throughput is not None:
            assert result.throughput_tokens_per_sec >= min_throughput, \
                f"Throughput {result.throughput_tokens_per_sec:.2f} tokens/s < {min_throughput:.2f}"

        if max_latency_ms is not None:
            assert result.latency_ms <= max_latency_ms, \
                f"Latency {result.latency_ms:.2f}ms > {max_latency_ms:.2f}ms"


class Context:
    """性能监控上下文管理器"""

    def __init__(self, monitor: PerformanceMonitor, name: str, token_count: int = 0):
        self.monitor = monitor
        self.name = name
        self.token_count = token_count

    def __enter__(self):
        self.monitor.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop(self.name, self.token_count)


@pytest.fixture
def performance_monitor():
    """性能监控�?fixture"""
    return PerformanceMonitor()


class PrecisionComparator:
    """
    精度对比�?- 用于 NPU vs CPU 精度对比测试

    用法:
    def test_precision():
        comparator = PrecisionComparator(tolerance=1e-3)

        # CPU 输出
        cpu_output = cpu_runner.generate(["Hello"])
        # NPU 输出
        npu_output = npu_runner.generate(["Hello"])

        is_close = comparator.compare(cpu_output, npu_output)
        assert is_close, f"Outputs differ beyond tolerance"
    """

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.results: List[Dict[str, Any]] = []

    def compare(self, cpu_result: Any, npu_result: Any) -> bool:
        """比较两个结果是否在容差范围内"""
        try:
            if isinstance(cpu_result, (list, tuple)) and isinstance(npu_result, (list, tuple)):
                for c, n in zip(cpu_result, npu_result):
                    if not self._compare_values(c, n):
                        return False
                return True
            return self._compare_values(cpu_result, npu_result)
        except Exception:
            return False

    def _compare_values(self, cpu_val: Any, npu_val: Any) -> bool:
        """比较单个�?""
        if isinstance(cpu_val, str):
            return cpu_val == npu_val
        if hasattr(cpu_val, '__iter__') and hasattr(npu_val, '__iter__'):
            try:
                import numpy as np
                diff = np.abs(np.array(cpu_val) - np.array(npu_val))
                return float(np.max(diff)) < self.tolerance
            except Exception:
                return cpu_val == npu_val
        try:
            import numpy as np
            diff = np.abs(float(cpu_val) - float(npu_val))
            return diff < self.tolerance
        except (ValueError, TypeError):
            return cpu_val == npu_val

    def record(self, test_name: str, cpu_result: Any, npu_result: Any, is_close: bool):
        """记录对比结果"""
        self.results.append({
            "test_name": test_name,
            "cpu_result": str(cpu_result)[:100],
            "npu_result": str(npu_result)[:100],
            "is_close": is_close,
            "tolerance": self.tolerance
        })

    def get_summary(self) -> Dict[str, int]:
        """获取对比结果汇�?""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["is_close"])
        failed = total - passed
        return {"total": total, "passed": passed, "failed": failed}


@pytest.fixture
def precision_comparator():
    """精度对比�?fixture"""
    return PrecisionComparator()


def require_hardware(hardware: str):
    """
    装饰�? 要求特定硬件类型

    用法:
    @require_hardware("Ascend910B")
    def test_910b_only():
        ...
    """
    def decorator(func):
        @pytest.mark.skipif(
            not _check_hardware(hardware),
            reason=f"Test requires {hardware} hardware"
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def _check_hardware(required: str) -> bool:
    """检查硬件类�?""
    try:
        import os
        hw_info = os.environ.get("ASCEND_HARDWARE_TYPE", "")
        return required in hw_info
    except Exception:
        return True


def require_npu_device(device_id: Optional[int] = None):
    """
    装饰�? 要求 NPU 设备可用

    用法:
    @require_npu_device(0)
    def test_single_npu():
        ...
    """
    def decorator(func):
        @pytest.mark.skipif(
            not _check_npu_available(device_id),
            reason=f"NPU device {device_id} not available"
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def _check_npu_available(device_id: Optional[int] = None) -> bool:
    """检�?NPU 是否可用"""
    try:
        import torch
        if not hasattr(torch, 'npu') and not hasattr(torch, 'cuda'):
            return False
        if device_id is not None:
            return torch.npu.is_available() and torch.npu.device_count() > device_id
        return torch.npu.is_available()
    except Exception:
        return False


class TestReport:
    """
    测试报告生成�?
    用法:
    report = TestReport()
    report.add_test("test_name", "PASSED", duration=1.5)
    report.add_test("test_name2", "FAILED", error="AssertionError")
    report.save("test_report.json")
    """

    def __init__(self):
        self.tests: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def set_metadata(self, scene: str, model: str, version: str):
        """设置测试元数�?""
        self.metadata = {
            "scene": scene,
            "model": model,
            "vllm_version": version,
        }

    def add_test(self, name: str, status: str, duration: float = 0,
                 error: Optional[str] = None, metrics: Optional[Dict] = None):
        """添加测试结果"""
        self.tests.append({
            "name": name,
            "status": status,
            "duration_sec": duration,
            "error": error,
            "metrics": metrics or {}
        })

    def save(self, filepath: str):
        """保存报告到文�?""
        import json
        report = {
            "metadata": self.metadata,
            "tests": self.tests,
            "summary": self.get_summary()
        }
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def get_summary(self) -> Dict[str, int]:
        """获取测试汇�?""
        passed = sum(1 for t in self.tests if t["status"] == "PASSED")
        failed = sum(1 for t in self.tests if t["status"] == "FAILED")
        skipped = sum(1 for t in self.tests if t["status"] == "SKIPPED")
        return {
            "total": len(self.tests),
            "passed": passed,
            "failed": failed,
            "skipped": skipped
        }


@pytest.fixture
def test_report():
    """测试报告 fixture"""
    return TestReport()
