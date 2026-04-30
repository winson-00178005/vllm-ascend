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

"""Execution environment factory for ST tests.

Provides CPU Mock and NPU Real environment classes.
"""

import time
from unittest.mock import MagicMock, patch

import torch


class ExecutionEnvironment:
    """Base class for execution environments."""

    def __init__(self, mode: str):
        self.mode = mode
        self.mock_patches = []

    def enter(self):
        """Enter environment (start Mock or initialize real environment)."""
        raise NotImplementedError

    def exit(self):
        """Exit environment (cleanup Mock or release real resources)."""
        raise NotImplementedError

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()


class CPUMockEnvironment(ExecutionEnvironment):
    """CPU Mock environment: Mock torch_npu and torch.npu.

    Reference: tests/ut pattern (179 patch calls)
    """

    def __init__(self):
        super().__init__("cpu_mock")
        self.mock_patches = []

    def enter(self):
        """Start Mock patches for NPU operations."""
        try:
            import torch_npu

            def mock_npu_quantize(x):
                return x

            def mock_npu_swiglu(x):
                return x

            self.mock_patches.append(
                patch("torch_npu.npu_quantize", side_effect=mock_npu_quantize))
            self.mock_patches.append(
                patch("torch_npu.npu_swiglu", side_effect=mock_npu_swiglu))

            for p in self.mock_patches:
                p.start()
        except ImportError:
            pass

        try:
            from vllm_ascend.utils import is_310p
            self.mock_patches.append(
                patch("vllm_ascend.utils.is_310p", return_value=False))
            for p in self.mock_patches[-1:]:
                p.start()
        except ImportError:
            pass

    def exit(self):
        """Stop all Mock patches."""
        for p in self.mock_patches:
            p.stop()
        self.mock_patches.clear()


class NPURealEnvironment(ExecutionEnvironment):
    """NPU Real environment: Use real torch_npu operations."""

    def __init__(self, device_id: int = 0):
        super().__init__("npu_real")
        self.device_id = device_id

        try:
            assert torch.npu.is_available(), "NPU not available"
            torch.npu.set_device(device_id)
        except Exception:
            raise RuntimeError("NPU not available for real execution")

    def measure_performance(self, func):
        """Measure performance: execution time, memory, throughput.

        Args:
            func: Function to measure

        Returns:
            dict with performance metrics
        """
        start_time = time.time()
        try:
            start_mem = torch.npu.memory_allocated()
        except Exception:
            start_mem = 0

        result = func()

        end_time = time.time()
        try:
            end_mem = torch.npu.memory_allocated()
        except Exception:
            end_mem = 0

        return {
            "execution_time": end_time - start_time,
            "memory_usage": end_mem - start_mem,
            "result": result
        }

    def measure_precision(self, result_cpu, result_npu):
        """Measure precision: compare CPU and NPU results.

        Args:
            result_cpu: Reference result from CPU
            result_npu: Result from NPU

        Returns:
            bool: True if results are close enough
        """
        return torch.allclose(result_cpu, result_npu, rtol=1e-3, atol=1e-5)

    def exit(self):
        """Cleanup NPU resources."""
        try:
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()
        except Exception:
            pass


def create_environment(mode: str, **kwargs) -> ExecutionEnvironment:
    """Create execution environment based on mode.

    Args:
        mode: "cpu_mock" or "npu_real"
        **kwargs: Additional arguments for environment

    Returns:
        ExecutionEnvironment instance
    """
    if mode == "cpu_mock":
        return CPUMockEnvironment()
    elif mode == "npu_real":
        device_id = kwargs.get("device_id", 0)
        return NPURealEnvironment(device_id=device_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")