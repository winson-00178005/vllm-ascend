#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 平台检测与资源管理模块

基于上游 vLLM tests/utils.py 移植并针对 NPU 平台优化:
1. NPU 平台检测
2. 硬件资源标记 (@large_memory_test, @npu_device_test)
3. GPU/NPU 内存追踪
4. 多设备管理
"""

import os
import time
import functools
from typing import Callable, TypeVar, Optional, Literal
from dataclasses import dataclass

import pytest


@dataclass
class DeviceMemoryInfo:
    """设备内存信息"""
    device_id: int
    total_gb: float
    used_gb: float
    free_gb: float
    usage_ratio: float


class NPUPlatform:
    """NPU 平台检测与管理"""

    @staticmethod
    def is_available() -> bool:
        """检查 NPU 是否可用"""
        try:
            import torch
            return hasattr(torch, 'npu') and torch.npu.is_available()
        except (ImportError, AttributeError):
            return False

    @staticmethod
    def device_count() -> int:
        """获取可用 NPU 设备数量"""
        try:
            import torch
            if NPUPlatform.is_available():
                return torch.npu.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        """获取设备名称"""
        try:
            import torch
            if NPUPlatform.is_available():
                return torch.npu.get_device_name(device_id)
        except Exception:
            pass
        return "Unknown"

    @staticmethod
    def get_hardware_type() -> str:
        """获取硬件类型 (Ascend910B, Ascend910C, Ascend310P 等)"""
        device_name = NPUPlatform.get_device_name()
        if "910B" in device_name or "Ascend" in device_name and "B" in device_name:
            return "Ascend910B"
        elif "910C" in device_name or "Ascend" in device_name and "C" in device_name:
            return "Ascend910C"
        elif "310P" in device_name or "Ascend" in device_name and "P" in device_name:
            return "Ascend310P"
        elif "NPU" in device_name:
            return "NPU"
        return "Unknown"

    @staticmethod
    def get_memory_info(device_id: int = 0) -> Optional[DeviceMemoryInfo]:
        """获取设备内存信息"""
        try:
            import torch
            if not NPUPlatform.is_available():
                return None

            mem_allocated = torch.npu.memory_allocated(device_id) / (1024 ** 3)
            mem_reserved = torch.npu.memory_reserved(device_id) / (1024 ** 3)

            total_str = torch.npu.get_device_properties(device_id).total_memory / (1024 ** 3)

            return DeviceMemoryInfo(
                device_id=device_id,
                total_gb=total_str,
                used_gb=mem_allocated,
                free_gb=total_str - mem_allocated,
                usage_ratio=mem_allocated / total_str if total_str > 0 else 0
            )
        except Exception:
            return None


def wait_for_npu_memory_to_clear(
    devices: Optional[list[int]] = None,
    threshold_ratio: float = 0.1,
    timeout_s: float = 120.0,
) -> None:
    """等待 NPU 内存释放到指定阈值

    基于上游 vLLM wait_for_gpu_memory_to_clear 移植

    Args:
        devices: 要检查的设备列表，默认所有可用设备
        threshold_ratio: 内存使用比例阈值，低于此值认为内存已释放
        timeout_s: 超时时间（秒）
    """
    if not NPUPlatform.is_available():
        return

    if devices is None:
        devices = list(range(NPUPlatform.device_count()))

    start_time = time.time()

    while True:
        all_free = True
        for device_id in devices:
            mem_info = NPUPlatform.get_memory_info(device_id)
            if mem_info is None:
                continue

            print(f"[NPU {device_id}] memory: {mem_info.used_gb:.2f}/{mem_info.total_gb:.2f} GB "
                  f"({mem_info.usage_ratio:.1%})")

            if mem_info.usage_ratio > threshold_ratio:
                all_free = False

        if all_free:
            elapsed = time.time() - start_time
            print(f"[NPU] All devices memory below threshold {threshold_ratio:.1%} "
                  f"after {elapsed:.1f}s")
            return

        elapsed = time.time() - start_time
        if elapsed >= timeout_s:
            raise RuntimeError(
                f"NPU memory did not clear within {timeout_s}s. "
                f"Devices: {devices}, threshold: {threshold_ratio:.1%}"
            )

        time.sleep(5)


_F = TypeVar("_F")


def npu_memory_required(min_gb: float) -> Callable[[_F], _F]:
    """装饰器：要求最小 NPU 内存

    用法:
    @npu_memory_required(min_gb=40)
    def test_large_model():
        ...
    """
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not NPUPlatform.is_available():
                pytest.skip("NPU not available")

            total_mem = 0
            for i in range(NPUPlatform.device_count()):
                mem_info = NPUPlatform.get_memory_info(i)
                if mem_info:
                    total_mem += mem_info.total_gb

            if total_mem < min_gb:
                pytest.skip(f"Need at least {min_gb}GB NPU memory (have {total_mem:.1f}GB)")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def npu_device_required(num_devices: int = 1) -> Callable[[_F], _F]:
    """装饰器：要求最小 NPU 设备数量

    用法:
    @npu_device_required(num_devices=2)
    def test_multicard():
        ...
    """
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not NPUPlatform.is_available():
                pytest.skip("NPU not available")

            device_count = NPUPlatform.device_count()
            if device_count < num_devices:
                pytest.skip(f"Need at least {num_devices} NPU devices (have {device_count})")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def large_npu_test(min_gb: float = 40):
    """装饰器：要求大内存的 NPU 测试

    用法:
    @large_npu_test(min_gb=80)
    def test_large_model():
        ...
    """
    return npu_memory_required(min_gb)


def single_npu_only(func: _F) -> _F:
    """装饰器：仅在单卡环境运行

    用法:
    @single_npu_only
    def test_single_card():
        ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not NPUPlatform.is_available():
            pytest.skip("NPU not available")

        device_count = NPUPlatform.device_count()
        if device_count > 1:
            pytest.skip(f"Single NPU test (have {device_count} devices)")

        return func(*args, **kwargs)
    return wrapper


def multi_npu_only(num_gpus: int = 2):
    """装饰器：仅在多卡环境运行

    用法:
    @multi_npu_only(num_gpus=4)
    def test_4card():
        ...
    """
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not NPUPlatform.is_available():
                pytest.skip("NPU not available")

            device_count = NPUPlatform.device_count()
            if device_count < num_gpus:
                pytest.skip(f"Need at least {num_gpus} NPU devices (have {device_count})")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def npu_tier_mark(
    min_devices: int = 1,
    max_devices: Optional[int] = None,
    min_memory_gb: float = 0
) -> pytest.MarkDecorator:
    """生成 NPU 层级标记

    用法:
    @npu_tier_mark(min_devices=2, max_devices=4)
    def test_2to4_cards():
        ...

    @npu_tier_mark(min_memory_gb=80)
    def test_large_memory():
        ...
    """
    marks = []

    if not NPUPlatform.is_available():
        marks.append(pytest.mark.skip("NPU not available"))
        return marks[0] if len(marks) == 1 else pytest.mark.skip("NPU not available")

    device_count = NPUPlatform.device_count()

    reasons = []
    if device_count < min_devices:
        reasons.append(f"Need at least {min_devices} NPU devices (have {device_count})")
    if max_devices is not None and device_count > max_devices:
        reasons.append(f"Need at most {max_devices} NPU devices (have {device_count})")

    if min_memory_gb > 0:
        total_mem = 0
        for i in range(device_count):
            mem_info = NPUPlatform.get_memory_info(i)
            if mem_info:
                total_mem += mem_info.total_gb
        if total_mem < min_memory_gb:
            reasons.append(f"Need at least {min_memory_gb}GB total memory (have {total_mem:.1f}GB)")

    if reasons:
        marks.append(pytest.mark.skip(reason="; ".join(reasons)))

    return marks[0] if len(marks) == 1 else pytest.mark.skip(reason="; ".join(reasons))


class NPUResourceGuard:
    """NPU 资源守卫 - 确保测试前后资源状态一致

    用法:
    def test_something():
        guard = NPUResourceGuard()
        guard.ensure_memory_free()
        # ... 执行测试 ...
        guard.assert_memory_released()
    """

    def __init__(self, devices: Optional[list[int]] = None):
        self.devices = devices or list(range(NPUPlatform.device_count()))
        self.baseline_memory: dict[int, float] = {}

    def record_baseline(self):
        """记录当前内存状态作为基准"""
        self.baseline_memory = {}
        for device_id in self.devices:
            mem_info = NPUPlatform.get_memory_info(device_id)
            if mem_info:
                self.baseline_memory[device_id] = mem_info.used_gb

    def ensure_memory_free(self, threshold_gb: float = 2.0):
        """确保有足够可用内存"""
        for device_id in self.devices:
            mem_info = NPUPlatform.get_memory_info(device_id)
            if mem_info and mem_info.free_gb < threshold_gb:
                pytest.skip(f"NPU {device_id} memory low: {mem_info.free_gb:.1f}GB free")

    def assert_memory_released(self, tolerance_gb: float = 1.0):
        """断言测试后内存已释放"""
        for device_id in self.devices:
            mem_info = NPUPlatform.get_memory_info(device_id)
            if not mem_info:
                continue

            baseline = self.baseline_memory.get(device_id, 0)
            current = mem_info.used_gb

            if current > baseline + tolerance_gb:
                raise AssertionError(
                    f"NPU {device_id} memory not released: "
                    f"baseline={baseline:.2f}GB, current={current:.2f}GB, "
                    f"leaked={current - baseline:.2f}GB"
                )


@pytest.fixture
def npu_platform():
    """NPU 平台信息 fixture"""
    return NPUPlatform


@pytest.fixture
def npu_memory_guard():
    """NPU 内存守卫 fixture"""
    guard = NPUResourceGuard()
    guard.record_baseline()
    return guard


@pytest.fixture
def wait_for_npu_memory():
    """等待 NPU 内存释放 fixture"""
    return wait_for_npu_memory_to_clear
