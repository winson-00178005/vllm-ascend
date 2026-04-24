"""
@npu_test 装饰器 - 标记测试的 NPU 资源需求

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
1. CI 路由 - AST 解析器读取装饰器参数，分组测试到 runner
2. 运行时跳过 - 当环境缺少所需 NPU 资源时自动跳过测试

与 UT 框架统一:
- tests/ut/conftest.py::npu_test
- tests/st/framework/npu_test.py::npu_test
"""
import functools
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Any
import pytest
import logging

from .device_types import RunnerDeviceType, RunnerKey

logger = logging.getLogger(__name__)

try:
    subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    _npu_available = True
except (subprocess.CalledProcessError, FileNotFoundError):
    _npu_available = False


@dataclass
class NpuTestInfo:
    """NPU 测试信息（用于 CI 路由）"""
    num_npus: int
    npu_type: RunnerDeviceType
    
    @property
    def runner_key(self) -> RunnerKey:
        """获取 runner key"""
        return (self.num_npus, self.npu_type)


def npu_test(
    num_npus: int = 1,
    npu_type: RunnerDeviceType = RunnerDeviceType.A2,
) -> Callable:
    """
    装饰器 - 标记测试的 NPU 资源需求
    
    功能:
    1. CI 路由 - AST 解析器读取参数分组测试
    2. 运行时跳过 - 资源不足时自动跳过
    
    Args:
        num_npus: NPU 数量（默认 1）
        npu_type: NPU 类型（默认 A2/910B）
    
    用法:
        @npu_test(num_npus=2, npu_type=RunnerDeviceType._310P)
        def test_qwen3_tp2():
            # 需要 2 张 310P
            pass
        
        @npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
        def test_qwen3_single():
            # 需要 1 张 910B
            pass
        
        @npu_test(npu_type=RunnerDeviceType.CPU)
        def test_cpu_only():
            # CPU 测试（不需要 NPU）
            pass
    
    注意:
        - 装饰器参数名必须与 determine_st_scope.py AST 解析器同步
        - npu_type 支持字符串或枚举值
    """
    if not isinstance(npu_type, RunnerDeviceType):
        npu_type = RunnerDeviceType(npu_type)
    
    def decorator(func: Callable) -> Callable:
        # 存储 NPU 测试信息（用于 CI 路由）
        func._npu_test_info = NpuTestInfo(num_npus=num_npus, npu_type=npu_type)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # CPU 类型 - 无需检查 NPU
            if npu_type == RunnerDeviceType.CPU:
                return func(*args, **kwargs)
            
            # 运行时检查 NPU 可用性
            if not _npu_available:
                pytest.skip(f"NPU not available (need {npu_type.display_name} x{num_npus})")
                return None
            
            import torch
            device_count = torch.npu.device_count()
            if device_count < num_npus:
                pytest.skip(f"Not enough NPUs: need {num_npus}, have {device_count}")
                return None
            
            # 执行测试
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_npu_test_info(func: Callable) -> Optional[NpuTestInfo]:
    """
    获取函数的 NPU 测试信息
    
    Args:
        func: 测试函数
    
    Returns:
        NpuTestInfo 或 None（无 @npu_test 装饰器）
    """
    return getattr(func, '_npu_test_info', None)


def is_npu_test(func: Callable) -> bool:
    """
    检查函数是否标记了 @npu_test
    
    Args:
        func: 测试函数
    
    Returns:
        是否有 @npu_test 装饰器
    """
    return hasattr(func, '_npu_test_info')


__all__ = [
    "npu_test",
    "NpuTestInfo",
    "get_npu_test_info",
    "is_npu_test",
]