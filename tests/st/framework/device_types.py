"""
设备类型定义 - 与 UT 框架和 runner_label.json 统一

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

设备类型枚举，用于：
- tests/st/conftest.py (npu_test 装饰器)
- tests/st/framework/npu_test.py (装饰器实现)
- scripts/determine_st_scope.py (AST 解析器)
- config/runner_label.json (runner 配置)
"""
from enum import Enum


class RunnerDeviceType(str, Enum):
    """
    芯片类型 - 值必须与 runner_label.json chip 字段匹配
    
    与 UT 框架共享:
    - tests/ut/conftest.py (npu_test 装饰器)
    - .github/workflows/scripts/determine_smart_e2e_scope.py (AST parser)
    
    命名规范：
    - A2 = 910B (Ascend 910B)
    - A3 = 910C (Ascend 910C)
    - _310P = Ascend 310P
    - CPU = 无 NPU（纯 CPU 测试）
    """
    
    A2 = "a2"       # Ascend 910B
    A3 = "a3"       # Ascend 910C
    _310P = "310p"  # Ascend 310P
    CPU = "cpu"     # CPU（无 NPU）
    
    @property
    def is_npu(self) -> bool:
        """是否为 NPU 设备"""
        return self != RunnerDeviceType.CPU
    
    @property
    def display_name(self) -> str:
        """显示名称"""
        names = {
            RunnerDeviceType.A2: "910B",
            RunnerDeviceType.A3: "910C",
            RunnerDeviceType._310P: "310P",
            RunnerDeviceType.CPU: "CPU"
        }
        return names[self]


# 设备类型别名（向后兼容）
DeviceType310P = RunnerDeviceType._310P
DeviceType910B = RunnerDeviceType.A2
DeviceTypeCPU = RunnerDeviceType.CPU


# 默认 runner key（无装饰器的测试）
DEFAULT_RUNNER_KEY = (0, RunnerDeviceType.CPU)


# RunnerKey 类型别名
RunnerKey = tuple[int, RunnerDeviceType]


__all__ = [
    "RunnerDeviceType",
    "DeviceType310P",
    "DeviceType910B",
    "DeviceTypeCPU",
    "DEFAULT_RUNNER_KEY",
    "RunnerKey",
]