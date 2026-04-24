"""
CI 环境变量集中管理（参考 vllm/tests/ci_envs.py）

这些环境变量用于控制测试行为，如：
- 是否跳过某些模型测试
- 强制使用特定 dtype
- 控制 enforce_eager 等
"""

import os
from collections.abc import Callable
from typing import Any


def _maybe_convert_bool(value: str | None) -> bool | None:
    """转换字符串为布尔值"""
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on")


environment_variables: dict[str, Callable[[], Any]] = {
    # 是否测试所有模型（默认只测试 enable_test=true 的模型）
    "ST_CI_NO_SKIP": lambda: bool(int(os.getenv("ST_CI_NO_SKIP", "0"))),
    # 强制使用特定 dtype
    "ST_CI_DTYPE": lambda: os.getenv("ST_CI_DTYPE", None),
    # 是否强制使用 eager 模式
    "ST_CI_ENFORCE_EAGER": lambda: _maybe_convert_bool(
        os.getenv("ST_CI_ENFORCE_EAGER", None)
    ),
    # 目标测试套件（用于跳过不匹配的测试）
    "ST_CI_TARGET_SUITE": lambda: os.getenv("ST_CI_TARGET_SUITE", None),
}


def __getattr__(name: str):
    """延迟求值环境变量"""
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """检查环境变量是否显式设置"""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
