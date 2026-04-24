"""
测试工具函数（参考 vllm/tests/utils.py）

提供常用的测试装饰器和工具函数
"""

import contextlib
import functools
import os
import signal
import tempfile
from collections.abc import Callable
from typing import ParamSpec

import pytest

_P = ParamSpec("_P")


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """
    装饰器：为每个测试函数 fork 新进程（参考 vllm/tests/utils.py）

    用于需要完全隔离的测试场景，避免 GPU 内存泄漏等问题
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        from _pytest.outcomes import Skipped

        with tempfile.NamedTemporaryFile(
            delete=False,
            mode="w+b",
            prefix=f"st_test_{func.__name__}_{os.getpid()}_",
            suffix=".exc",
        ) as exc_file:
            exc_file_path = exc_file.name
            pid = os.fork()

            if pid == 0:
                # 子进程
                os.setpgrp()
                try:
                    func(*args, **kwargs)
                except Skipped:
                    os._exit(0)
                except Exception:
                    import traceback

                    with open(exc_file_path, "w") as f:
                        f.write(traceback.format_exc())
                    os._exit(1)
                else:
                    os._exit(0)
            else:
                # 父进程
                pgid = pid
                _pid, _exitcode = os.waitpid(pid, 0)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGTERM)

                if _exitcode != 0:
                    if os.path.exists(exc_file_path):
                        with open(exc_file_path) as f:
                            tb = f.read()
                        raise AssertionError(
                            f"Test {func.__name__} failed (exit code: {_exitcode}):\n{tb}"
                        )
                    raise AssertionError(
                        f"Test {func.__name__} failed (exit code: {_exitcode})"
                    )

    return wrapper


def multi_gpu_test(num_gpus: int):
    """
    装饰器：仅在满足 GPU 数量要求时执行测试（参考 vllm/tests/utils.py）

    自动跳过 GPU 数量不足的环境，自动 fork 进程隔离
    """

    def decorator(func: Callable[_P, None]) -> Callable[_P, None]:
        marks = [
            pytest.mark.distributed(num_gpus=num_gpus),
            pytest.mark.skipif(
                _get_device_count() < num_gpus,
                reason=f"Need at least {num_gpus} GPUs to run the test.",
            ),
        ]

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
            return func(*args, **kwargs)

        func = fork_new_process_for_each_test()(wrapper)
        for mark in reversed(marks):
            func = mark(func)
        return func

    return decorator


def large_gpu_test(min_gb: int):
    """
    装饰器：跳过 GPU 内存不足的测试（参考 vllm/tests/utils.py）
    """
    try:
        memory_gb = _get_total_memory_gb()
    except Exception:
        memory_gb = 0

    return pytest.mark.skipif(
        memory_gb < min_gb,
        reason=f"Need at least {min_gb}GB GPU memory to run the test.",
    )


def get_vllm_extra_kwargs(
    model_info, vllm_extra_kwargs: dict | None = None
) -> dict:
    """
    获取 vLLM 额外参数（参考 vllm/tests/models/utils.py）

    根据 CI 环境变量和模型信息调整 vLLM 参数
    """
    from .ci_envs import ST_CI_DTYPE, ST_CI_ENFORCE_EAGER, ST_CI_NO_SKIP

    vllm_extra_kwargs = vllm_extra_kwargs or {}

    # 检查是否跳过测试
    if not ST_CI_NO_SKIP and not model_info.enable_test:
        pytest.skip("Skipping test: model disabled in config.")

    # 设置 dtype
    vllm_extra_kwargs["dtype"] = ST_CI_DTYPE or model_info.dtype

    # 设置 enforce_eager
    if ST_CI_ENFORCE_EAGER is not None:
        vllm_extra_kwargs["enforce_eager"] = ST_CI_ENFORCE_EAGER

    return vllm_extra_kwargs


def _get_device_count() -> int:
    """获取可用 NPU 设备数"""
    try:
        import torch_npu

        return torch_npu.npu.device_count()
    except Exception:
        return 0


def _get_total_memory_gb() -> float:
    """获取单卡总内存（GB）"""
    try:
        import torch_npu

        return torch_npu.npu.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return 0
