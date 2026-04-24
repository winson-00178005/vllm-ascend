"""
Fixture 工厂 - 创建不同生命周期的 vLLM runner fixture

支持三种生命周期：
1. session 级：整个测试会话共享一个 LLM 实例
2. module 级：每个测试模块创建一个 LLM 实例
3. 参数化：通过 pytest 参数化动态创建
"""

import os
from typing import Optional

import pytest


def create_vllm_runner_session_fixture():
    """
    创建 session 级 vllm_runner_session fixture

    整个测试会话共享一个 LLM 实例，适用于轻量级用例
    """

    @pytest.fixture(scope="session")
    def vllm_runner_session(request):
        from .model import ModelManager
        from .scene import SceneManager

        scene_mgr = SceneManager.get_instance()
        model_mgr = ModelManager.get_instance()
        scene_cfg = scene_mgr.get_config()

        # 应用场景环境变量
        scene_mgr.apply_env_vars()

        def _runner(model_name: str, **kwargs):
            model_info = model_mgr.get_info(model_name)
            if not model_info:
                raise ValueError(f"未知模型: {model_name}")

            # 延迟导入，避免在没有 vLLM 的环境中报错
            from tests.e2e.conftest import VllmRunner

            return VllmRunner(
                model_name=model_info.model_id,
                dtype=kwargs.pop("dtype", model_info.dtype),
                max_model_len=kwargs.pop(
                    "max_model_len", model_info.max_model_len or 4096
                ),
                tensor_parallel_size=kwargs.pop(
                    "tensor_parallel_size", scene_cfg.get("tp_size", 1)
                ),
                **kwargs,
            )

        yield _runner

    return vllm_runner_session


def create_vllm_runner_module_fixture():
    """
    创建 module 级 vllm_runner_module fixture

    每个测试模块创建一个 LLM 实例，适用于需要模块隔离的场景
    """

    @pytest.fixture(scope="module")
    def vllm_runner_module(request):
        from .model import ModelManager
        from .scene import SceneManager

        scene_mgr = SceneManager.get_instance()
        model_mgr = ModelManager.get_instance()
        scene_cfg = scene_mgr.get_config()

        # 应用场景环境变量
        scene_mgr.apply_env_vars()

        # 从参数获取模型名称
        model_name = getattr(request, "param", None)
        if not model_name:
            pytest.skip("未指定模型参数")

        model_info = model_mgr.get_info(model_name)
        if not model_info:
            pytest.skip(f"未知模型: {model_name}")

        # 延迟导入
        from tests.e2e.conftest import VllmRunner

        runner = VllmRunner(
            model_name=model_info.model_id,
            dtype=model_info.dtype,
            max_model_len=model_info.max_model_len or 4096,
            tensor_parallel_size=scene_cfg.get("tp_size", 1),
        )

        yield runner

        # 清理资源
        try:
            runner.model = None
        except Exception:
            pass

    return vllm_runner_module


def create_hf_runner_session_fixture():
    """
    创建 session 级 hf_runner_session fixture

    用于 HuggingFace 模型对比测试
    """

    @pytest.fixture(scope="session")
    def hf_runner_session(request):
        from tests.e2e.conftest import HfRunner

        def _runner(model_name: str, **kwargs):
            return HfRunner(model_name=model_name, **kwargs)

        yield _runner

    return hf_runner_session
