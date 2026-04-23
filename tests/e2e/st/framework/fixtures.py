import pytest
import os
from typing import Optional

DEFAULT_SCENE = os.getenv("VLLM_TEST_SCENE", "SINGLECARD")
DEFAULT_MODEL = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-8B")


@pytest.fixture(scope="session", autouse=True)
def setup_test_framework():
    from tests.e2e.st.framework.scene_manager import SceneManager
    from tests.e2e.st.framework.model_config import ModelConfig

    scene_manager = SceneManager.get_instance()
    model_config = ModelConfig.get_instance()

    scene_manager.set_current_scene(DEFAULT_SCENE)
    model_config.current_model = DEFAULT_MODEL

    yield

    SceneManager.reset_instance()
    ModelConfig.reset_instance()


@pytest.fixture
def scene_info():
    manager = SceneManager.get_instance()
    return manager.get_scene_info()


@pytest.fixture
def current_model():
    config = ModelConfig.get_instance()
    return config.current_model


@pytest.fixture
def scene_manager():
    return SceneManager.get_instance()


@pytest.fixture
def model_config():
    return ModelConfig.get_instance()


@pytest.fixture
def vllm_runner_for_scene():
    from tests.e2e.conftest import VllmRunner

    def _create(model_name: str, **kwargs):
        manager = SceneManager.get_instance()
        scene = manager.get_scene_info()

        if scene:
            kwargs.setdefault("tensor_parallel_size", scene.tp_size)

        return VllmRunner(model_name, **kwargs)

    return _create


@pytest.fixture
def model_info():
    def _get(model_name: str = None):
        config = ModelConfig.get_instance()
        name = model_name or config.current_model
        return config.get_model_info(name)
    return _get
