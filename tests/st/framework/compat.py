import pytest
from typing import List, Optional, Callable
from functools import wraps

from tests.st.framework.scene_manager import SceneManager
from tests.st.framework.model_config import ModelConfig


class SceneAwareParametrize:
    def __init__(self, param_name: str, model_list: List[str]):
        self.param_name = param_name
        self.model_list = model_list

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        scene_manager = SceneManager.get_instance()
        current_scene = scene_manager.current_scene

        supported_models = []
        model_config = ModelConfig.get_instance()

        for model in self.model_list:
            if model_config.is_model_supported(model, current_scene):
                supported_models.append(model)

        if not supported_models:
            pytest.skip(f"No models from {self.model_list} supported in scene {current_scene}")

        return pytest.mark.parametrize(self.param_name, supported_models)(wrapper)


def scene_aware_parametrize(param_name: str, model_list: List[str]):
    return SceneAwareParametrize(param_name, model_list)


class ModelFromConfigParametrize:
    def __init__(self, param_name: str, group: Optional[str] = None):
        self.param_name = param_name
        self.group = group

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        scene_manager = SceneManager.get_instance()
        current_scene = scene_manager.current_scene
        model_config = ModelConfig.get_instance()

        if self.group:
            models = model_config.model_groups.get(self.group, [])
        else:
            models = model_config.all_models

        supported = [m for m in models if current_scene in m.supported_scenes]

        if not supported:
            pytest.skip(f"No models in group '{self.group}' supported for scene {current_scene}")

        model_names = [m.name for m in supported]
        return pytest.mark.parametrize(self.param_name, model_names)(wrapper)


def model_from_config(param_name: str, group: Optional[str] = None):
    return ModelFromConfigParametrize(param_name, group)


def skip_if_model_not_supported(model: str, scene: Optional[str] = None):
    scene_manager = SceneManager.get_instance()
    model_config = ModelConfig.get_instance()

    current_scene = scene or scene_manager.current_scene
    if not model_config.is_model_supported(model, current_scene):
        pytest.skip(f"Model {model} not supported in scene {current_scene}")


def get_models_for_current_scene(model_names: List[str]) -> List[str]:
    scene_manager = SceneManager.get_instance()
    current_scene = scene_manager.current_scene
    model_config = ModelConfig.get_instance()

    supported = []
    for model in model_names:
        if model_config.is_model_supported(model, current_scene):
            supported.append(model)

    return supported
