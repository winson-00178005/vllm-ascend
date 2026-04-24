from typing import List, Callable
import pytest
from functools import wraps


def require_scene(*scenes: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _check_scene_match(list(scenes)):
                pytest.skip(f"Test requires scene {scenes}, current scene does not match")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_model(*models: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _check_model_match(list(models)):
                pytest.skip(f"Test requires model {models}, current model does not match")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_scene_and_model(scene: str, model: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            scene_ok = _check_scene_match([scene])
            model_ok = _check_model_match([model])

            if not scene_ok:
                pytest.skip(f"Test requires scene {scene}")
            if not model_ok:
                pytest.skip(f"Test requires model {model}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_scene_match(scenes: List[str]) -> bool:
    from tests.st.framework.scene_manager import SceneManager
    manager = SceneManager.get_instance()
    return manager.is_scene_match(scenes)


def _check_model_match(models: List[str]) -> bool:
    from tests.st.framework.model_config import ModelConfig
    config = ModelConfig.get_instance()
    current = getattr(config, 'current_model', None)
    if current is None:
        return True
    return current in models
