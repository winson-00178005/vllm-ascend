from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
import os
from pathlib import Path


@dataclass
class SceneInfo:
    name: str
    hardware: str
    tp_size: int
    dp_size: int
    description: str
    env: Dict[str, str] = field(default_factory=dict)


class SceneManager:
    _instance: Optional["SceneManager"] = None

    def __init__(self, config_path: str):
        self.scenes: dict[str, SceneInfo] = {}
        self.current_scene: Optional[str] = None
        self._original_env: Dict[str, str] = {}
        self._load_config(config_path)

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "SceneManager":
        if cls._instance is None:
            path = config_path or cls._get_default_config_path()
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def _get_default_config_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "config", "scene.yaml")

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._restore_env()
        cls._instance = None

    def _load_config(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for scene_id, info in config["scenarios"].items():
            self.scenes[scene_id] = SceneInfo(
                name=info["name"],
                hardware=info["hardware"],
                tp_size=info["tp_size"],
                dp_size=info["dp_size"],
                description=info.get("description", ""),
                env=info.get("env", {})
            )

    def set_current_scene(self, scene_id: str) -> bool:
        if scene_id not in self.scenes:
            return False

        self._restore_env()
        self.current_scene = scene_id
        self._apply_env()
        return True

    def _apply_env(self):
        if self.current_scene is None:
            return
        scene = self.scenes[self.current_scene]
        self._original_env = {}
        for key, value in scene.env.items():
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_env(self):
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._original_env = {}

    def is_scene_match(self, required_scenes: List[str]) -> bool:
        if not required_scenes:
            return True
        if self.current_scene is None:
            return True
        return self.current_scene in required_scenes

    def get_scene_info(self, scene_id: Optional[str] = None) -> Optional[SceneInfo]:
        if scene_id is None:
            scene_id = self.current_scene
        return self.scenes.get(scene_id)

    def get_current_scene(self) -> Optional[str]:
        return self.current_scene
