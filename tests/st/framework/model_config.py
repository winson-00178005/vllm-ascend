from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml
import os


@dataclass
class ModelInfo:
    name: str
    supported_scenes: List[str]
    max_model_len: int
    tp_size: List[int]
    quantizations: List[str] = field(default_factory=list)
    enforce_eager: bool = False
    enable_expert_parallel: bool = False
    is_multimodal: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class ModelConfig:
    _instance: Optional["ModelConfig"] = None

    def __init__(self, config_path: str):
        self.model_groups: Dict[str, List[ModelInfo]] = {}
        self.all_models: List[ModelInfo] = []
        self.current_model: Optional[str] = None
        self._load_config(config_path)

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ModelConfig":
        if cls._instance is None:
            path = config_path or cls._get_default_config_path()
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def _get_default_config_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def _load_config(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for group_name, models in config["model_groups"].items():
            model_list = []
            for model in models:
                model_info = ModelInfo(
                    name=model["name"],
                    supported_scenes=model["supported_scenes"],
                    max_model_len=model["max_model_len"],
                    tp_size=model["tp_size"],
                    quantizations=model.get("quantizations", []),
                    enforce_eager=model.get("enforce_eager", False),
                    enable_expert_parallel=model.get("enable_expert_parallel", False),
                    is_multimodal=model.get("is_multimodal", False),
                    extra_kwargs=model.get("extra_kwargs", {})
                )
                model_list.append(model_info)
                self.all_models.append(model_info)
            self.model_groups[group_name] = model_list

    def get_models_for_scene(self, scene_id: str) -> List[ModelInfo]:
        return [m for m in self.all_models if scene_id in m.supported_scenes]

    def is_model_supported(self, model_name: str, scene_id: str) -> bool:
        for model in self.all_models:
            if model.name == model_name and scene_id in model.supported_scenes:
                return True
        return False

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        for model in self.all_models:
            if model.name == model_name:
                return model
        return None

    def get_tp_size_for_model(self, model_name: str, scene_id: str) -> Optional[int]:
        model_info = self.get_model_info(model_name)
        if model_info is None:
            return None
        scene_info = self._get_scene_info(scene_id)
        if scene_info is None:
            return None
        tp = scene_info.tp_size
        if tp in model_info.tp_size:
            return tp
        return model_info.tp_size[0] if model_info.tp_size else None

    def _get_scene_info(self, scene_id: str):
        from tests.st.framework.scene_manager import SceneManager
        manager = SceneManager.get_instance()
        return manager.get_scene_info(scene_id)
