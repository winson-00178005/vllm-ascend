"""
模型管理器 - 负责加载和管理测试模型配置

参考 vLLM 社区的 ModelInfo 数据类模式
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AscendModelInfo:
    """模型信息数据类（参考 vllm/tests/models/utils.py 的 ModelInfo）"""

    name: str
    model_id: str
    architecture: str = ""
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    supported_scenes: list[str] = field(default_factory=list)
    enable_test: bool = True
    is_multimodal: bool = False
    is_moe: bool = False
    min_gpu_gb: int = 0
    hf_ppl: Optional[float] = None


class ModelManager:
    """模型管理器（单例模式）"""

    _instance = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置实例（用于测试）"""
        cls._instance = None

    def __init__(self):
        self._models: dict[str, AscendModelInfo] = {}
        self._load_config()

    def _load_config(self) -> None:
        """从 models.yaml 加载模型配置"""
        config_path = Path(__file__).parent.parent / "models.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "models" not in data:
            raise ValueError("模型配置文件格式错误：缺少 'models' 键")

        for name, cfg in data["models"].items():
            self._models[name] = AscendModelInfo(
                name=name,
                model_id=cfg.get("model_id", name),
                architecture=cfg.get("architecture", ""),
                dtype=cfg.get("dtype", "auto"),
                max_model_len=cfg.get("max_model_len"),
                supported_scenes=cfg.get("supported_scenes", []),
                enable_test=cfg.get("enable_test", True),
                is_multimodal=cfg.get("is_multimodal", False),
                is_moe=cfg.get("is_moe", False),
                min_gpu_gb=cfg.get("min_gpu_gb", 0),
                hf_ppl=cfg.get("hf_ppl"),
            )

    def get_models_for_scene(self, scene: str) -> list[str]:
        """获取指定场景下支持的所有模型"""
        return [
            name
            for name, info in self._models.items()
            if scene in info.supported_scenes and info.enable_test
        ]

    def is_supported(self, model: str, scene: str) -> bool:
        """检查模型是否支持指定场景"""
        info = self._models.get(model)
        if not info:
            return False
        return scene in info.supported_scenes

    def get_info(self, model: str) -> Optional[AscendModelInfo]:
        """获取模型信息"""
        return self._models.get(model)

    @property
    def all_models(self) -> list[str]:
        """所有可用模型列表"""
        return list(self._models.keys())

    def check_gpu_requirement(self, model: str) -> bool:
        """检查当前 GPU 是否满足模型要求"""
        info = self._models.get(model)
        if not info or info.min_gpu_gb == 0:
            return True

        try:
            import torch_npu

            total_memory_gb = (
                torch_npu.npu.get_device_properties(0).total_memory / (1024**3)
            )
            return total_memory_gb >= info.min_gpu_gb
        except Exception:
            # 无法检测时返回 True，让测试继续执行
            return True
