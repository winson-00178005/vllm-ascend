"""
场景管理器 - 负责加载和管理测试场景配置

支持三级优先级：
1. 命令行参数 (--scene)
2. 环境变量 (ST_SCENE)
3. 自动探测硬件环境

Phase 2 新增：
- 场景继承（base_scene）
- 场景组合（features）
- 量化配置（quantization）
- 图模式配置（graph_mode）
"""

import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml


class SceneManager:
    """场景管理器（单例模式）"""

    _instance = None

    @classmethod
    def get_instance(cls) -> "SceneManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置实例（用于测试）"""
        cls._instance = None

    def __init__(self, cli_scene: Optional[str] = None):
        self._scenes = self._load_config()
        self._current = cli_scene or os.environ.get("ST_SCENE") or self._auto_detect()

    def _load_config(self) -> dict:
        """从 scenes.yaml 加载场景配置"""
        config_path = Path(__file__).parent.parent / "scenes.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"场景配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "scenes" not in data:
            raise ValueError("场景配置文件格式错误：缺少 'scenes' 键")

        return data["scenes"]

    def _resolve_scene_inheritance(self, scene_name: str) -> dict:
        """
        解析场景继承关系（Phase 2 新增）

        如果场景配置了 base_scene，则继承基础场景的配置并覆盖
        """
        config = self._scenes.get(scene_name)
        if config is None:
            raise ValueError(f"未知场景: {scene_name}")

        # 如果没有 base_scene，直接返回
        if "base_scene" not in config:
            return deepcopy(config)

        # 递归解析基础场景
        base_name = config["base_scene"]
        base_config = self._resolve_scene_inheritance(base_name)

        # 合并配置（子场景覆盖基础场景）
        merged = deepcopy(base_config)

        # 合并 env_vars
        if "env_vars" in config:
            merged.setdefault("env_vars", {})
            merged["env_vars"].update(config["env_vars"])

        # 覆盖其他字段
        for key in config:
            if key != "base_scene" and key != "env_vars":
                merged[key] = config[key]

        return merged

    def _auto_detect(self) -> str:
        """自动探测当前硬件环境"""
        # 1. 检测 NPU 卡数
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if visible:
            card_count = len([x for x in visible.split(",") if x.strip()])
        else:
            card_count = self._detect_via_npu_smi()

        # 2. 检测硬件型号
        hardware = self._detect_hardware()

        # 3. 匹配场景
        if hardware == "310p":
            return "ascend_310p"
        if card_count == 1:
            return "single_card"
        if card_count == 2:
            return "multi_card_tp2"
        if card_count >= 4:
            return "multi_card_tp4"

        # 默认返回 single_card
        return "single_card"

    def _detect_via_npu_smi(self) -> int:
        """通过 npu-smi 检测 NPU 卡数"""
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # 解析输出中的卡数
                # 示例输出: "Total Count: 8"
                for line in result.stdout.split("\n"):
                    if "Total Count" in line:
                        return int(line.split(":")[-1].strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return 1

    def _detect_hardware(self) -> str:
        """通过 npu-smi 检测硬件型号"""
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "device-info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                if "310P" in result.stdout or "310p" in result.stdout:
                    return "310p"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "910b"

    @property
    def current(self) -> str:
        """当前场景名称"""
        return self._current

    @property
    def all_scenes(self) -> list[str]:
        """所有可用场景列表"""
        return list(self._scenes.keys())

    def get_config(self) -> dict:
        """获取当前场景配置（支持继承解析）"""
        return self.get_scene_config(self._current)

    def get_scene_config(self, scene_name: str) -> dict:
        """获取指定场景配置（支持继承解析）"""
        return self._resolve_scene_inheritance(scene_name)

    def is_valid_scene(self, scene_name: str) -> bool:
        """检查场景是否有效"""
        return scene_name in self._scenes

    def apply_env_vars(self, scene_name: Optional[str] = None) -> None:
        """应用场景的环境变量"""
        config = self.get_scene_config(scene_name or self._current)
        env_vars = config.get("env_vars", {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)

    def get_features(self, scene_name: Optional[str] = None) -> list[str]:
        """获取场景启用的特性列表（Phase 2 新增）"""
        config = self.get_scene_config(scene_name or self._current)
        return config.get("features", [])

    def has_feature(self, feature: str, scene_name: Optional[str] = None) -> bool:
        """检查场景是否启用了某个特性（Phase 2 新增）"""
        features = self.get_features(scene_name)
        return feature in features

    def get_quantization_config(self, scene_name: Optional[str] = None) -> Optional[str]:
        """获取场景量化配置（Phase 2 新增）"""
        config = self.get_scene_config(scene_name or self._current)
        return config.get("quantization")

    def get_graph_mode(self, scene_name: Optional[str] = None) -> Optional[str]:
        """获取场景图模式配置（Phase 2 新增）"""
        config = self.get_scene_config(scene_name or self._current)
        return config.get("graph_mode")
