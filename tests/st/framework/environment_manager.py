"""
Environment Manager - 环境管理器

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

管理运行环境配置，提供：
- Runner 配置加载
- 环境选择和设置
- 模型支持判断

三层架构 Layer 1: Environment Layer
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

from .device_types import RunnerDeviceType

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_LABEL_PATH = Path(__file__).parent.parent / "config" / "runner_label.json"


@dataclass
class EnvironmentInfo:
    """环境信息"""
    label: str
    npu_type: RunnerDeviceType
    num_npus: int
    image_tag: str = ""
    description: str = ""
    
    @property
    def is_npu(self) -> bool:
        """是否为 NPU 环境"""
        return self.npu_type != RunnerDeviceType.CPU
    
    def __str__(self) -> str:
        return f"{self.label} ({self.npu_type.display_name} x{self.num_npus})"


class EnvironmentManager:
    """
    环境管理器 - 单例模式
    
    功能:
    - 加载 runner 配置
    - 设置当前运行环境
    - 检查模型是否支持当前环境
    
    用法:
        env_manager = EnvironmentManager()
        env_manager.load_runners()
        env = env_manager.set_environment("linux-aarch64-310p-2")
        
        # 检查模型支持
        if env_manager.is_model_supported(model_config):
            ...
    """
    
    _instance: Optional['EnvironmentManager'] = None
    _current_env: Optional[EnvironmentInfo] = None
    _runners: Dict[str, dict] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'EnvironmentManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_runners(self, config_path: Optional[Path] = None):
        """
        加载 runner 配置
        
        Args:
            config_path: 配置文件路径（默认使用 runner_label.json）
        """
        if config_path is None:
            config_path = DEFAULT_RUNNER_LABEL_PATH
        
        if not config_path.exists():
            logger.warning(f"Runner config not found: {config_path}")
            return
        
        with open(config_path, encoding='utf-8') as f:
            self._runners = json.load(f)
        
        logger.info(f"Loaded {len(self._runners)} runners from {config_path}")
    
    def set_environment(self, runner_label: str) -> EnvironmentInfo:
        """
        设置当前运行环境
        
        Args:
            runner_label: Runner 标签（如 "linux-aarch64-310p-2")
        
        Returns:
            EnvironmentInfo: 环境信息
        
        Raises:
            ValueError: 未知的 runner
        """
        if not self._runners:
            self.load_runners()
        
        if runner_label not in self._runners:
            raise ValueError(f"Unknown runner: {runner_label}")
        
        runner_config = self._runners[runner_label]
        
        self._current_env = EnvironmentInfo(
            label=runner_label,
            npu_type=RunnerDeviceType(runner_config.get("chip", "cpu")),
            num_npus=runner_config.get("npu_num", 0),
            image_tag=runner_config.get("image_tag", ""),
            description=runner_config.get("description", ""),
        )
        
        logger.info(f"Environment set: {self._current_env}")
        return self._current_env
    
    def get_environment(self) -> Optional[EnvironmentInfo]:
        """获取当前环境"""
        return self._current_env
    
    def get_all_runners(self) -> Dict[str, dict]:
        """获取所有 runner 配置"""
        if not self._runners:
            self.load_runners()
        return self._runners
    
    def get_runners_by_npu_type(self, npu_type: RunnerDeviceType) -> List[str]:
        """获取指定 NPU 类型的 runner 列表"""
        if not self._runners:
            self.load_runners()
        
        return [
            label for label, config in self._runners.items()
            if config.get("chip") == npu_type.value
        ]
    
    def is_model_supported(self, model_config) -> bool:
        """
        检查模型是否支持当前环境
        
        Args:
            model_config: 模型配置（ModelConfig对象）
        
        Returns:
            bool: 是否支持
        """
        if self._current_env is None:
            logger.warning("No environment set")
            return False
        
        # 检查设备类型
        supported_devices = getattr(model_config, 'supported_devices', [])
        if self._current_env.npu_type.value not in supported_devices:
            logger.debug(
                f"Model {model_config.name} not supported: "
                f"device {self._current_env.npu_type.value} not in {supported_devices}"
            )
            return False
        
        # 检查多卡要求
        requires_multi_card = getattr(model_config, 'requires_multi_card', False)
        if requires_multi_card and self._current_env.num_npus < 2:
            logger.debug(
                f"Model {model_config.name} requires multi-card, "
                f"but environment has {self._current_env.num_npus} cards"
            )
            return False
        
        # 检查最小卡数要求
        min_cards = getattr(model_config, 'min_cards', 1)
        if self._current_env.num_npus < min_cards:
            logger.debug(
                f"Model {model_config.name} requires {min_cards} cards, "
                f"but environment has {self._current_env.num_npus} cards"
            )
            return False
        
        return True
    
    def list_environments(self) -> None:
        """打印所有可用环境"""
        if not self._runners:
            self.load_runners()
        
        print("\n" + "=" * 60)
        print("Available Environments (Runners)")
        print("=" * 60)
        
        for label, config in self._runners.items():
            npu_type = RunnerDeviceType(config.get("chip", "cpu"))
            num_npus = config.get("npu_num", 0)
            print(f"\n{label}:")
            print(f"  NPU Type: {npu_type.display_name}")
            print(f"  NPU Count: {num_npus}")
            if config.get("image_tag"):
                print(f"  Image: {config['image_tag']}")
            if config.get("description"):
                print(f"  Description: {config['description']}")
        
        print("\n" + "=" * 60)


__all__ = [
    "EnvironmentInfo",
    "EnvironmentManager",
]