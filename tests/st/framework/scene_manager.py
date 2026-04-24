"""
场景管理器 - 负责加载、管理和切换测试场景

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- 从配置文件加载场景定义
- 管理当前运行的场景
- 提供场景查询和匹配功能
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Scene:
    """场景定义数据类"""
    name: str
    description: str
    device_type: str
    card_count: int
    models: List[str]
    tags: Set[str]
    node_count: int = 1
    
    def matches(self, required_scenes: str) -> bool:
        """
        检查是否匹配要求的场景
        
        Args:
            required_scenes: 要求的场景名称，多个场景用 | 分隔
        
        Returns:
            是否匹配
        """
        required_set = set(required_scenes.split('|'))
        return self.name in required_set
    
    def supports_model(self, model_name: str) -> bool:
        """
        检查是否支持指定模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否支持
        """
        return model_name in self.models or 'all' in self.models
    
    def __str__(self) -> str:
        return f"Scene({self.name}: {self.device_type}, {self.card_count} cards)"

class SceneManager:
    """
    场景管理器 - 单例模式
    
    功能:
    - 加载场景配置
    - 管理当前场景
    - 提供场景查询
    """
    
    _instance = None
    _current_scene: Optional[Scene] = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化场景管理器"""
        if self._initialized:
            return
        self._initialized = True
        self.scenes: Dict[str, Scene] = {}
        self._load_scenes()
    
    def _load_scenes(self):
        """
        从配置文件加载场景
        
        配置文件路径: config/scenes.yaml
        """
        config_path = Path(__file__).parent.parent / "config" / "scenes.yaml"
        
        if not config_path.exists():
            logger.warning(f"Scene config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config or 'scenes' not in config:
                logger.warning("Invalid scene config format")
                return
            
            for scene_name, scene_config in config['scenes'].items():
                scene = Scene(
                    name=scene_name,
                    description=scene_config.get('description', ''),
                    device_type=scene_config['device_type'],
                    card_count=scene_config['card_count'],
                    models=scene_config['models'],
                    tags=set(scene_config.get('tags', [])),
                    node_count=scene_config.get('node_count', 1)
                )
                self.scenes[scene_name] = scene
                logger.info(f"Loaded scene: {scene_name} - {scene.description}")
        
        except Exception as e:
            logger.error(f"Failed to load scenes: {e}")
    
    def get_scene(self, scene_name: str) -> Optional[Scene]:
        """
        获取指定场景
        
        Args:
            scene_name: 场景名称
        
        Returns:
            场景对象，如果不存在返回None
        """
        return self.scenes.get(scene_name)
    
    def set_current_scene(self, scene_name: str) -> bool:
        """
        设置当前场景
        
        Args:
            scene_name: 场景名称
        
        Returns:
            是否成功设置
        """
        scene = self.get_scene(scene_name)
        if scene:
            self._current_scene = scene
            logger.info(f"Set current scene to: {scene_name}")
            return True
        logger.warning(f"Scene not found: {scene_name}")
        return False
    
    def get_current_scene(self) -> Optional[Scene]:
        """
        获取当前场景
        
        Returns:
            当前场景对象
        """
        return self._current_scene
    
    def get_all_scenes(self) -> List[Scene]:
        """
        获取所有场景
        
        Returns:
            场景列表
        """
        return list(self.scenes.values())
    
    def get_scenes_by_tag(self, tag: str) -> List[Scene]:
        """
        根据标签获取场景
        
        Args:
            tag: 标签名称
        
        Returns:
            匹配的场景列表
        """
        return [s for s in self.scenes.values() if tag in s.tags]
    
    def get_scenes_by_device(self, device_type: str) -> List[Scene]:
        """
        根据设备类型获取场景
        
        Args:
            device_type: 设备类型（如"310P", "910B"）
        
        Returns:
            匹配的场景列表
        """
        return [s for s in self.scenes.values() if s.device_type == device_type]
    
    def list_scenes(self) -> str:
        """
        列出所有场景（用于命令行输出）
        
        Returns:
            格式化的场景列表字符串
        """
        output = "\n可用场景:\n"
        for scene in self.get_all_scenes():
            output += f"  - {scene.name}: {scene.description}\n"
            output += f"    设备: {scene.device_type}, 卡数: {scene.card_count}\n"
            output += f"    支持模型: {', '.join(scene.models)}\n"
        return output