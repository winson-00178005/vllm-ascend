"""
模型加载器 - 负责加载和管理测试模型配置

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- 从配置文件加载模型定义
- 提供模型配置查询
- 管理模型加载状态（模拟）
"""
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str
    model_path: str
    supported_devices: List[str]
    min_memory: str
    max_batch_size: int
    requires_multi_card: bool = False
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def supports_device(self, device_type: str) -> bool:
        """
        检查是否支持指定设备
        
        Args:
            device_type: 设备类型
        
        Returns:
            是否支持
        """
        return device_type in self.supported_devices
    
    def __str__(self) -> str:
        return f"Model({self.name}: {self.model_path})"

class ModelLoader:
    """
    模型加载器 - 单例模式
    
    功能:
    - 加载模型配置
    - 提供模型查询
    - 管理模型加载状态（模拟）
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型加载器"""
        if self._initialized:
            return
        self._initialized = True
        self.models: Dict[str, ModelConfig] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._load_model_configs()
    
    def _load_model_configs(self):
        """
        从配置文件加载模型
        
        配置文件路径: config/models.yaml
        """
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        
        if not config_path.exists():
            logger.warning(f"Model config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config or 'models' not in config:
                logger.warning("Invalid model config format")
                return
            
            for model_name, model_config in config['models'].items():
                model = ModelConfig(
                    name=model_name,
                    model_path=model_config['model_path'],
                    supported_devices=model_config['supported_devices'],
                    min_memory=model_config['min_memory'],
                    max_batch_size=model_config['max_batch_size'],
                    requires_multi_card=model_config.get('requires_multi_card', False),
                    tags=model_config.get('tags', [])
                )
                self.models[model_name] = model
                logger.info(f"Loaded model config: {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to load model configs: {e}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型配置对象，如果不存在返回None
        """
        return self.models.get(model_name)
    
    def get_all_models(self) -> List[ModelConfig]:
        """
        获取所有模型配置
        
        Returns:
            模型配置列表
        """
        return list(self.models.values())
    
    def get_models_by_device(self, device_type: str) -> List[ModelConfig]:
        """
        根据设备类型获取模型
        
        Args:
            device_type: 设备类型
        
        Returns:
            支持该设备的模型列表
        """
        return [m for m in self.models.values() if m.supports_device(device_type)]
    
    def get_models_by_tag(self, tag: str) -> List[ModelConfig]:
        """
        根据标签获取模型
        
        Args:
            tag: 标签名称
        
        Returns:
            包含该标签的模型列表
        """
        return [m for m in self.models.values() if tag in m.tags]
    
    def load_model(self, model_name: str, scene_config: dict = None) -> Any:
        """
        加载模型（模拟）
        
        Args:
            model_name: 模型名称
            scene_config: 场景配置
        
        Returns:
            模型实例（模拟）
        
        注意: 实际使用时应该调用vLLM的模型加载逻辑
        """
        if model_name in self._loaded_models:
            logger.info(f"Model already loaded: {model_name}")
            return self._loaded_models[model_name]
        
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model not found: {model_name}")
        
        logger.info(f"Loading model: {model_name} from {model_config.model_path}")
        
        # 模拟模型加载（实际应该调用vLLM的LLM类）
        model_instance = {
            'name': model_name,
            'config': model_config,
            'scene_config': scene_config,
            'status': 'loaded',
            'model_path': model_config.model_path
        }
        
        self._loaded_models[model_name] = model_instance
        logger.info(f"Model loaded successfully: {model_name}")
        return model_instance
    
    def unload_model(self, model_name: str):
        """
        卸载模型
        
        Args:
            model_name: 模型名称
        """
        if model_name in self._loaded_models:
            logger.info(f"Unloading model: {model_name}")
            del self._loaded_models[model_name]
    
    def unload_all_models(self):
        """卸载所有模型"""
        for model_name in list(self._loaded_models.keys()):
            self.unload_model(model_name)
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """
        获取已加载的模型
        
        Returns:
            已加载模型的字典
        """
        return self._loaded_models.copy()
    
    def list_models(self) -> str:
        """
        列出所有模型（用于命令行输出）
        
        Returns:
            格式化的模型列表字符串
        """
        output = "\n可用模型:\n"
        for model in self.get_all_models():
            output += f"  - {model.name}: {model.model_path}\n"
            output += f"    支持设备: {', '.join(model.supported_devices)}\n"
            output += f"    最小内存: {model.min_memory}\n"
            if model.requires_multi_card:
                output += f"    需要多卡: True\n"
        return output