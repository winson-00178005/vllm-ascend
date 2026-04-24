"""
Model Registry - 模型注册器（Session级别共享）

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

管理模型实例，提供：
- Session级别模型加载（只加载一次）
- 模型实例共享（跨测试共享）
- 自动清理

三层架构 Layer 2: Model Layer
"""
import logging
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable

from .environment_manager import EnvironmentManager, EnvironmentInfo
from .model_loader import ModelLoader, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """模型实例信息"""
    name: str
    config: ModelConfig
    runner: Any  # VllmRunner 实例
    loaded: bool = True


class ModelRegistry:
    """
    模型注册器 - Session级别共享模型实例
    
    关键特性:
    - 模型在 session 中只加载一次
    - 所有测试共享同一实例
    - 自动清理
    
    用法:
        # 加载模型（session开始时）
        ModelRegistry.load_models(["qwen-7b", "qwen3-8b"], env)
        
        # 获取模型实例（测试中使用）
        runner = ModelRegistry.get_model("qwen-7b")
        
        # 清理（session结束时）
        ModelRegistry.cleanup()
    """
    
    _models: Dict[str, ModelInstance] = {}
    _loaded: bool = False
    _loader: Optional[ModelLoader] = None
    
    @classmethod
    def initialize(cls):
        """初始化"""
        cls._models = {}
        cls._loaded = False
        cls._loader = ModelLoader()
    
    @classmethod
    def load_models(
        cls,
        models: List[str],
        env: EnvironmentInfo,
        runner_factory: Optional[Callable] = None,
    ) -> List[str]:
        """
        在环境中加载模型（Session级别，只加载一次）
        
        Args:
            models: 模型名称列表
            env: 环境信息
            runner_factory: VllmRunner 创建工厂（可选）
        
        Returns:
            List[str]: 成功加载的模型列表
        """
        if cls._loader is None:
            cls.initialize()
        
        env_manager = EnvironmentManager.get_instance()
        loaded_models = []
        
        for model_name in models:
            # 检查是否已加载
            if model_name in cls._models and cls._models[model_name].loaded:
                logger.info(f"Model {model_name} already loaded, skipping")
                loaded_models.append(model_name)
                continue
            
            # 获取模型配置
            model_config = cls._loader.get_model_config(model_name)
            if model_config is None:
                logger.warning(f"Model {model_name} not found in config, skipping")
                continue
            
            # 检查环境支持
            if not env_manager.is_model_supported(model_config):
                logger.info(
                    f"Model {model_name} not supported in {env.label}, skipping"
                )
                continue
            
            # 加载模型
            try:
                runner = cls._create_runner(model_config, env, runner_factory)
                
                cls._models[model_name] = ModelInstance(
                    name=model_name,
                    config=model_config,
                    runner=runner,
                    loaded=True,
                )
                
                loaded_models.append(model_name)
                logger.info(f"Loaded model {model_name} in {env.label}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                cls._models[model_name] = ModelInstance(
                    name=model_name,
                    config=model_config,
                    runner=None,
                    loaded=False,
                )
        
        cls._loaded = True
        return loaded_models
    
    @classmethod
    def _create_runner(
        cls,
        model_config: ModelConfig,
        env: EnvironmentInfo,
        runner_factory: Optional[Callable] = None,
    ) -> Any:
        """
        创建 VllmRunner 实例
        
        Args:
            model_config: 模型配置
            env: 环境信息
            runner_factory: 自定义创建工厂
        
        Returns:
            VllmRunner 实例
        """
        if runner_factory is not None:
            return runner_factory(model_config, env)
        
        # 默认创建方式
        try:
            import sys
            tests_path = Path(__file__).parent.parent.parent
            if str(tests_path) not in sys.path:
                sys.path.insert(0, str(tests_path))
            
            try:
                from tests.e2e.conftest import VllmRunner
            except ImportError:
                from e2e.conftest import VllmRunner
            
            runner = VllmRunner(
                model_config.model_path,
                tensor_parallel_size=env.num_npus,
                enforce_eager=True,
                dtype="float16",
                max_model_len=8192,
            )
            
            return runner
            
        except ImportError as e:
            logger.warning(f"VllmRunner not available: {e}")
            # 返回 mock runner 用于测试框架验证
            return cls._create_mock_runner(model_config, env)
    
    @classmethod
    def _create_mock_runner(cls, model_config: ModelConfig, env: EnvironmentInfo):
        """创建 mock runner（用于框架验证）"""
        class MockRunner:
            def __init__(self, config, environment):
                self.config = config
                self.env = environment
                self.model_name = config.name
                logger.info(f"MockRunner created: {config.name} in {environment.label}")
            
            def generate_greedy(self, prompts, max_tokens):
                logger.info(f"Mock generate: {len(prompts)} prompts, {max_tokens} tokens")
                return [(list(range(max_tokens)), f"Mock output for {prompts[0]}")]
            
            def cleanup(self):
                logger.info(f"MockRunner cleanup: {self.model_name}")
        
        return MockRunner(model_config, env)
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        """
        获取已加载的模型实例
        
        Args:
            model_name: 模型名称
        
        Returns:
            VllmRunner 实例或 None
        """
        if model_name not in cls._models:
            return None
        
        instance = cls._models[model_name]
        if not instance.loaded:
            return None
        
        return instance.runner
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model_name not in cls._models:
            return None
        return cls._models[model_name].config
    
    @classmethod
    def get_loaded_models(cls) -> List[str]:
        """获取已加载的模型列表"""
        return [
            name for name, instance in cls._models.items()
            if instance.loaded
        ]
    
    @classmethod
    def is_model_loaded(cls, model_name: str) -> bool:
        """检查模型是否已加载"""
        return (
            model_name in cls._models and
            cls._models[model_name].loaded
        )
    
    @classmethod
    def cleanup(cls):
        """清理所有模型实例"""
        logger.info("Cleaning up all model instances")
        
        for model_name, instance in cls._models.items():
            if instance.runner is not None:
                try:
                    if hasattr(instance.runner, 'cleanup'):
                        instance.runner.cleanup()
                    elif hasattr(instance.runner, '__exit__'):
                        instance.runner.__exit__(None, None, None)
                    
                    del instance.runner
                    
                except Exception as e:
                    logger.warning(f"Cleanup {model_name} failed: {e}")
        
        cls._models.clear()
        cls._loaded = False
        
        # 强制垃圾回收
        gc.collect()
        logger.info("Model registry cleanup complete")
    
    @classmethod
    def list_models(cls) -> None:
        """打印已加载的模型"""
        loaded = cls.get_loaded_models()
        
        print("\n" + "=" * 60)
        print("Loaded Models (Model Registry)")
        print("=" * 60)
        
        if not loaded:
            print("\nNo models loaded")
        else:
            for name in loaded:
                instance = cls._models[name]
                print(f"\n{name}:")
                print(f"  Path: {instance.config.model_path}")
                print(f"  Devices: {instance.config.supported_devices}")
        
        print("\n" + "=" * 60)


__all__ = [
    "ModelInstance",
    "ModelRegistry",
]