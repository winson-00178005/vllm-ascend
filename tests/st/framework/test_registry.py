"""
Test Registry - 测试注册器（测试-模型映射）

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

管理测试与模型的映射，提供：
- 测试声明支持的模型
- 查询模型支持的测试
- 自动跳过不支持的模型

三层架构 Layer 3: Test Layer
"""
import functools
import logging
import pytest
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Set, Callable, Any

logger = logging.getLogger(__name__)

DEFAULT_TEST_REGISTRY_PATH = Path(__file__).parent.parent / "config" / "test_registry.yaml"


@dataclass
class TestInfo:
    """测试信息"""
    name: str
    func: Callable
    models: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def supports_model(self, model_name: str) -> bool:
        """检查是否支持该模型"""
        # "all" 表示支持所有模型
        if "all" in self.models:
            return True
        return model_name in self.models


class TestRegistry:
    """
    测试注册器 - 管理测试与模型的映射
    
    功能:
    - 注册测试及其支持的模型
    - 查询模型支持的测试列表
    - 检查测试是否支持某个模型
    
    用法:
        # 注册测试
        @model_test(models=["qwen-7b", "qwen3-8b"])
        def test_inference():
            pass
        
        # 查询模型支持的测试
        tests = TestRegistry.get_tests_for_model("qwen-7b")
        
        # 检查支持
        if TestRegistry.is_test_supported("test_inference", "qwen-7b"):
            ...
    """
    
    _tests: Dict[str, TestInfo] = {}
    _config_loaded: bool = False
    
    @classmethod
    def initialize(cls):
        """初始化"""
        cls._tests = {}
        cls._config_loaded = False
    
    @classmethod
    def load_config(cls, config_path: Optional[Path] = None):
        """
        从配置文件加载测试注册信息
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = DEFAULT_TEST_REGISTRY_PATH
        
        if not config_path.exists():
            logger.warning(f"Test registry config not found: {config_path}")
            return
        
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config or 'tests' not in config:
            logger.warning("Empty or invalid test registry config")
            return
        
        for test_name, test_config in config['tests'].items():
            cls._tests[test_name] = TestInfo(
                name=test_name,
                func=None,  # 函数会在装饰器中设置
                models=test_config.get('models', ['all']),
                tags=test_config.get('tags', []),
                description=test_config.get('description', ''),
            )
        
        cls._config_loaded = True
        logger.info(f"Loaded {len(cls._tests)} tests from config")
    
    @classmethod
    def register(cls, test_func: Callable, models: List[str], tags: List[str] = None):
        """
        注册测试及其支持的模型
        
        Args:
            test_func: 测试函数
            models: 支持的模型列表
            tags: 标签列表
        """
        test_name = test_func.__name__
        
        cls._tests[test_name] = TestInfo(
            name=test_name,
            func=test_func,
            models=models or ['all'],
            tags=tags or [],
            description=test_func.__doc__ or "",
        )
        
        logger.debug(f"Registered test {test_name} with models: {models}")
    
    @classmethod
    def get_test_info(cls, test_name: str) -> Optional[TestInfo]:
        """获取测试信息"""
        return cls._tests.get(test_name)
    
    @classmethod
    def get_tests_for_model(cls, model_name: str) -> List[str]:
        """
        获取支持该模型的测试列表
        
        Args:
            model_name: 模型名称
        
        Returns:
            List[str]: 测试名称列表
        """
        return [
            test.name for test in cls._tests.values()
            if test.func is not None and test.supports_model(model_name)
        ]
    
    @classmethod
    def get_tests_by_tag(cls, tag: str) -> List[str]:
        """获取包含某标签的测试列表"""
        return [
            test.name for test in cls._tests.values()
            if tag in test.tags
        ]
    
    @classmethod
    def is_test_supported(cls, test_name: str, model_name: str) -> bool:
        """
        检查测试是否支持该模型
        
        Args:
            test_name: 测试名称
            model_name: 模型名称
        
        Returns:
            bool: 是否支持
        """
        if test_name not in cls._tests:
            # 未注册的测试默认支持所有模型
            return True
        
        return cls._tests[test_name].supports_model(model_name)
    
    @classmethod
    def get_all_tests(cls) -> Dict[str, TestInfo]:
        """获取所有注册的测试"""
        return cls._tests
    
    @classmethod
    def list_tests(cls) -> None:
        """打印所有注册的测试"""
        print("\n" + "=" * 60)
        print("Registered Tests (Test Registry)")
        print("=" * 60)
        
        if not cls._tests:
            print("\nNo tests registered")
        else:
            for name, test in cls._tests.items():
                print(f"\n{name}:")
                print(f"  Models: {test.models}")
                if test.tags:
                    print(f"  Tags: {test.tags}")
                if test.description:
                    print(f"  Description: {test.description[:50]}...")
        
        print("\n" + "=" * 60)


def model_test(
    models: List[str] = None,
    tags: List[str] = None,
):
    """
    装饰器 - 标记测试支持的模型
    
    功能:
    1. 注册测试到 TestRegistry
    2. 运行时检查模型支持，自动跳过
    
    Args:
        models: 支持的模型列表（默认 ["all"]）
        tags: 标签列表
    
    用法:
        @model_test(models=["qwen-7b", "qwen3-8b"])
        def test_qwen_inference():
            # 只在 qwen-7b 或 qwen3-8b 上运行
            pass
        
        @model_test(models=["all"])
        def test_all_models():
            # 在所有模型上运行
            pass
        
        @model_test(tags=["inference", "dense"])
        def test_dense_model():
            # 标记为 inference 和 dense 类型
            pass
    
    注意:
        - 与 @pytest.mark.parametrize 配合使用时，
          @model_test 应放在最外层
    """
    if models is None:
        models = ["all"]
    
    if tags is None:
        tags = []
    
    def decorator(func: Callable) -> Callable:
        # 注册到 TestRegistry
        TestRegistry.register(func, models, tags)
        
        # 存储 test info 用于运行时检查
        func._model_test_info = {
            'models': models,
            'tags': tags,
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查当前模型（从 kwargs 或 pytest fixture）
            current_model = kwargs.get('model_name', None)
            
            if current_model:
                if not TestRegistry.is_test_supported(func.__name__, current_model):
                    pytest.skip(
                        f"Test {func.__name__} not supported for model {current_model}"
                    )
                    return None
            
            # 执行测试
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_model_test_info(func: Callable) -> Optional[Dict]:
    """获取函数的 model_test 信息"""
    return getattr(func, '_model_test_info', None)


def is_model_test(func: Callable) -> bool:
    """检查函数是否标记了 @model_test"""
    return hasattr(func, '_model_test_info')


__all__ = [
    "TestInfo",
    "TestRegistry",
    "model_test",
    "get_model_test_info",
    "is_model_test",
]