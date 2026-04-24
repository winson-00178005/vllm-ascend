"""
测试装饰器 - 用于标记测试用例的场景和模型要求

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- require_scene: 标记测试需要的场景
- require_model: 标记测试需要的模型
- require_device: 标记测试需要的设备类型
- require_tags: 标记测试需要场景包含的标签
"""
import pytest
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def require_scene(scene_names: str) -> Callable:
    """
    装饰器：标记测试需要在指定场景下执行
    
    Args:
        scene_names: 场景名称，多个场景用 | 分隔
                    例如："SINGLE_CARD_310P|MULTI_CARD_310P_2"
    
    用法:
        @require_scene("SINGLE_CARD_310P|MULTI_CARD_310P_2")
        def test_inference():
            # 测试代码
            pass
    
    Returns:
        包装后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 导入场景管理器
            from .scene_manager import SceneManager
            scene_manager = SceneManager()
            current_scene = scene_manager.get_current_scene()
            
            # 检查是否有场景设置
            if current_scene is None:
                pytest.skip("No scene is set")
                return
            
            # 检查场景是否匹配
            required_scenes = set(scene_names.split('|'))
            if current_scene.name not in required_scenes:
                pytest.skip(
                    f"Test requires scene {scene_names}, "
                    f"but current scene is {current_scene.name}"
                )
                return
            
            # 执行测试
            logger.info(f"Running test in scene: {current_scene.name}")
            return func(*args, **kwargs)
        
        # 添加元数据
        wrapper._required_scenes = scene_names
        wrapper._decorator_type = 'scene'
        return wrapper
    
    return decorator

def require_model(model_names: str) -> Callable:
    """
    装饰器：标记测试需要在指定模型下执行
    
    Args:
        model_names: 模型名称，多个模型用 | 分隔
                    例如："qwen-7b|llama-7b"
    
    用法:
        @require_model("qwen-7b|llama-7b")
        def test_model_inference():
            # 测试代码
            pass
    
    Returns:
        包装后的函数
    
    注意: 模型信息应该通过fixture注入，实际判断在fixture中
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 模型检查应该在fixture中完成
            return func(*args, **kwargs)
        
        # 添加元数据
        wrapper._required_models = model_names
        wrapper._decorator_type = 'model'
        return wrapper
    
    return decorator

def require_device(device_types: str) -> Callable:
    """
    装饰器：标记测试需要在指定设备类型下执行
    
    Args:
        device_types: 设备类型，多个设备用 | 分隔
                    例如："310P|910B"
    
    用法:
        @require_device("310P")
        def test_310p_only():
            # 310P专用测试
            pass
    
    Returns:
        包装后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 导入场景管理器
            from .scene_manager import SceneManager
            scene_manager = SceneManager()
            current_scene = scene_manager.get_current_scene()
            
            # 检查是否有场景设置
            if current_scene is None:
                pytest.skip("No scene is set")
                return
            
            # 检查设备是否匹配
            required_devices = set(device_types.split('|'))
            if current_scene.device_type not in required_devices:
                pytest.skip(
                    f"Test requires device {device_types}, "
                    f"but current device is {current_scene.device_type}"
                )
                return
            
            # 执行测试
            logger.info(f"Running test on device: {current_scene.device_type}")
            return func(*args, **kwargs)
        
        # 添加元数据
        wrapper._required_devices = device_types
        wrapper._decorator_type = 'device'
        return wrapper
    
    return decorator

def require_tags(tags: str) -> Callable:
    """
    装饰器：标记测试需要场景包含指定标签
    
    Args:
        tags: 标签，多个标签用 | 分隔
             例如："single|npu"
    
    用法:
        @require_tags("single|npu")
        def test_single_npu():
            # 单NPU场景测试
            pass
    
    Returns:
        包装后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 导入场景管理器
            from .scene_manager import SceneManager
            scene_manager = SceneManager()
            current_scene = scene_manager.get_current_scene()
            
            # 检查是否有场景设置
            if current_scene is None:
                pytest.skip("No scene is set")
                return
            
            # 检查标签是否匹配
            required_tags = set(tags.split('|'))
            if not required_tags.issubset(current_scene.tags):
                pytest.skip(
                    f"Test requires tags {tags}, "
                    f"but current scene has tags {current_scene.tags}"
                )
                return
            
            # 执行测试
            logger.info(f"Running test with tags: {current_scene.tags}")
            return func(*args, **kwargs)
        
        # 添加元数据
        wrapper._required_tags = tags
        wrapper._decorator_type = 'tags'
        return wrapper
    
    return decorator

def skip_if_no_scene(func: Callable) -> Callable:
    """
    装饰器：如果没有设置场景则跳过测试
    
    用法:
        @skip_if_no_scene
        def test_something():
            pass
    
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from .scene_manager import SceneManager
        scene_manager = SceneManager()
        current_scene = scene_manager.get_current_scene()
        
        if current_scene is None:
            pytest.skip("No scene is set, skipping test")
            return
        
        return func(*args, **kwargs)
    
    return wrapper