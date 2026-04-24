"""
现有测试适配器 - 为现有tests/e2e/测试提供新框架的功能

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- adapt_legacy_test: 自动为现有测试添加场景和模型装饰器
- LegacyTestWrapper: 包装VllmRunner等类以支持场景管理
- auto_patch_conftest: 自动包装conftest中的Runner类
"""
import functools
from typing import Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def adapt_legacy_test(test_func: Callable) -> Callable:
    """
    装饰器：自动为现有测试添加场景和模型标记
    
    用法:
        @adapt_legacy_test
        def test_qwen3_dense_tp1_fp16():
            # 现有测试代码，无需修改
            pass
    
    功能:
        - 从测试文件路径自动检测场景
        - 从测试代码自动检测模型
        - 自动添加装饰器
        - 保持原有测试功能不变
    """
    # 获取测试文件路径
    test_file_path = Path(test_func.__code__.co_filename)
    
    # 自动检测场景
    from .scene_detector import SceneDetector
    scene_info = SceneDetector.detect_all(str(test_file_path), '')
    
    # 应用场景装饰器（如果检测到）
    if scene_info['scene_name']:
        from .decorators import require_scene
        test_func = require_scene(scene_info['scene_name'])(test_func)
        test_func._auto_detected_scene = scene_info['scene_name']
        logger.info(f"Auto-detected scene for {test_func.__name__}: {scene_info['scene_name']}")
    
    # 应用模型装饰器（如果检测到）
    if scene_info['model_name']:
        from .decorators import require_model
        test_func = require_model(scene_info['model_name'])(test_func)
        test_func._auto_detected_model = scene_info['model_name']
        logger.info(f"Auto-detected model for {test_func.__name__}: {scene_info['model_name']}")
    
    # 记录检测信息
    test_func._scene_info = scene_info
    
    return test_func

class LegacyTestWrapper:
    """
    现有测试包装器 - 包装VllmRunner以适配新框架
    
    功能:
        - 自动从当前场景获取tensor_parallel_size
        - 自动记录场景信息
        - 保持原有功能不变
    """
    
    @staticmethod
    def wrap_vllm_runner(original_runner_class):
        """
        包装VllmRunner类，使其支持场景管理
        
        Args:
            original_runner_class: 原始VllmRunner类
        
        Returns:
            包装后的VllmRunner类
        
        功能:
            - 自动从当前场景获取tensor_parallel_size
            - 自动记录场景信息
            - 保持原有功能不变
        """
        class WrappedVllmRunner(original_runner_class):
            def __init__(self, *args, **kwargs):
                # 从场景管理器获取当前场景
                from .scene_manager import SceneManager
                scene_manager = SceneManager()
                current_scene = scene_manager.get_current_scene()
                
                # 根据场景自动调整参数
                if current_scene:
                    # 自动设置tensor_parallel_size（如果未指定）
                    if 'tensor_parallel_size' not in kwargs:
                        kwargs['tensor_parallel_size'] = current_scene.card_count
                    
                    # 记录场景信息
                    logger.info(
                        f"Creating VllmRunner in scene {current_scene.name}, "
                        f"tensor_parallel_size={kwargs.get('tensor_parallel_size')}"
                    )
                else:
                    logger.warning("No scene is set, using default parameters")
                
                # 调用原始初始化
                super().__init__(*args, **kwargs)
        
        # 保持类名和文档
        WrappedVllmRunner.__name__ = original_runner_class.__name__
        WrappedVllmRunner.__doc__ = original_runner_class.__doc__
        
        return WrappedVllmRunner
    
    @staticmethod
    def wrap_remote_server(original_server_class):
        """
        包装RemoteOpenAIServer类，使其支持场景管理
        
        Args:
            original_server_class: 原始RemoteOpenAIServer类
        
        Returns:
            包装后的RemoteOpenAIServer类
        """
        class WrappedRemoteOpenAIServer(original_server_class):
            def __init__(self, *args, **kwargs):
                from .scene_manager import SceneManager
                scene_manager = SceneManager()
                current_scene = scene_manager.get_current_scene()
                
                if current_scene:
                    logger.info(f"Starting RemoteOpenAIServer in scene {current_scene.name}")
                else:
                    logger.warning("No scene is set for RemoteOpenAIServer")
                
                super().__init__(*args, **kwargs)
        
        WrappedRemoteOpenAIServer.__name__ = original_server_class.__name__
        WrappedRemoteOpenAIServer.__doc__ = original_server_class.__doc__
        
        return WrappedRemoteOpenAIServer

def auto_patch_conftest(conftest_module):
    """
    自动包装conftest中的Runner类
    
    Args:
        conftest_module: conftest模块对象（tests.e2e.conftest）
    
    功能:
        - 包装VllmRunner
        - 包装RemoteOpenAIServer
        - 包装DPVllmRunner
        - 保持向后兼容
    
    注意: 需要在pytest_configure中调用此函数
    """
    # 包装VllmRunner
    if hasattr(conftest_module, 'VllmRunner'):
        original_vllm_runner = conftest_module.VllmRunner
        conftest_module.VllmRunner = LegacyTestWrapper.wrap_vllm_runner(
            original_vllm_runner
        )
        # 保持引用以便其他模块使用
        conftest_module._OriginalVllmRunner = original_vllm_runner
        logger.info("VllmRunner wrapped successfully")
    
    # 包装RemoteOpenAIServer
    if hasattr(conftest_module, 'RemoteOpenAIServer'):
        original_remote_server = conftest_module.RemoteOpenAIServer
        conftest_module.RemoteOpenAIServer = LegacyTestWrapper.wrap_remote_server(
            original_remote_server
        )
        conftest_module._OriginalRemoteOpenAIServer = original_remote_server
        logger.info("RemoteOpenAIServer wrapped successfully")
    
    # 包装DPVllmRunner
    if hasattr(conftest_module, 'DPVllmRunner'):
        original_dp_runner = conftest_module.DPVllmRunner
        conftest_module.DPVllmRunner = LegacyTestWrapper.wrap_vllm_runner(
            original_dp_runner
        )
        conftest_module._OriginalDPVllmRunner = original_dp_runner
        logger.info("DPVllmRunner wrapped successfully")
    
    # 包装HfRunner
    if hasattr(conftest_module, 'HfRunner'):
        # HfRunner不需要特殊包装，但可以记录场景信息
        logger.info("HfRunner detected, no wrapping needed")

def create_compatible_runner(original_runner_class, scene_manager):
    """
    创建兼容的Runner实例
    
    Args:
        original_runner_class: 原始Runner类
        scene_manager: 场景管理器实例
    
    Returns:
        Runner创建函数
    
    用法:
        runner_factory = create_compatible_runner(VllmRunner, scene_manager)
        runner = runner_factory("Qwen/Qwen3-8B")  # 自动使用当前场景的参数
    """
    def runner_factory(model_name, **kwargs):
        current_scene = scene_manager.get_current_scene()
        
        # 自动调整参数
        if current_scene and 'tensor_parallel_size' not in kwargs:
            kwargs['tensor_parallel_size'] = current_scene.card_count
        
        # 创建Runner实例
        return original_runner_class(model_name, **kwargs)
    
    return runner_factory