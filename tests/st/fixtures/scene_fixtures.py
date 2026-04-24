"""
场景相关pytest fixtures

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

提供场景管理器和当前场景的pytest fixtures
"""
import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def scene_manager():
    """
    场景管理器fixture（session级别）
    
    在整个测试会话期间保持单例状态
    
    Returns:
        SceneManager: 场景管理器实例
    
    用法:
        def test_something(scene_manager):
            scenes = scene_manager.get_all_scenes()
            assert len(scenes) > 0
    """
    from framework.scene_manager import SceneManager
    
    manager = SceneManager()
    logger.info(f"SceneManager initialized with {len(manager.get_all_scenes())} scenes")
    yield manager

@pytest.fixture(scope="session")
def model_loader():
    """
    模型加载器fixture（session级别）
    
    在整个测试会话期间保持单例状态
    会话结束时自动卸载所有模型
    
    Returns:
        ModelLoader: 模型加载器实例
    
    用法:
        def test_something(model_loader):
            models = model_loader.get_all_models()
            assert len(models) > 0
    """
    from framework.model_loader import ModelLoader
    
    loader = ModelLoader()
    logger.info(f"ModelLoader initialized with {len(loader.get_all_models())} models")
    
    yield loader
    
    # 清理：卸载所有模型
    loader.unload_all_models()
    logger.info("All models unloaded")

@pytest.fixture(scope="session")
def current_scene(request, scene_manager):
    """
    当前场景fixture（session级别）
    
    通过命令行参数 --scene 指定场景
    如果未指定，自动检测或使用默认场景
    
    Args:
        request: pytest request对象
        scene_manager: 场景管理器fixture
    
    Returns:
        Scene: 当前场景对象
    
    用法:
        def test_something(current_scene):
            assert current_scene.name == "SINGLE_CARD_310P"
            assert current_scene.card_count == 1
    
    命令行使用:
        pytest --scene=SINGLE_CARD_310P
    """
    # 从命令行获取场景参数
    scene_name = request.config.getoption("--scene", default=None)
    
    if scene_name:
        # 用户指定了场景
        if scene_manager.set_current_scene(scene_name):
            scene = scene_manager.get_current_scene()
            logger.info(f"Running tests with scene: {scene_name}")
            yield scene
        else:
            pytest.fail(f"Invalid scene: {scene_name}. Use --list-scenes to see available scenes.")
    else:
        # 未指定场景，尝试自动检测
        scene = _auto_detect_scene(scene_manager, request)
        if scene:
            logger.info(f"Auto-detected scene: {scene.name}")
            yield scene
        else:
            # 使用默认场景
            default_scene = scene_manager.get_scene("SINGLE_CARD_310P")
            if default_scene:
                scene_manager.set_current_scene(default_scene.name)
                logger.info(f"Using default scene: {default_scene.name}")
                yield default_scene
            else:
                pytest.fail("No scene available. Please specify --scene or check scene configuration.")

def _auto_detect_scene(scene_manager, request):
    """
    自动检测当前场景
    
    Args:
        scene_manager: 场景管理器
        request: pytest request
    
    Returns:
        Scene: 检测到的场景，如果无法检测返回None
    
    检测策略:
    1. 检查环境变量 SCENE_NAME
    2. 检查硬件设备类型（需要torch_npu）
    3. 检查测试文件路径
    """
    import os
    
    # 1. 从环境变量检测
    env_scene = os.getenv("SCENE_NAME")
    if env_scene:
        scene = scene_manager.get_scene(env_scene)
        if scene:
            scene_manager.set_current_scene(env_scene)
            return scene
    
    # 2. 尝试硬件检测（需要torch_npu支持）
    try:
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            # 检测设备类型
            device_name = torch.npu.get_device_name(0)
            device_count = torch.npu.device_count()
            
            # 根据设备推断场景
            if "310P" in device_name or "Ascend310P" in device_name:
                if device_count == 1:
                    inferred_scene = "SINGLE_CARD_310P"
                elif device_count == 2:
                    inferred_scene = "MULTI_CARD_310P_2"
                elif device_count >= 4:
                    inferred_scene = "MULTI_CARD_310P_4"
            elif "910B" in device_name or "Ascend910B" in device_name:
                if device_count == 1:
                    inferred_scene = "SINGLE_CARD_910B"
                elif device_count == 2:
                    inferred_scene = "MULTI_CARD_910B_2"
            else:
                inferred_scene = None
            
            if inferred_scene:
                scene = scene_manager.get_scene(inferred_scene)
                if scene:
                    scene_manager.set_current_scene(inferred_scene)
                    return scene
    except Exception as e:
        logger.debug(f"Hardware detection failed: {e}")
    
    # 3. 从测试路径检测
    if hasattr(request, 'fspath'):
        from framework.scene_detector import SceneDetector
        test_path = str(request.fspath)
        scene_name, scene = SceneDetector.detect_from_path(test_path)
        if scene:
            scene_manager.set_current_scene(scene_name)
            return scene
    
    return None

@pytest.fixture(scope="function")
def loaded_model(request, current_scene, model_loader):
    """
    已加载模型fixture（function级别）
    
    通过命令行参数 --model 指定模型
    如果未指定，使用场景的默认模型
    
    Args:
        request: pytest request对象
        current_scene: 当前场景fixture
        model_loader: 模型加载器fixture
    
    Returns:
        dict: 已加载模型的信息字典
    
    用法:
        def test_inference(loaded_model):
            model = loaded_model['model']
            config = loaded_model['config']
            assert model['status'] == 'loaded'
    
    命令行使用:
        pytest --model=qwen-7b
    """
    # 从命令行获取模型参数
    model_name = request.config.getoption("--model", default=None)
    
    # 检查测试函数是否指定了模型要求
    required_models = getattr(request.function, '_required_models', None)
    
    # 确定要加载的模型
    if not model_name:
        if required_models:
            # 使用装饰器指定的第一个模型
            model_name = required_models.split('|')[0]
            logger.info(f"Using model from decorator: {model_name}")
        elif current_scene and current_scene.models:
            # 使用场景的默认模型（第一个）
            model_name = current_scene.models[0]
            logger.info(f"Using scene's default model: {model_name}")
        else:
            pytest.skip("No model specified and scene has no default model")
            return None
    
    # 检查场景是否支持该模型
    if current_scene and not current_scene.supports_model(model_name):
        pytest.skip(
            f"Scene {current_scene.name} does not support model {model_name}. "
            f"Supported models: {', '.join(current_scene.models)}"
        )
        return None
    
    # 检查模型配置是否存在
    model_config = model_loader.get_model_config(model_name)
    if not model_config:
        pytest.skip(f"Model config not found: {model_name}. Use --list-models to see available models.")
        return None
    
    # 加载模型（模拟）
    model = model_loader.load_model(model_name, {'scene': current_scene.name if current_scene else None})
    
    yield {
        'model': model,
        'config': model_config,
        'name': model_name
    }
    
    # 清理（可选，如果需要每个测试后卸载）
    # model_loader.unload_model(model_name)

@pytest.fixture(scope="session")
def scene_list(scene_manager):
    """
    场景列表fixture（session级别）
    
    用于需要访问所有场景信息的测试
    
    Args:
        scene_manager: 场景管理器fixture
    
    Returns:
        list: 所有场景列表
    
    用法:
        def test_scene_coverage(scene_list):
            assert len(scene_list) >= 9
    """
    return scene_manager.get_all_scenes()

@pytest.fixture(scope="session") 
def model_list(model_loader):
    """
    模型列表fixture（session级别）
    
    用于需要访问所有模型信息的测试
    
    Args:
        model_loader: 模型加载器fixture
    
    Returns:
        list: 所有模型列表
    
    用法:
        def test_model_coverage(model_list):
            assert len(model_list) >= 8
    """
    return model_loader.get_all_models()